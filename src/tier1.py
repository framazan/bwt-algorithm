import os
import numpy as np
import ctypes
import time
from typing import List
from .models import TandemRepeat
from .motif_utils import MotifUtils
from .bwt_core import BWTCore
from .accelerators import extend_with_mismatches

# Try to load C extension for fast run detection
_c_lib = None
try:
    from .c_extensions.build import load as _load_c_lib
    _c_lib = _load_c_lib()
except Exception:
    pass


class Tier1STRFinder:
    """Tier 1: Short Tandem Repeat Finder (1-9bp) using sliding window scan.

    Directly scans the sequence for consecutive period-k repeats using
    character comparison, then extends seeds with mismatch tolerance.
    Much faster than FM-index enumeration of all possible k-mers.
    """

    def __init__(self, text_arr: np.ndarray, bwt_core: BWTCore, max_motif_length: int = 9,
                 min_motif_length: int = 1,
                 allowed_mismatch_rate: float = 0.2, allowed_indel_rate: float = 0.1,
                 show_progress: bool = False):
        self.text_arr = text_arr
        self.bwt = bwt_core
        self.max_motif_length = max(1, max_motif_length)
        self.min_motif_length = max(1, min(min_motif_length, self.max_motif_length))
        # Detection thresholds — tunable via env vars for parameter sweeps.
        # Defaults reproduce the original hardcoded behaviour exactly.
        self.min_copies = int(os.environ.get("TIER1_MIN_COPIES", "3"))
        self.min_array_length = int(os.environ.get("TIER1_MIN_ARRAY_LEN", "26"))
        self.min_entropy = float(os.environ.get("TIER1_MIN_ENTROPY", "1.0"))
        # entropy gate is OFF by default (preserves baseline); opt-in for sweeps
        self.entropy_gate = bool(int(os.environ.get("TIER1_ENTROPY_GATE", "0")))
        self.min_score = float(os.environ.get("TIER1_MIN_SCORE", "30"))
        # Period-stratified relaxation of the length/score acceptance gate.
        # Short motifs (motif_len <= short_period_max) frequently form short
        # perfect cores (e.g. a 7-copy dinucleotide = 14 bp) that sit inside a
        # much larger adotto region but get rejected by the global
        # required_threshold / min_score gates. These knobs let the gate be
        # relaxed ONLY for short motifs while keeping longer motifs strict.
        # Defaults reproduce baseline exactly: short_period_max=0 disables the
        # stratification, and the short thresholds inherit the global ones.
        self.short_period_max = int(os.environ.get("TIER1_SHORT_PERIOD_MAX", "0"))
        self.short_min_array_length = int(
            os.environ.get("TIER1_SHORT_MIN_ARRAY_LEN", str(self.min_array_length)))
        self.short_min_score = float(
            os.environ.get("TIER1_SHORT_MIN_SCORE", str(self.min_score)))
        # dynamic_min_copies = max(min_copies, copy_base // motif_len + copy_add)
        self.copy_base = int(os.environ.get("TIER1_COPYBASE", "12"))
        self.copy_add = int(os.environ.get("TIER1_COPYADD", "3"))
        # perfect seed copies required before mismatch extension is attempted
        self.ext_copies_short = int(os.environ.get("TIER1_EXT_COPIES", "5"))
        # Stitch-seeding: merge adjacent same-period perfect sub-runs separated
        # by a short, phase-aligned gap (<= stitch_gap * motif_len) into a single
        # candidate before extend/refine. Recovers cores fragmented by isolated
        # mismatches that drop individual sub-runs below the copy threshold.
        # Default 0 = disabled (baseline behaviour preserved).
        self.stitch_gap = int(os.environ.get("TIER1_STITCH_GAP", "0"))
        _mm = os.environ.get("TIER1_MISMATCH")
        self.allowed_mismatch_rate = max(0.0, float(_mm) if _mm else allowed_mismatch_rate)
        self.allowed_indel_rate = max(0.0, allowed_indel_rate)
        self.show_progress = show_progress

    def _build_repeat(self, chromosome: str, refined, tier: int = 1) -> TandemRepeat:
        return MotifUtils.refined_to_repeat(chromosome, refined, tier, self.text_arr, strand='+')

    def _stitch_candidates(self, candidates, motif_len, sequence_str, n, seen_mask):
        """Merge phase-aligned adjacent perfect sub-runs of the same period.

        Two candidates are stitched when the next run starts within
        ``stitch_gap * motif_len`` bp of the current run's end, the gap is a
        whole number of motif units (phase-aligned, so both runs share the same
        period frame), and both encode the same motif. Returns a new candidate
        list of ``(start, end, copies)`` tuples; copies are recomputed from the
        merged span. Candidates are assumed sorted by start (the run scanner
        emits them in order).
        """
        max_gap = self.stitch_gap * motif_len
        merged = []
        cur_s, cur_e, _ = candidates[0]
        cur_motif = sequence_str[cur_s:cur_s + motif_len]
        for nxt_s, nxt_e, _ in candidates[1:]:
            gap = nxt_s - cur_e
            phase_aligned = gap >= 0 and (gap % motif_len) == 0
            same_motif = sequence_str[nxt_s:nxt_s + motif_len] == cur_motif
            if gap <= max_gap and phase_aligned and same_motif:
                # Extend the current merged run; skip if the stitch span has
                # been claimed by a longer motif already.
                mid = (cur_s + nxt_e) // 2
                if seen_mask[cur_s] or seen_mask[min(mid, n - 1)]:
                    # Current run is in claimed territory — flush and restart.
                    merged.append((cur_s, cur_e, (cur_e - cur_s) // motif_len))
                    cur_s, cur_e = nxt_s, nxt_e
                    cur_motif = sequence_str[nxt_s:nxt_s + motif_len]
                    continue
                cur_e = nxt_e
            else:
                merged.append((cur_s, cur_e, (cur_e - cur_s) // motif_len))
                cur_s, cur_e = nxt_s, nxt_e
                cur_motif = sequence_str[nxt_s:nxt_s + motif_len]
        merged.append((cur_s, cur_e, (cur_e - cur_s) // motif_len))
        return merged

    def find_strs(self, chromosome: str) -> List[TandemRepeat]:
        t0 = time.time()
        text_arr = self.text_arr
        n = text_arr.size
        sequence_str = text_arr.tobytes().decode('ascii', errors='replace')
        repeats = []

        max_len = min(self.max_motif_length, 9)
        min_len = max(1, self.min_motif_length)
        if min_len > max_len:
            return repeats

        if self.show_progress:
            print(f"  [{chromosome}] Tier 1 sliding window scan (k={min_len}-{max_len})...", flush=True)

        seen_mask = np.zeros(n, dtype=bool)

        # Process longest motifs first so they claim space before shorter ones
        for motif_len in range(max_len, min_len - 1, -1):
            # Period-stratified gate: short motifs may use a relaxed length/score
            # floor (defaults to the global values, so unset == baseline).
            is_short = motif_len <= self.short_period_max
            eff_min_array_length = self.short_min_array_length if is_short else self.min_array_length
            eff_min_score = self.short_min_score if is_short else self.min_score
            # Dynamic min copies: shorter motifs need more copies
            dynamic_min_copies = max(self.min_copies, self.copy_base // motif_len + self.copy_add)
            required_threshold = max(eff_min_array_length, motif_len * dynamic_min_copies)
            min_run = required_threshold // motif_len  # minimum consecutive matching positions

            seed_min_copies = 2
            max_candidates = min(n // motif_len + 1, 1_000_000)

            # Use C extension for fast run detection if available
            if _c_lib is not None:
                text_ptr = text_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
                seen_ptr = seen_mask.view(np.uint8).ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
                out_starts = (ctypes.c_int * max_candidates)()
                out_ends = (ctypes.c_int * max_candidates)()
                out_copies = (ctypes.c_int * max_candidates)()
                n_found = _c_lib.find_period_runs(
                    text_ptr, n, motif_len, seed_min_copies,
                    seen_ptr, out_starts, out_ends, out_copies, max_candidates
                )
                candidates = [(out_starts[ci], out_ends[ci], out_copies[ci])
                              for ci in range(n_found)]
            else:
                # Pure Python fallback
                match_arr = (text_arr[:n - motif_len] == text_arr[motif_len:n])
                candidates = []
                i = 0
                limit = n - motif_len
                while i < limit:
                    if not match_arr[i]:
                        i += 1
                        continue
                    run_start = i
                    j = i + 1
                    while j < limit and match_arr[j]:
                        j += 1
                    array_start = run_start
                    array_end = j + motif_len
                    seed_copies = (array_end - array_start) // motif_len
                    i = j
                    if seed_copies < seed_min_copies:
                        continue
                    mid = (array_start + array_end) // 2
                    if seen_mask[array_start] or seen_mask[min(mid, n - 1)]:
                        continue
                    motif_check = sequence_str[array_start:array_start + motif_len]
                    if '$' in motif_check or 'N' in motif_check:
                        continue
                    candidates.append((array_start, array_end, seed_copies))

            # Stitch-seeding: merge phase-aligned adjacent perfect sub-runs of
            # this period that are separated by a short gap. Fragmented cores
            # (split by isolated mismatches) are rejoined so the combined span
            # clears the length/copy gates. No-op when stitch_gap == 0.
            if self.stitch_gap > 0 and len(candidates) > 1:
                candidates = self._stitch_candidates(
                    candidates, motif_len, sequence_str, n, seen_mask)

            for array_start, array_end, seed_copies in candidates:
                seed_length = array_end - array_start

                # Extract motif, skip invalid
                motif = sequence_str[array_start:array_start + motif_len]
                if MotifUtils.smallest_period_str(motif) < motif_len:
                    continue

                # Extend with mismatch tolerance to capture imperfect copies
                perfect_length = seed_length
                ext_start = array_start
                ext_end = array_end
                ext_length = seed_length
                ext_copies = seed_copies
                min_copies_for_ext = self.ext_copies_short if motif_len <= 3 else 2
                if seed_copies >= min_copies_for_ext:
                    ext_res = extend_with_mismatches(
                        text_arr, array_start, motif_len, n,
                        self.allowed_mismatch_rate
                    )
                    if ext_res is not None:
                        arr_s, arr_e, ec, full_s, full_e = ext_res
                        if full_e - full_s > seed_length:
                            ext_start = full_s
                            ext_end = full_e
                            ext_length = ext_end - ext_start
                            ext_copies = ec

                # Check EXTENDED length against the real threshold
                if ext_length < required_threshold or ext_copies < dynamic_min_copies:
                    continue

                entropy = MotifUtils.calculate_entropy(motif)

                # Optional low-complexity quality gate (opt-in; default off so
                # baseline behaviour is preserved). Suppresses spurious calls on
                # low-entropy motifs when aggressive length/score thresholds are used.
                if self.entropy_gate and entropy < self.min_entropy:
                    continue

                refined = MotifUtils.refine_repeat(
                    sequence_str,
                    ext_start,
                    ext_end,
                    motif,
                    mismatch_fraction=self.allowed_mismatch_rate,
                    indel_fraction=self.allowed_indel_rate,
                    min_copies=self.min_copies
                )

                # If extended region was rejected, fall back to seed region
                if refined is None and ext_start != array_start:
                    refined = MotifUtils.refine_repeat(
                        sequence_str,
                        array_start,
                        array_start + perfect_length,
                        motif,
                        mismatch_fraction=self.allowed_mismatch_rate,
                        indel_fraction=self.allowed_indel_rate,
                        min_copies=self.min_copies
                    )

                if refined:
                    rep = self._build_repeat(chromosome, refined, tier=1)
                    # Quality filter: score = length * (1 - mismatch_rate) must be >= 30
                    rep_score = (rep.end - rep.start) * (1.0 - rep.mismatch_rate)
                    if rep_score < eff_min_score:
                        continue
                    repeats.append(rep)
                    seed_end = min(array_start + perfect_length, n)
                    seen_mask[array_start:seed_end] = True

        if self.show_progress:
            print(f"  [{chromosome}] Tier 1 found {len(repeats)} repeats in {time.time() - t0:.2f}s", flush=True)

        return repeats
