import os
import math
import itertools
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
        self.short_period_max = int(os.environ.get("TIER1_SHORT_PERIOD_MAX") or "0")
        self.short_min_array_length = int(
            os.environ.get("TIER1_SHORT_MIN_ARRAY_LEN") or str(self.min_array_length))
        self.short_min_score = float(
            os.environ.get("TIER1_SHORT_MIN_SCORE") or str(self.min_score))
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

        # --- FM-index motif-enumeration STR detector (new, opt-in) -----------
        # A statistically-gated, gap-tolerant short-STR detector that finds
        # diverged arrays the exact adjacent-copy seeder misses. Default OFF
        # (TIER1_FMSCAN=0) so the existing pipeline is byte-for-byte unchanged.
        #   0 = off (baseline)
        #   1 = run as an ADDITIONAL source merged with the sliding-window scan
        #   2 = run as a REPLACEMENT for the sliding-window scan
        self.fmscan_mode = int(os.environ.get("TIER1_FMSCAN") or "0")
        # Period range for the FM-index detector (independent of the global
        # tier1 range so it can be restricted for proof-of-concept / speed).
        self.fmscan_min_p = int(os.environ.get("TIER1_FMSCAN_MIN_P") or "1")
        self.fmscan_max_p = int(os.environ.get("TIER1_FMSCAN_MAX_P") or "6")
        # Gap tolerance: a periodic run may skip up to this many consecutive
        # expected copies (each skip = a diverged/mismatched copy) and still be
        # treated as one array.
        self.fmscan_max_gap_copies = int(os.environ.get("TIER1_FMSCAN_MAX_GAP") or "2")
        # Minimum observed perfect-motif occurrences (anchors) in a run.
        self.fmscan_min_occ = int(os.environ.get("TIER1_FMSCAN_MIN_OCC") or "3")
        # Minimum reported array span (bp) for an accepted run.
        self.fmscan_min_span = int(os.environ.get("TIER1_FMSCAN_MIN_SPAN") or "20")
        # Density floor: observed occurrences / expected copies (span/p). A
        # perfect array -> 1.0; diverged arrays sit a bit below 1.0; random
        # low-complexity stretches sit well below. THIS is the discriminator.
        # Default 0.50 = the validated operating point (see docstring / docs):
        # additive mode at density=0.50, llr=8 dominates the gate-tuning OP1 on
        # chr21 AND chr22 (more recall at equal adjusted precision).
        self.fmscan_min_density = float(os.environ.get("TIER1_FMSCAN_MIN_DENSITY") or "0.50")
        # Significance floor: Poisson log-likelihood ratio of the observed
        # perfect-copy count vs the i.i.d. background expectation. Accept only
        # runs whose LLR exceeds this. Higher = stricter / higher precision.
        # Default 8.0 = the validated operating point.
        self.fmscan_min_llr = float(os.environ.get("TIER1_FMSCAN_MIN_LLR") or "8.0")
        # Skip motifs with more than this many genome-wide occurrences (pure
        # low-complexity floods, e.g. poly-A); they cannot be localized cheaply.
        self.fmscan_max_occ_total = int(os.environ.get("TIER1_FMSCAN_MAX_OCC_TOTAL") or "20000000")

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

        # New FM-index motif-enumeration detector (opt-in).
        #   mode 2 = REPLACEMENT: the sole Tier-1 source (return immediately).
        #   mode 1 = ADDITIVE: run AFTER the sliding-window scan on the regions
        #            it did NOT already claim, so the FM detector only adds the
        #            diverged STRs the exact-seed scanner missed. Running on the
        #            residual keeps the candidate count (and the downstream merge
        #            cost) low and avoids double-reporting the same array.
        fmscan_repeats: List[TandemRepeat] = []
        if self.fmscan_mode == 2:
            fmscan_repeats = self._fmscan_strs(chromosome, sequence_str, n)
            if self.show_progress:
                print(f"  [{chromosome}] Tier 1 FM-index scan (replacement) "
                      f"found {len(fmscan_repeats)} repeats", flush=True)
            return fmscan_repeats

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

        # Additive mode: run the FM-index detector on the residual (regions the
        # sliding-window scan did NOT claim) and append its calls. This recovers
        # diverged short STRs that have no long perfect adjacent run — the class
        # the exact-seed scanner is blind to — while the shared seen_mask keeps
        # it from re-reporting arrays already found, so the merged candidate
        # count stays small.
        if self.fmscan_mode == 1:
            fmscan_repeats = self._fmscan_strs(
                chromosome, sequence_str, n, exclude_mask=seen_mask)
            if fmscan_repeats:
                repeats.extend(fmscan_repeats)

        if self.show_progress:
            print(f"  [{chromosome}] Tier 1 found {len(repeats)} repeats in {time.time() - t0:.2f}s", flush=True)

        return repeats

    # ------------------------------------------------------------------ #
    #  FM-index motif-enumeration STR detector (new, opt-in)             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _primitive_pmers(p: int):
        """Yield every primitive p-mer string over ACGT (all rotations).

        Unlike MotifUtils.enumerate_motifs (which canonicalises away rotations
        and strand), we need EACH distinct rotation because a tandem array
        presents one fixed rotation in the text, and the FM-index query is for
        that exact string.
        """
        for tup in itertools.product("ACGT", repeat=p):
            s = "".join(tup)
            if MotifUtils.is_primitive_motif(s):
                yield s

    @staticmethod
    def _base_freqs(sequence_str: str, n: int):
        """Per-base frequencies of A/C/G/T over the analysed sequence."""
        counts = {"A": 0, "C": 0, "G": 0, "T": 0}
        # Sample to keep this O(1)-ish on huge chromosomes; composition is stable.
        step = max(1, n // 2_000_000)
        tot = 0
        for i in range(0, n, step):
            c = sequence_str[i]
            if c in counts:
                counts[c] += 1
                tot += 1
        if tot == 0:
            return {b: 0.25 for b in "ACGT"}
        return {b: counts[b] / tot for b in "ACGT"}

    def _periodic_runs_from_positions(self, pos, p):
        """Group sorted occurrence positions of one p-mer into gap-tolerant runs.

        A run is a maximal chain where each successive occurrence sits at
        last + g*p for 1 <= g <= max_gap_copies+1 (g-1 skipped copies = diverged
        copies absorbed). Vectorised with NumPy so it scales to the millions of
        occurrences a dinucleotide has on a whole chromosome.

        Parameters
        ----------
        pos : np.ndarray (sorted int64)
            Occurrence positions of one motif.
        p : int
            Period.

        Returns
        -------
        list[(run_start, last_pos, n_occ)]
        """
        m = pos.size
        if m < self.fmscan_min_occ:
            return []
        max_step = (self.fmscan_max_gap_copies + 1) * p
        diffs = pos[1:] - pos[:-1]
        # A link continues a run iff the step is a positive multiple of p within
        # the gap budget. (pos sorted -> diffs >= 0; duplicate positions, diff 0,
        # break the run but are harmless.)
        ok = (diffs > 0) & (diffs <= max_step) & ((diffs % p) == 0)
        # Run boundaries: a new run starts at index 0 and wherever the previous
        # link is broken.
        breaks = np.empty(m, dtype=bool)
        breaks[0] = True
        breaks[1:] = ~ok
        starts_idx = np.flatnonzero(breaks)
        ends_idx = np.empty_like(starts_idx)
        ends_idx[:-1] = starts_idx[1:] - 1
        ends_idx[-1] = m - 1
        occ_counts = ends_idx - starts_idx + 1
        keep = occ_counts >= self.fmscan_min_occ
        if not keep.any():
            return []
        s_keep = starts_idx[keep]
        e_keep = ends_idx[keep]
        o_keep = occ_counts[keep]
        run_starts = pos[s_keep]
        last_positions = pos[e_keep]
        return list(zip(run_starts.tolist(), last_positions.tolist(), o_keep.tolist()))

    def _fmscan_strs(self, chromosome: str, sequence_str: str, n: int,
                     exclude_mask=None) -> List[TandemRepeat]:
        """Detect diverged short STRs by enumerating motifs and statistically
        gating gap-tolerant periodic runs of their FM-index occurrences.

        Periods are processed longest-first; within each period all motif runs
        that clear the density + Poisson-LLR gates are collected, ranked by LLR,
        and greedily claim genomic territory via a ``seen_mask``. This means each
        array is reported once by its single best-supported motif/phase rather
        than once per overlapping rotation, which keeps the candidate count low
        (avoids the O(n^2) blow-up in downstream merge/overlap filtering) and
        raises precision.

        Parameters
        ----------
        exclude_mask : np.ndarray or None
            If given (additive mode), runs whose core is already covered by this
            boolean mask are skipped, so the FM detector only adds arrays the
            sliding-window scanner missed.
        """
        t0 = time.time()
        bwt = self.bwt
        repeats: List[TandemRepeat] = []
        min_p = max(1, self.fmscan_min_p, self.min_motif_length)
        max_p = min(9, self.fmscan_max_p, self.max_motif_length)
        if min_p > max_p:
            return repeats

        freqs = self._base_freqs(sequence_str, n)
        log_pb = {b: math.log(max(freqs[b], 1e-9)) for b in "ACGT"}
        n_queries = 0
        # Greedy territory claim: a run is dropped if its core is already owned
        # by a longer/better motif (this period scan) OR by the sliding-window
        # scan (exclude_mask in additive mode). Start from the exclude mask so
        # already-claimed regions are off-limits.
        if exclude_mask is not None:
            seen_mask = exclude_mask.copy()
        else:
            seen_mask = np.zeros(n, dtype=bool)

        # Longest period first (a true period-6 array would otherwise also be
        # picked up as period-2/3 sub-runs).
        for p in range(max_p, min_p - 1, -1):
            # Gather every qualifying (LLR, run_start, run_end, occ, motif) for
            # this period before claiming space, so the highest-LLR runs win.
            period_runs = []
            for motif in self._primitive_pmers(p):
                log_p_bg = sum(log_pb[b] for b in motif)
                sp, ep = bwt.backward_search(motif)
                n_queries += 1
                if sp < 0:
                    continue
                occ_total = ep - sp + 1
                if occ_total < self.fmscan_min_occ:
                    continue
                if occ_total > self.fmscan_max_occ_total:
                    continue  # safety valve for pathological motifs only

                positions = np.sort(bwt.suffix_array[sp:ep + 1].astype(np.int64))

                for run_start, last_pos, occ in self._periodic_runs_from_positions(positions, p):
                    span = (last_pos + p) - run_start
                    if span < self.fmscan_min_span:
                        continue
                    expected_copies = span / p
                    if expected_copies <= 0:
                        continue
                    density = occ / expected_copies
                    if density < self.fmscan_min_density:
                        continue
                    # Poisson log-likelihood ratio of the observed perfect-copy
                    # count vs the i.i.d. background expectation for this motif.
                    # Under background, observing `occ` copies of a rare p-mer in
                    # `span` bp is improbable; LLR grows with both excess count
                    # and motif rarity. THIS is the discriminator crude gates
                    # lack: it separates diverged-but-real arrays from random
                    # low-complexity DNA.
                    exp_bg = span * math.exp(log_p_bg)
                    if exp_bg <= 0:
                        exp_bg = 1e-9
                    if occ > 0:
                        llr = occ * math.log(occ / exp_bg) - (occ - exp_bg)
                    else:
                        llr = 0.0
                    if llr < self.fmscan_min_llr:
                        continue
                    period_runs.append((llr, int(run_start), int(last_pos + p),
                                        int(occ), motif))

            # Claim space best-first within this period.
            period_runs.sort(key=lambda x: x[0], reverse=True)
            for llr, run_start, run_end, occ, motif in period_runs:
                core_mid = (run_start + run_end) // 2
                if seen_mask[run_start] or seen_mask[min(core_mid, n - 1)] \
                        or seen_mask[min(run_end - 1, n - 1)]:
                    continue  # core already owned by a better/longer motif

                # Mismatch-tolerant boundary extension to absorb diverged
                # flanking copies, then refine to a primitive TandemRepeat.
                arr_start = run_start
                arr_end = run_end
                ext_res = extend_with_mismatches(
                    self.text_arr, arr_start, p, n, self.allowed_mismatch_rate
                )
                if ext_res is not None:
                    _as, _ae, _ec, full_s, full_e = ext_res
                    if full_e - full_s >= arr_end - arr_start:
                        arr_start, arr_end = full_s, full_e

                motif_check = sequence_str[arr_start:arr_start + p]
                if "$" in motif_check or "N" in motif_check:
                    continue

                refined = MotifUtils.refine_repeat(
                    sequence_str, arr_start, arr_end, motif,
                    mismatch_fraction=self.allowed_mismatch_rate,
                    indel_fraction=self.allowed_indel_rate,
                    min_copies=self.min_copies,
                )
                if not refined:
                    continue
                rep = self._build_repeat(chromosome, refined, tier=1)
                if (rep.end - rep.start) < self.fmscan_min_span:
                    continue
                repeats.append(rep)
                # Claim the run core (not the extension, which may over-reach
                # into a neighbouring array of a different period).
                seen_mask[run_start:min(run_end, n)] = True

        if self.show_progress:
            print(f"  [{chromosome}] Tier 1 FM-index scan: {len(repeats)} runs from "
                  f"{n_queries} motif queries (p={min_p}-{max_p}) in "
                  f"{time.time() - t0:.2f}s", flush=True)
        return repeats
