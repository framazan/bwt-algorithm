import math
import ctypes
import os

import numpy as np
from typing import List, Tuple, Set, Optional
import time
from .models import TandemRepeat
from .motif_utils import MotifUtils
from .bwt_core import BWTCore, _kasai_lcp_uint8
from .accelerators import extend_with_mismatches, lcp_tandem_candidates, find_tandem_runs
from .bwt_seed import bwt_kmer_seed_scan

# Load Tier 2 C acceleration library
try:
    from .c_extensions.build import load_tier2 as _load_tier2
    _c_tier2 = _load_tier2()
except Exception:
    _c_tier2 = None


def _c_smallest_period_str(motif_bytes: bytes) -> int:
    """Fast C-accelerated smallest period detection."""
    if _c_tier2 is None:
        return MotifUtils.smallest_period_str(motif_bytes.decode('ascii', errors='replace'))
    n = len(motif_bytes)
    arr = (ctypes.c_ubyte * n)(*motif_bytes)
    return _c_tier2.smallest_period_str(arr, n)


def _c_smallest_period_str_approx(motif_bytes: bytes, max_error_rate: float = 0.02) -> int:
    """Fast C-accelerated approximate period detection."""
    if _c_tier2 is None:
        return MotifUtils.smallest_period_str_approx(
            motif_bytes.decode('ascii', errors='replace'), max_error_rate=max_error_rate)
    n = len(motif_bytes)
    arr = (ctypes.c_ubyte * n)(*motif_bytes)
    return _c_tier2.smallest_period_str_approx(arr, n, ctypes.c_double(max_error_rate))


def _reduce_to_primitive(motif_bytes: bytes) -> str:
    """Reduce motif bytes to primitive period motif string."""
    primitive_period = _c_smallest_period_str(motif_bytes)
    if primitive_period == len(motif_bytes):
        primitive_period = _c_smallest_period_str_approx(motif_bytes, 0.02)
    return motif_bytes[:primitive_period].decode('ascii', errors='replace')


class Tier2LCPFinder:
    """Tier 2: BWT/FM-index based repeat finder for ALL motif lengths with imperfect repeat support.

    Handles both short repeats (with mismatches) and medium/long repeats using:
    - FM-index backward search for motif occurrences
    - LCP arrays for longer period detection
    - Seed-and-extend with mismatch tolerance
    """

    def __init__(self, bwt_core: BWTCore, min_period: int = 1, max_period: int = 1000,
                 max_short_motif: int = 9, allow_mismatches: bool = True,
                 allowed_mismatch_rate: float = 0.2, allowed_indel_rate: float = 0.1,
                 show_progress: bool = False):
        self.bwt = bwt_core
        # Tier 2 is designed for non-microsatellite motifs; enforce >=10bp
        self.min_period = max(10, min_period)
        self.max_period = max_period
        self.max_short_motif = max_short_motif  # For FM-index search (1-9bp)
        # Detection thresholds — tunable via env vars for parameter sweeps.
        # Defaults reproduce the original hardcoded behaviour exactly.
        self.min_copies = int(os.environ.get("TIER2_MIN_COPIES", "3"))  # Require at least 3 copies
        self.min_array_length = int(os.environ.get("TIER2_MIN_ARRAY_LEN", "6"))
        self.min_entropy = 1.0  # Minimum Shannon entropy
        # required copies for short (<20bp) periods in the simple scan
        self.short_req_copies = int(os.environ.get("TIER2_SHORT_REQ_COPIES", "3"))
        # Approximate (autocorrelation) seeding for the diverged period-10-20
        # notch — opt-in; default off reproduces baseline exactly. Detects local
        # periodicity directly instead of requiring an exact seed (LCP plateau or
        # exact k-mer recurrence), which diverged arrays never form.
        self.approx_seed = int(os.environ.get("TIER2_APPROX_SEED") or "0")
        self.approx_min_identity = float(os.environ.get("TIER2_APPROX_MIN_IDENTITY") or "0.78")
        self.approx_max_period = int(os.environ.get("TIER2_APPROX_MAX_P") or "20")
        self.approx_min_copies = int(os.environ.get("TIER2_APPROX_MIN_COPIES") or "2")
        self.allow_mismatches = allow_mismatches
        self.show_progress = show_progress
        self.period_step = 1  # Step size for period scanning (increase to speed up)
        _mm = os.environ.get("TIER2_MISMATCH")
        self.allowed_mismatch_rate = max(0.0, float(_mm) if _mm else allowed_mismatch_rate)
        self.allowed_indel_rate = max(0.0, allowed_indel_rate)
        # Maximum period for Phase A's per-copy seed-and-extend + banded-DP
        # refinement.  Phase A's aligner is O(period^2) per copy; on dense
        # satellite arrays the LCP scan emits huge spurious periods (integer
        # multiples of the true unit, up to ~max_period) that never yield a valid
        # repeat yet dominate runtime/memory (observed ~30 GB / OOM on a 3 Mb
        # alpha-HOR array).  Periods above this bound are skipped in Phase A —
        # genuine long units are recovered by Phase B (BWT k-mer seeding) and
        # Tier 3 (anchor-based), which scale to long periods.  Tunable; the
        # default comfortably covers real tandem units (e.g. the ~2 kb alpha HOR)
        # with several-fold margin.
        self.long_unit_dp_max_period = int(
            os.environ.get("TIER2_LONGUNIT_DP_MAX_PERIOD", "8192")
        )
        self.sequence_str = self.bwt.text_arr.tobytes().decode('ascii', errors='replace')
        self._lcp_cache = None  # Lazily computed LCP array
        self._bwt_call_count = 0  # Track BWT usage for diagnostics

    def _get_lcp_array(self) -> np.ndarray:
        """Get (or compute and cache) the LCP array."""
        if self._lcp_cache is None:
            self._lcp_cache = self._compute_lcp_array()
        return self._lcp_cache

    def _refine_and_create_repeat(self, chromosome: str, start: int, end: int, motif: str,
                                  tier: int, min_copies: Optional[int] = None,
                                  strand: str = '+') -> Optional[TandemRepeat]:
        if min_copies is None:
            min_copies = self.min_copies

        refined = MotifUtils.refine_repeat(
            self.sequence_str,
            start,
            end,
            motif,
            mismatch_fraction=self.allowed_mismatch_rate,
            indel_fraction=self.allowed_indel_rate,
            min_copies=min_copies
        )

        if not refined:
            return None

        return MotifUtils.refined_to_repeat(chromosome, refined, tier, self.bwt.text_arr, strand=strand)

    def find_long_unit_repeats_strict(self, chromosome: str, min_unit_len: int = 20,
                                      max_unit_len: int = 120, max_mismatch: int = 2,
                                      min_copies: int = 2) -> List[TandemRepeat]:
        """Find long-unit tandem repeats using BWT/LCP-driven detection with brute-force fallback.

        Phase A: Uses suffix array and LCP array to identify tandem repeat
                 signatures, then FM-index backward_search/locate_positions
                 to find all occurrences and extend with mismatch tolerance.
        Phase B: Falls back to brute-force sliding window on regions not
                 covered by Phase A.

        Args:
            chromosome: Chromosome name
            min_unit_len: Minimum unit length to consider (default 20bp)
            max_unit_len: Maximum unit length to scan (default 120bp)
            max_mismatch: Maximum Hamming distance per unit comparison (default 2)
            min_copies: Minimum number of adjacent copies required (default 2 for long units)

        Returns:
            List of long-unit tandem repeats
        """
        repeats = []
        text_arr = self.bwt.text_arr
        n = int(text_arr.size)

        # Exclude sentinel
        if n > 0 and text_arr[n - 1] == 36:  # '$' = 36
            n -= 1

        covered: Set[Tuple[int, int]] = set()

        # ===== Phase A: LCP-driven detection =====
        # Per the plan: scan LCP array for plateaus, inspect SA value
        # differences to infer candidate periods, validate with mismatch tolerance.
        # min_lcp_threshold=10 catches imperfect repeats (two copies sharing
        # at least 10bp of common prefix, even if the motif is much longer).
        if self.show_progress:
            print(f"  [{chromosome}] Tier 2 long-unit: Phase A (SA+LCP) starting...", flush=True)

        lcp = self._get_lcp_array()
        sa = self.bwt.suffix_array.astype(np.int32, copy=False)

        candidates = lcp_tandem_candidates(
            sa, lcp, n, min_unit_len, max_unit_len, min_lcp_threshold=10
        )

        # Group candidates by period
        period_seeds: dict = {}
        for period, start_pos in candidates:
            if period not in period_seeds:
                period_seeds[period] = []
            period_seeds[period].append(start_pos)

        covered_mask = np.zeros(n, dtype=bool)
        bwt_found = 0

        # Process each period (longest first for proper nesting)
        for period in sorted(period_seeds.keys(), reverse=True):
            # Skip pathologically large periods: Phase A's per-copy banded DP is
            # O(period^2) and such periods (spurious LCP multiples on satellite
            # arrays) blow up runtime/memory without producing valid repeats.
            # Long real units are handled by Phase B and Tier 3.
            if period > self.long_unit_dp_max_period:
                continue
            seeds = sorted(set(period_seeds[period]))

            for seed_pos in seeds:
                if seed_pos + period > n:
                    continue
                if covered_mask[seed_pos]:
                    continue

                # Validate with mismatch-tolerant extension
                res = extend_with_mismatches(
                    text_arr, seed_pos, period, n,
                    self.allowed_mismatch_rate
                )

                if res is None:
                    continue

                arr_start, arr_end, copies, full_start, full_end = res
                if copies < min_copies:
                    continue

                # Tier 2 post-processing: primitive period reduction
                motif_bytes = text_arr[full_start:full_start + period].tobytes()
                motif = _reduce_to_primitive(motif_bytes)

                repeat = self._refine_and_create_repeat(
                    chromosome, full_start, full_end, motif,
                    tier=2, min_copies=max(2, min_copies)
                )
                if repeat:
                    repeats.append(repeat)
                    covered.add((repeat.start, repeat.end))
                    covered_mask[repeat.start:min(repeat.end, n)] = True
                    bwt_found += 1

        if self.show_progress:
            print(f"  [{chromosome}] Tier 2 long-unit: Phase A found {bwt_found} repeats "
                  f"({len(candidates)} LCP candidates)", flush=True)

        # ===== Phase B: BWT k-mer seeding on uncovered regions =====
        if self.show_progress:
            print(f"  [{chromosome}] Tier 2 long-unit: Phase B (BWT k-mer seeding)...", flush=True)

        kmer_size = min(10, min_unit_len - 1) if min_unit_len > 10 else max(6, min_unit_len - 1)
        seed_candidates = bwt_kmer_seed_scan(
            bwt=self.bwt,
            min_period=min_unit_len,
            max_period=max_unit_len,
            kmer_size=kmer_size,
            stride=10,
            min_copies=min_copies,
            allowed_mismatch_rate=self.allowed_mismatch_rate,
            covered_mask=covered_mask,
            show_progress=False,
            label=f"{chromosome} Tier2-long-B",
        )

        fallback_found = 0
        for cand in seed_candidates:
            motif = _reduce_to_primitive(cand.motif.encode('ascii', errors='replace'))

            repeat = self._refine_and_create_repeat(
                chromosome, cand.start, cand.end, motif,
                tier=2, min_copies=max(2, min_copies)
            )
            if repeat:
                repeats.append(repeat)
                covered.add((repeat.start, repeat.end))
                covered_mask[repeat.start:min(repeat.end, n)] = True
                fallback_found += 1

        if self.show_progress:
            print(f"  [{chromosome}] Tier 2 long-unit: Phase B found {fallback_found} additional repeats", flush=True)

        return repeats

    def _get_max_mismatches_for_array(self, motif_len: int, n_copies: int) -> int:
        """Calculate maximum allowed mismatches for full array.

        Args:
            motif_len: Length of the motif/period
            n_copies: Number of tandem copies

        Returns:
            Maximum allowed mismatches across entire repeat array
        """
        total_length = motif_len * n_copies

        if motif_len == 1:
            return 0

        allowed_rate = max(0.01, min(0.5, self.allowed_mismatch_rate))
        return max(1, int(np.ceil(allowed_rate * total_length)))
    
    def find_long_repeats(self, chromosome: str, tier1_seen: Optional[Set[Tuple[int, int]]] = None,
                         max_scan_period: Optional[int] = None) -> List[TandemRepeat]:
        """Find medium to long tandem repeats using a lightweight period scan.

        This avoids building large LCP structures and is fast for moderate sequences.

        Args:
            chromosome: Chromosome name
            tier1_seen: Set of (start, end) regions already found by Tier1 (to skip)
            max_scan_period: Optional maximum period to scan (overrides default logic)
        """
        return self._find_repeats_simple(chromosome, tier1_seen or set(), max_scan_period)
    
    def _compute_lcp_array(self) -> np.ndarray:
        """Compute LCP array using Kasai over uint8 codes (Numba-accelerated when available)."""
        n = self.bwt.n
        if n == 0:
            return np.array([], dtype=np.int32)
        # Use the text codes directly for fast comparisons
        text_codes = self.bwt.text_arr
        sa = self.bwt.suffix_array.astype(np.int32, copy=False)
        return _kasai_lcp_uint8(text_codes, sa)
    
    def _find_repeats_simple(self, chromosome: str, tier1_seen: Set[Tuple[int, int]],
                            max_scan_period: Optional[int] = None) -> List[TandemRepeat]:
        """BWT/LCP-driven repeat detection with brute-force fallback.

        Phase A: Uses suffix array and LCP array to identify tandem repeat
                 signatures, then FM-index to find all occurrences and extend
                 with mismatch tolerance.
        Phase B: Falls back to brute-force period scanning on uncovered regions.

        Args:
            chromosome: Chromosome name
            tier1_seen: Set of (start, end) regions already found by Tier1 (to skip)
            max_scan_period: Optional maximum period to scan
        """
        s_arr = self.bwt.text_arr
        n = int(s_arr.size)
        # Exclude trailing sentinel if present ('$' == 36)
        if n > 0 and s_arr[n - 1] == 36:
            n -= 1

        # Determine max period
        if max_scan_period is not None:
            max_p = max_scan_period
        else:
            max_p = min(self.max_period, max(1, n // 2))
            if n > 100_000:
                max_p = min(max_p, 500)
            elif n > 10_000:
                max_p = min(max_p, 1000)
            elif n > 1_000:
                max_p = min(max_p, n // 2)
            else:
                max_p = min(max_p, n // 2)

        min_p = min(self.min_period, max_p)
        results: List[TandemRepeat] = []
        seen: Set[Tuple[int, int, str]] = set()
        covered: Set[Tuple[int, int]] = set()

        # Build tier1 mask
        tier1_mask = np.zeros(n, dtype=bool)
        for start, end in tier1_seen:
            tier1_mask[start:min(end, n)] = True

        start_time = time.time()

        # ===== Phase A: LCP-driven detection =====
        if self.show_progress:
            print(f"  [{chromosome}] Tier 2 simple scan: Phase A (SA+LCP) starting, n={n}, min_p={min_p}, max_p={max_p}", flush=True)

        lcp = self._get_lcp_array()
        sa = self.bwt.suffix_array.astype(np.int32, copy=False)

        # Dynamic LCP threshold: shorter periods use lower threshold
        lcp_thresh = max(8, min_p // 2)
        candidates = lcp_tandem_candidates(
            sa, lcp, n, min_p, max_p, min_lcp_threshold=lcp_thresh
        )

        # Group candidates by period, skip tier1-covered positions
        period_seeds: dict = {}
        for period, start_pos in candidates:
            if start_pos < n and tier1_mask[start_pos]:
                continue
            if period not in period_seeds:
                period_seeds[period] = []
            period_seeds[period].append(start_pos)

        covered_mask = tier1_mask.copy()
        bwt_found = 0

        for period in sorted(period_seeds.keys(), reverse=True):
            seeds = sorted(set(period_seeds[period]))

            for seed_pos in seeds:
                if seed_pos + period > n:
                    continue
                if covered_mask[seed_pos]:
                    continue

                res = extend_with_mismatches(
                    s_arr, seed_pos, period, n,
                    self.allowed_mismatch_rate
                )

                if res is None:
                    continue

                arr_start, arr_end, copies, full_start, full_end = res
                required_copies = 2 if period >= 20 else self.short_req_copies
                if copies < required_copies:
                    continue

                motif = _reduce_to_primitive(s_arr[full_start:full_start + period].tobytes())

                # Skip if candidate region already mostly covered
                cand_span = full_end - full_start
                if cand_span > 0:
                    cov_frac = int(np.sum(covered_mask[full_start:min(full_end, n)])) / cand_span
                    if cov_frac > 0.5:
                        continue

                repeat = self._refine_and_create_repeat(
                    chromosome, full_start, full_end, motif, tier=2
                )
                if repeat:
                    results.append(repeat)
                    covered.add((repeat.start, repeat.end))
                    covered_mask[repeat.start:min(repeat.end, n)] = True
                    bwt_found += 1

        if self.show_progress:
            print(f"  [{chromosome}] Tier 2 simple scan: Phase A found {bwt_found} repeats "
                  f"({len(candidates)} LCP candidates) in {time.time() - start_time:.2f}s", flush=True)

        # ===== Phase B: BWT k-mer seeding on uncovered regions =====
        if self.show_progress:
            print(f"  [{chromosome}] Tier 2 simple scan: Phase B (BWT k-mer seeding)...", flush=True)

        kmer_size = min(10, max(6, min_p - 1))
        # Dynamic stride: balance speed vs sensitivity
        seed_stride = min(10, max(5, min_p // 2))
        seed_candidates = bwt_kmer_seed_scan(
            bwt=self.bwt,
            min_period=min_p,
            max_period=max_p,
            kmer_size=kmer_size,
            stride=seed_stride,
            min_copies=2 if min_p >= 20 else self.min_copies,
            allowed_mismatch_rate=self.allowed_mismatch_rate,
            covered_mask=covered_mask,
            show_progress=False,
            label=f"{chromosome} Tier2-simple-B",
        )

        fallback_found = 0
        for cand in seed_candidates:
            # Skip candidates that are mostly in already-covered regions
            cand_len = cand.end - cand.start
            if cand_len > 0:
                cov_count = int(np.sum(covered_mask[cand.start:min(cand.end, n)]))
                if cov_count / cand_len > 0.5:
                    continue

            motif = _reduce_to_primitive(cand.motif.encode('ascii', errors='replace'))

            repeat = self._refine_and_create_repeat(
                chromosome, cand.start, cand.end, motif, tier=2
            )
            if repeat:
                results.append(repeat)
                covered.add((repeat.start, repeat.end))
                covered_mask[repeat.start:min(repeat.end, n)] = True
                fallback_found += 1

        if self.show_progress:
            print(f"  [{chromosome}] Tier 2 simple scan: Phase B found {fallback_found} additional repeats "
                  f"in {time.time() - start_time:.2f}s total", flush=True)

        # ===== Phase C: autocorrelation seeding for the diverged period 10-20 notch =====
        # (opt-in via TIER2_APPROX_SEED). Phases A/B require an exact seed, so an
        # array with a substitution in every copy yields zero seeds. Here we detect
        # local periodicity directly (vectorized identity between offset-p windows),
        # recovering diverged short/medium STRs that the exact paths cannot find.
        if self.approx_seed:
            ac_min_p = max(min_p, 10)
            ac_max_p = min(self.approx_max_period, max_p)
            if ac_max_p >= ac_min_p:
                # Pass the original tier1 mask (not covered_mask): Phase B marks
                # covered_mask for candidates it *speculatively* seeds even when
                # they are then rejected, which would wrongly hide whole diverged
                # arrays from Phase C. Phase C rebuilds exclusion from tier1 +
                # the actually-accepted results.
                approx_found = self._autocorr_seed(
                    s_arr, n, ac_min_p, ac_max_p, tier1_mask,
                    chromosome, results, covered,
                )
                if self.show_progress:
                    print(f"  [{chromosome}] Tier 2 simple scan: Phase C (autocorr) "
                          f"found {approx_found} repeats in {time.time() - start_time:.2f}s total",
                          flush=True)

        return results

    def _autocorr_seed(self, s_arr, n, min_p, max_p, protect_mask,
                       chromosome, results, covered) -> int:
        """Vectorized autocorrelation seeding for periods [min_p, max_p].

        For each period p, identity[i] = fraction of matching bases between the
        window s[i:i+W] and s[i+p:i+p+W] (W = 4*p), computed in O(n) via a cumsum
        sliding window. Positions whose local identity clears
        ``approx_min_identity`` (well above the 0.25 random-DNA expectation) and
        that are not already covered seed ``extend_with_mismatches`` for real
        boundaries; coverage marking keeps one call per array.

        ``protect_mask`` marks positions claimed by earlier tiers; this method
        rebuilds its own exclusion mask from it plus the already-accepted
        ``covered`` repeats, so speculative (and rejected) Phase-B marks do not
        hide diverged arrays.
        """
        found = 0
        min_id = self.approx_min_identity
        s = s_arr[:n]
        # Exclusion mask = other-tier regions + accepted results only.
        seen = protect_mask.copy()
        for cs_, ce_ in covered:
            if 0 <= cs_ < n:
                seen[cs_:min(ce_, n)] = True
        for period in range(min_p, max_p + 1):
            window = 4 * period
            if n <= period + window:
                continue
            eq = (s[:n - period] == s[period:n]).astype(np.int32)  # length n-period
            cs = np.empty(eq.size + 1, dtype=np.int64)
            cs[0] = 0
            np.cumsum(eq, out=cs[1:])
            m = eq.size - window  # number of valid window starts
            if m <= 0:
                continue
            winsum = cs[window:window + m] - cs[:m]      # matches in [i, i+window)
            ident = winsum.astype(np.float64) / window
            cand = np.nonzero(ident >= min_id)[0]
            if cand.size == 0:
                continue
            # Split candidates into contiguous runs and seed each run at its
            # maximum-identity position — a clean fully-periodic window interior,
            # not a boundary where the seed window straddles flank/array and the
            # extender's consensus is corrupted. extend_with_mismatches then
            # recovers the true boundaries in both directions.
            runs = np.split(cand, np.nonzero(np.diff(cand) > 1)[0] + 1)
            for run in runs:
                seed = int(run[int(np.argmax(ident[run]))])
                if seen[seed]:
                    continue
                res = extend_with_mismatches(
                    s_arr, seed, period, n, self.allowed_mismatch_rate
                )
                if res is None:
                    continue
                arr_start, arr_end, copies, full_start, full_end = res
                req = 2 if period >= 20 else self.short_req_copies
                if copies < max(req, self.approx_min_copies):
                    continue
                cand_span = full_end - full_start
                if cand_span <= 0:
                    continue
                cov = int(np.sum(seen[full_start:min(full_end, n)]))
                if cov / cand_span > 0.5:
                    continue
                # Derive the motif from the clean interior seed copy, NOT from
                # full_start: the extender may over-run a few bp into flanking
                # non-repetitive DNA, which would corrupt a full_start-derived
                # motif and make refine_repeat reject the whole array.
                motif = _reduce_to_primitive(
                    s_arr[seed:seed + period].tobytes()
                )
                # The extender can creep a few bp into non-repetitive flank; a
                # dirty leading/trailing partial copy makes refine_repeat reject
                # the whole array. Retry with the edges trimmed inward by one
                # period so refine sees clean copy boundaries.
                repeat = None
                for ss, ee in ((full_start, full_end),
                               (full_start + period, full_end),
                               (full_start + period, full_end - period)):
                    if ee - ss < 2 * period:
                        continue
                    repeat = self._refine_and_create_repeat(
                        chromosome, ss, ee, motif, tier=2
                    )
                    if repeat:
                        break
                if repeat:
                    results.append(repeat)
                    covered.add((repeat.start, repeat.end))
                    seen[repeat.start:min(repeat.end, n)] = True
                    found += 1
        return found
