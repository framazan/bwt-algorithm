"""Shared BWT k-mer seeding core for tandem repeat detection.

Used by both Tier 2 and Tier 3.  The algorithm is:
  1. Sample k-mers from the text at a configurable stride.
  2. Use FM-index backward_search / locate_positions to find all occurrences.
  3. Detect periodic runs in the position arrays (arithmetic progressions).
  4. Extend seed positions with mismatch tolerance.
  5. Return raw candidate regions for tier-specific post-processing.
"""
from __future__ import annotations  # Allow string-form type hints for Python 3.9 and below

from dataclasses import dataclass   # Decorator for defining immutable data containers
from typing import List, Optional, Set, Tuple  # Generic types for type hints

import numpy as np  # NumPy for array operations and numerical computation

from .accelerators import extend_with_mismatches, find_periodic_runs
from .bwt_core import BWTCore  # FM-index core module (BWT, backward_search, locate_positions, etc.)


@dataclass
class SeedCandidate:
    """A raw repeat candidate from BWT seeding."""
    start: int      # Start position of the repeat array after extension (0-based)
    end: int        # End position of the repeat array after extension (0-based)
    period: int     # Detected repeat unit length (bp)
    copies: int     # Detected number of copies
    motif: str      # Representative motif string (extracted from full_start position)
    seed_pos: int   # Original seed position that generated this candidate (value of i)


def bwt_kmer_seed_scan(
    bwt: BWTCore,                        # FM-index object
    min_period: int,                     # Minimum repeat unit length to detect
    max_period: int,                     # Maximum repeat unit length to detect
    kmer_size: int = 16,                 # Length of k-mers to sample (default 16 bp)
    stride: int = 50,                    # K-mer sampling interval (default 50 bp)
    min_copies: int = 2,                 # Minimum number of copies to qualify as a repeat
    allowed_mismatch_rate: float = 0.20, # Allowed mismatch rate during extension (0.0-0.5)
    tolerance_ratio: float = 0.03,       # Tolerance ratio for period jitter in periodic run detection (default 3%)
    max_occurrences: int = 5000,         # Maximum k-mer occurrence count (skip if exceeded, likely low-complexity)
    covered_mask: Optional[np.ndarray] = None,  # Boolean mask of positions already found by previous tiers
    show_progress: bool = False,         # Whether to print progress
    label: str = "",                     # Label string for progress messages
) -> List[SeedCandidate]:
    """BWT k-mer seeding scan for tandem repeat candidates.

    Samples k-mers from the text at `stride` intervals, uses FM-index to
    find all occurrences, detects periodic runs in the occurrence positions,
    and extends seed hits with mismatch tolerance.

    Parameters
    ----------
    bwt : BWTCore
        The BWT/FM-index built on the sequence.
    min_period, max_period : int
        Period range to search for.
    kmer_size : int
        Length of k-mers to sample (default 16).  Should be shorter than
        min_period so that each repeat copy contains at least one full k-mer.
    stride : int
        Distance between consecutive k-mer samples (default 50).
    min_copies : int
        Minimum number of tandem copies required.
    allowed_mismatch_rate : float
        Mismatch tolerance for extension (0.0-0.5).
    tolerance_ratio : float
        Tolerance for period jitter in periodic run detection (default 3%).
    max_occurrences : int
        Skip k-mers with more occurrences than this (likely low-complexity).
    covered_mask : np.ndarray or None
        Boolean mask of positions already found by a previous tier.
        Positions where covered_mask[i] is True are skipped as seed origins.
    show_progress : bool
        Whether to print progress.
    label : str
        Label prefix for progress messages.

    Returns
    -------
    list[SeedCandidate]
        Raw repeat candidates.  Callers perform tier-specific post-processing
        (primitive period reduction, HOR detection, DP refinement, etc.).
    """
    text_arr = bwt.text_arr  # Original sequence used for BWT (numpy uint8 array)
    n = int(text_arr.size)   # Total sequence length (may include sentinel '$')

    # Exclude the sentinel character ('$', ASCII 36) used for BWT construction
    if n > 0 and text_arr[n - 1] == 36:   # '$'
        n -= 1  # Use actual sequence length excluding sentinel

    # Return immediately if sequence is too short to contain repeats meeting minimum copy count
    if n < min_period * min_copies:
        return []

    # Clamp k-mer size if larger than min_period (k-mer must fit within one copy)
    effective_kmer = min(kmer_size, min_period)  # Limit to not exceed repeat unit length
    if effective_kmer < 6:
        effective_kmer = min(6, min_period)  # Minimum 6 bp: shorter k-mers cause explosive FM-index hits

    candidates: List[SeedCandidate] = []               # List of detected repeat candidates
    seen_regions: Set[Tuple[int, int]] = set()         # Set of (start//bucket, period) keys for duplicate removal
    seen_kmers: Set[str] = set()                       # Set of already-queried k-mers (avoid re-querying)
    bwt_queries = 0  # FM-index query counter (for progress reporting)

    i = 0  # Current sampling position (incremented by stride)
    while i <= n - effective_kmer:  # Iterate until the final full-length k-mer start
        # Skip positions already found by previous tiers
        if covered_mask is not None and covered_mask[i]:
            i += stride  # Position already covered; move to next sample position
            continue

        # Extract k-mer at current position
        kmer_arr = text_arr[i:i + effective_kmer]  # Sequence slice for the k-mer
        kmer_str = kmer_arr.tobytes().decode('ascii', errors='replace').upper()  # Normalize input case

        # Skip if k-mer was already queried or contains non-DNA bases
        if kmer_str in seen_kmers:  # Check cache to avoid duplicate queries
            i += stride
            continue
        if not all(c in 'ACGT' for c in kmer_str):  # Skip if contains N, lowercase, or other non-DNA characters
            i += stride
            continue
        seen_kmers.add(kmer_str)  # Register queried k-mer in cache

        # --- Query all occurrence positions of k-mer via FM-index (BWT backward search) ---
        bwt_queries += 1  # Increment FM-index query counter
        sp, ep = bwt.backward_search(kmer_str)  # Returns suffix array range [sp, ep]
        if sp == -1:  # Not found (should not happen in theory, but defensive check)
            i += stride
            continue

        occ_count = ep - sp + 1  # Calculate total occurrence count of the k-mer
        if occ_count < min_copies or occ_count > max_occurrences:
            # Skip if too few occurrences (no repeat) or too many (low-complexity sequence)
            i += stride
            continue

        positions = bwt.suffix_array[sp:ep + 1].tolist()  # Reuse the SA interval from backward_search
        if len(positions) < min_copies:  # Skip if actual position count is below minimum copies
            i += stride
            continue

        # --- Detect arithmetic progressions (periodic runs) in occurrence position array ---
        pos_arr = np.array(sorted(positions), dtype=np.int64)
        # (Sorting is required for accurate arithmetic progression detection)

        # find_periodic_runs: detect evenly-spaced groups within tolerance_ratio in position array
        patterns = find_periodic_runs(
            pos_arr, min_period, max_period, min_copies, tolerance_ratio
        )  # Returns: [(run_start, run_end, period), ...] -- list of periodic k-mer occurrence runs

        for run_start, run_end, period in patterns:  # Process each periodic run
            run_start = int(run_start)  # Convert numpy type to Python int
            run_end = int(run_end)      # Convert numpy type to Python int
            period = int(period)        # Convert numpy type to Python int

            # Remove duplicate candidates for the same region: generate key from start position and period
            region_key = (run_start // max(period, 1), period)  # Key to identify the repeat region
            if region_key in seen_regions:  # Skip if this region was already processed
                continue

            # Skip if overlap with already covered region exceeds 50% (prevent redundant detection)
            if covered_mask is not None:
                covered_count = int(np.sum(
                    covered_mask[run_start:min(run_end + period, n)]
                ))  # Count already covered positions within this span
                span = run_end + period - run_start  # Total length of the run (including last k-mer)
                if span > 0 and covered_count > span * 0.5:
                    continue  # Skip if more than 50% is already covered

            # --- Mismatch-tolerant extension: extend the actual boundaries of the repeat array in both directions ---
            res = extend_with_mismatches(
                text_arr, run_start, period, n, allowed_mismatch_rate
            )  # Returns: (arr_start, arr_end, copies, full_start, full_end) or None

            if res is not None:
                arr_start, arr_end, copies, full_start, full_end = res
                # arr_start/arr_end: alignment-based boundaries, full_start/full_end: actual extended boundaries
            else:
                # Fallback when extension fails: use raw boundaries of the k-mer run
                full_start = run_start
                # run_end is the start position of the last k-mer, so add period to get end position
                full_end = run_end + period
                copies = max(1, (full_end - full_start) // period)  # Estimate copy count

            if copies < min_copies:  # Do not register as candidate if below minimum copy count
                continue

            # Extract motif string from the confirmed repeat region
            motif_start = max(0, full_start)  # Clamp to prevent negative index
            motif_arr = text_arr[motif_start:motif_start + period]  # Sequence slice for one copy
            motif_str = motif_arr.tobytes().decode('ascii', errors='replace')  # Convert byte array to string

            # Create SeedCandidate object and add to candidate list
            candidates.append(SeedCandidate(
                start=full_start,   # Extended repeat array start position
                end=full_end,       # Extended repeat array end position
                period=period,      # Detected repeat unit length
                copies=copies,      # Detected copy count
                motif=motif_str,    # Representative motif string
                seed_pos=i,         # Original seed position that generated this candidate
            ))
            seen_regions.add(region_key)  # Mark this region as processed

            # Mark discovered region in the mask: prevent re-seeding in the same span later
            if covered_mask is not None:
                covered_mask[full_start:min(full_end, n)] = True  # Mark this span as covered

        i += stride  # Move to next sample position

    # Print progress (only when show_progress is True)
    if show_progress:
        print(
            f"  [{label}] BWT k-mer seeding: {len(candidates)} candidates from "
            f"{bwt_queries} FM-index queries (kmer={effective_kmer}, stride={stride})",
            flush=True,
        )  # Print candidate count, FM-index query count, effective k-mer size, and sampling stride

    return candidates  # Return list of all detected raw repeat candidates
