import math
import itertools
import numpy as np
from typing import List, Tuple, Set, Optional
from .models import TandemRepeat  # Data class for tandem repeat results
from .motif_utils import MotifUtils  # Motif analysis utilities (alignment, consensus, TRF statistics, etc.)
from .bwt_core import BWTCore  # FM-index core module (BWT, backward search, etc.)
from .coverage import intervals_to_mask  # shared interval -> boolean mask
from .accelerators import anchor_scan_boundaries  # Cython-accelerated anchor-based boundary verification
from .bwt_seed import bwt_kmer_seed_scan  # Shared BWT k-mer seeding pipeline


def compute_adaptive_params(
    seq_len: int,        # Input sequence length (bp)
    gc_content: float,   # GC content (range 0 to 1)
    coverage_ratio: float,  # Fraction of sequence already covered by other tiers
    min_period: int,     # Minimum repeat unit length
    max_period: int,     # Maximum repeat unit length
    preset: str = "balanced",  # Speed/sensitivity preset ("fast", "balanced", "sensitive")
) -> dict:
    """Compute adaptive Tier 3 parameters based on input characteristics."""
    # Speed weight per preset: fast=faster (lower sensitivity), sensitive=slower (higher sensitivity)
    speed_weights = {"fast": 0.8, "balanced": 0.5, "sensitive": 0.2}
    speed_weight = speed_weights.get(preset, 0.5)  # Extract weight for the preset (default 0.5 if not found)
    speed_factor = speed_weight / 0.5  # Relative speed multiplier compared to balanced (0.5)

    # Compute base parameters using continuous functions (log-scaled by sequence length)
    safe_seq = max(seq_len, 1)  # Lower bound to prevent division by zero

    base_kmer = int(10 + 6 * math.log10(max(safe_seq / 1e5, 1)))  # k-mer size: larger k-mers for longer sequences
    base_stride = int(safe_seq / 40000)  # Sampling stride: proportional to sequence length
    base_max_occ = int(safe_seq / 30000)  # Max allowed k-mer occurrences: filters out low-complexity high-frequency k-mers
    base_scan_bw = int(50 * safe_seq / 1e8)  # Backward scan range (in repeat units)
    base_scan_fw = int(600 * safe_seq / 1e8)  # Forward scan range (in repeat units)

    # Accuracy parameters: depend only on sequence characteristics, not preset
    allowed_mismatch_rate = 0.15 + 0.10 * abs(gc_content - 0.5)  # Higher mismatch tolerance for extreme GC content
    tolerance_ratio = 0.02 + 0.02 * (max_period / 100000)  # Wider period tolerance for larger max periods
    anchor_match_pct = 0.70 + 0.10 * (1 - coverage_ratio)  # Relax anchor matching threshold when coverage is low

    # Apply preset multiplier to speed parameters
    kmer_size = int(base_kmer + (speed_factor - 1) * 2)  # Fast mode increases k-mer size (fewer hits)
    stride = int(base_stride * speed_factor)  # Fast mode uses larger stride for sparser sampling
    max_occurrences = int(base_max_occ / speed_factor)  # Fast mode allows fewer high-frequency k-mers
    scan_backward = int(base_scan_bw / speed_factor)  # Fast mode scans shorter backward range
    scan_forward = int(base_scan_fw / speed_factor)  # Fast mode scans shorter forward range

    # When coverage exceeds 50%, reduce stride to scan uncovered regions more finely
    if coverage_ratio > 0.5:
        stride = int(stride * (1 - 0.5 * coverage_ratio))  # Higher coverage leads to smaller stride

    # Strategy adjustments by sequence size range
    if safe_seq > 100_000_000:  # Over 100 Mbp: large chromosome mode
        stride = max(stride, 150)  # Ensure minimum stride to avoid excessive slowdown
        kmer_size = max(kmer_size, 20)  # Use longer k-mers for specificity on large sequences
        max_occurrences = min(max_occurrences, 500)  # Tighten upper bound to filter low-complexity repeats
    elif safe_seq < 100_000:  # Under 100 kbp: micro mode (short sequences)
        stride = max(stride, 20)  # Maintain minimum sampling stride even for short sequences
        kmer_size = max(kmer_size, 12)  # Ensure minimum k-mer size for short sequences

    # Clamp all parameters to allowed ranges
    kmer_size = max(12, min(28, kmer_size))  # k-mer size: restricted to 12-28 bp
    stride = max(20, min(300, stride))  # Sampling stride: restricted to 20-300
    allowed_mismatch_rate = max(0.15, min(0.20, allowed_mismatch_rate))  # Mismatch rate: 15%-20%
    tolerance_ratio = max(0.02, min(0.04, tolerance_ratio))  # Period tolerance: 2%-4%
    max_occurrences = max(200, min(1500, max_occurrences))  # Max occurrences: 200-1500
    anchor_match_pct = max(0.70, min(0.80, anchor_match_pct))  # Anchor match ratio: 70%-80%
    scan_backward = max(20, min(80, scan_backward))  # Backward scan: 20-80 units
    scan_forward = max(200, min(800, scan_forward))  # Forward scan: 200-800 units

    # Return final parameters as a dictionary
    return {
        "kmer_size": kmer_size,              # k-mer length for FM-index lookup
        "stride": stride,                    # k-mer sampling stride
        "allowed_mismatch_rate": allowed_mismatch_rate,  # Allowed mismatch rate during extension
        "tolerance_ratio": tolerance_ratio,  # Period detection tolerance ratio
        "max_occurrences": max_occurrences,  # Max k-mer occurrences (low-complexity filter)
        "anchor_match_pct": anchor_match_pct,  # Anchor-based boundary verification match threshold
        "scan_backward": scan_backward,      # Anchor backward scan range (in repeat units)
        "scan_forward": scan_forward,        # Anchor forward scan range (in repeat units)
    }


class Tier3LongReadFinder:
    """Tier 3: Long-read repeat finder using BWT k-mer seeding.

    Uses the shared BWT seeding core (bwt_seed.py) with Tier 3-specific
    parameters (long periods, sparse sampling, high divergence tolerance)
    and Tier 3-specific post-processing (anchor-based boundary verification,
    consensus from sampled copies).
    """

    def __init__(self, bwt_core: BWTCore, min_length: int = 100,
                 max_length: int = 100000, min_copies: float = 2.0,
                 mode: str = "balanced"):
        self.bwt = bwt_core          # Store FM-index object (used for subsequent searches)
        self.min_length = min_length  # Minimum repeat unit length to detect (bp)
        self.max_length = max_length  # Maximum repeat unit length to detect (bp)
        self.min_copies = min_copies  # Minimum number of copies required to qualify as a repeat
        self.mode = mode             # Speed/sensitivity preset string

    def find_long_repeats(self, chromosome: str, tier1_seen: Set[Tuple[int, int]],
                          tier2_seen: Set[Tuple[int, int]]) -> List[TandemRepeat]:
        """Find long repeats not caught by Tier 1 or Tier 2.

        Uses the shared BWT k-mer seeding pipeline with Tier 3 parameters:
        - Large k-mers (20bp) for uniqueness
        - Sparse sampling (stride=100) for speed
        - Wide period range (100bp-100kbp)
        - Higher divergence tolerance (20%)

        Tier 3 post-processing:
        - Anchor-based boundary verification for ultra-long arrays
        - Consensus from sampled copies (not full DP) for efficiency
        """
        text_arr = self.bwt.text_arr  # Original sequence used for BWT (numpy uint8 array)
        n = text_arr.size             # Total sequence length (including sentinel '$')

        # Sequence too short for detection; return empty results
        if n < self.min_length * 2:
            return []

        # Regions already claimed by Tier 1 and Tier 2
        mask = intervals_to_mask(itertools.chain(tier1_seen, tier2_seen), n)

        # Compute adaptive parameters based on sequence characteristics
        gc_content = float(np.mean((text_arr == ord('G')) | (text_arr == ord('C'))))  # Compute GC content
        coverage_ratio = float(np.mean(mask))  # Compute fraction of already-covered regions
        params = compute_adaptive_params(
            seq_len=n,
            gc_content=gc_content,
            coverage_ratio=coverage_ratio,
            min_period=self.min_length,
            max_period=self.max_length,
            preset=self.mode,
        )  # Parameter dictionary reflecting sequence length, GC content, coverage, etc.

        anchor_match_pct = params["anchor_match_pct"]  # Extract anchor-based boundary verification match threshold
        scan_bw_periods = params["scan_backward"]      # Extract backward anchor scan range
        scan_fw_periods = params["scan_forward"]       # Extract forward anchor scan range

        # ===== Phase A: Run shared BWT k-mer seeding pipeline =====
        seed_candidates = bwt_kmer_seed_scan(
            bwt=self.bwt,
            min_period=self.min_length,       # Minimum repeat unit length
            max_period=self.max_length,       # Maximum repeat unit length
            kmer_size=params["kmer_size"],    # k-mer size for FM-index lookup
            stride=params["stride"],          # k-mer sampling stride
            min_copies=int(self.min_copies),  # Minimum copies (cast to int)
            allowed_mismatch_rate=params["allowed_mismatch_rate"],  # Mismatch tolerance rate
            tolerance_ratio=params["tolerance_ratio"],              # Period tolerance ratio
            max_occurrences=params["max_occurrences"],              # Max k-mer occurrences
            covered_mask=mask,                # Mask of already-detected regions (skip seeding positions)
            show_progress=False,              # Disable progress output
            label=f"{chromosome} Tier3",      # Label for progress messages
        )  # Returns: list of SeedCandidate objects (raw repeat candidates)

        # ===== Tier 3 post-processing: convert candidates to TandemRepeat objects =====
        repeats = []          # Final results list
        seen_regions = set()  # Set of region keys for deduplication

        for cand in seed_candidates:  # Post-process each seed candidate
            region_key = (cand.start // max(cand.period, 1), cand.period)  # Region identification key (for deduplication)
            if region_key in seen_regions:  # Skip if this region was already processed
                continue

            period = cand.period      # Detected repeat unit length
            copies = cand.copies      # Detected copy count
            full_start = cand.start   # Repeat array start position after extension
            full_end = cand.end       # Repeat array end position after extension

            # Ultra-long repeats (copies >100 or length >10kb): use anchor-based boundary verification
            # Full DP alignment is too expensive, so use anchor scanning instead
            if copies > 100 or (full_end - full_start) > 10000:
                seed_pos = cand.seed_pos  # Original seed position that generated this candidate
                # Fall back to full_start if seed position exceeds sequence end
                if seed_pos + period > n:
                    seed_pos = full_start

                motif_arr = text_arr[seed_pos:seed_pos + period]  # Extract motif from seed position
                motif = motif_arr.tobytes().decode('ascii', errors='replace')  # Convert byte array to string

                # Anchor-based boundary verification: refine actual repeat start/end using Cython acceleration
                true_start, true_end = anchor_scan_boundaries(
                    text_arr, seed_pos, period, n,
                    anchor_match_pct, scan_bw_periods, scan_fw_periods,
                )  # Recompute boundaries based on anchor match ratio and scan range

                true_copies = (true_end - true_start) / period  # Recompute actual copy count

                if true_copies >= self.min_copies:  # Only produce results meeting the minimum copy count
                    max_consensus_copies = min(int(true_copies), 20)  # Max copies for consensus computation (capped at 20 for efficiency)
                    consensus_arr, mm_rate, max_mm = MotifUtils.build_consensus_motif_array(
                        text_arr, true_start, period, max_consensus_copies
                    )  # Compute consensus motif and mismatch statistics from sampled copies
                    consensus_motif = consensus_arr.tobytes().decode('ascii', errors='replace') if consensus_arr.size > 0 else motif
                    # Fall back to original motif if consensus array is empty

                    (percent_matches, percent_indels, score, composition,
                     entropy, actual_sequence) = MotifUtils.calculate_trf_statistics(
                        text_arr, true_start, true_end, consensus_motif, int(true_copies), mm_rate
                    )  # Compute TRF-compatible statistics: match rate, indel rate, score, base composition, entropy, actual sequence

                    # Create TandemRepeat object (using anchor-based boundaries)
                    repeat = TandemRepeat(
                        chrom=chromosome,           # Chromosome name
                        start=true_start,           # Actual repeat start position (0-based)
                        end=true_end,               # Actual repeat end position (0-based)
                        motif=motif,                # Motif extracted from original seed
                        copies=float(true_copies),  # Actual copy count (float)
                        length=true_end - true_start,  # Total repeat array length
                        tier=3,                     # Indicates this result was produced by Tier 3
                        confidence=max(0.5, 1.0 - mm_rate),  # Confidence: higher when mismatch rate is lower (min 0.5)
                        consensus_motif=consensus_motif,      # Consensus-based motif
                        mismatch_rate=mm_rate,                # Average mismatch rate
                        max_mismatches_per_copy=max_mm,       # Maximum mismatches per copy
                        n_copies_evaluated=max_consensus_copies,  # Number of copies used for consensus computation
                        strand='+',                           # Forward strand (Tier 3 always uses '+')
                        percent_matches=percent_matches,      # TRF statistic: match percentage
                        percent_indels=percent_indels,        # TRF statistic: indel percentage
                        score=score,                          # TRF statistic: score
                        composition=composition,              # TRF statistic: base composition
                        entropy=entropy,                      # TRF statistic: sequence entropy
                        actual_sequence=actual_sequence[:500] if len(actual_sequence) > 500 else actual_sequence,
                        # Actual sequence: truncated to 500bp to limit output size
                        variations=None  # Tier 3 does not record variation information
                    )
                else:
                    repeat = None  # Below minimum copy count: no result
            else:
                # Relatively short repeats (<100 copies or <10kb): refine with full DP alignment
                motif_arr = text_arr[full_start:full_start + period]  # Extract motif from repeat start position
                motif = motif_arr.tobytes().decode('ascii', errors='replace')  # Convert byte array to string

                refined = MotifUtils.refine_repeat(
                    self.bwt.text,        # Original text (string)
                    full_start,           # Repeat array start position
                    full_end,             # Repeat array end position
                    motif,                # Initial motif
                    mismatch_fraction=0.2,  # Allowed mismatch fraction (20%)
                    indel_fraction=0.1,   # Allowed indel fraction (10%)
                    min_copies=int(self.min_copies)  # Minimum copy count
                )  # DP alignment-based repeat refinement (includes primitive period reduction)

                if refined:
                    repeat = MotifUtils.refined_to_repeat(chromosome, refined, tier=3, text_arr=text_arr)
                    # Convert refinement result to TandemRepeat object
                else:
                    repeat = None  # DP refinement failed: no result

            if repeat:
                # Check containment against existing results (ignore new result if fully contained)
                is_new = True  # Assume new result by default
                for r in repeats:
                    if r.start <= repeat.start and r.end >= repeat.end:
                        is_new = False  # Existing result fully contains current repeat; treat as duplicate
                        break

                if is_new:
                    repeats.append(repeat)             # Add to final results list
                    seen_regions.add(region_key)       # Register this region key as processed
                    mask[repeat.start:min(repeat.end, n)] = True  # Mark discovered region in mask (skip subsequent seeding)

        return repeats  # Return all discovered long tandem repeat results
