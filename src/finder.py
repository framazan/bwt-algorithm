import os    # Standard library for environment-variable overrides
import time  # Standard library for measuring execution time
import numpy as np  # NumPy for numerical operations and array processing
from typing import List, Tuple, Set, Optional, Dict  # typing module for type hints
from .bwt_core import BWTCore  # BWT/FM-index core data structure
from .models import TandemRepeat  # Tandem repeat data class
from .tier1 import Tier1STRFinder  # Tier 1: Short perfect repeat finder
from .tier2 import Tier2LCPFinder  # Tier 2: Medium-length imperfect repeat finder
from .tier3 import Tier3LongReadFinder  # Tier 3: Long repeat sequence finder
from .motif_utils import MotifUtils  # Motif canonicalization and statistics utilities

class TandemRepeatFinder:
    """Main coordinator for multi-tier tandem repeat finding."""

    VALID_TIERS = {"tier1", "tier2", "tier3"}  # Set of valid detection tier names

    def __init__(self, sequence: str, chromosome: str = "chr1",
                 min_period: int = 1, max_period: int = 2000,
                 show_progress: bool = False,
                 enabled_tiers: Optional[Set[str]] = None,
                 min_array_bp: Optional[int] = None,
                 max_array_bp: Optional[int] = None,
                 tier3_mode: str = "balanced"):
        self.sequence = sequence  # DNA sequence string to analyze
        self.chromosome = chromosome  # Chromosome name (used in output records)
        self.min_period = min_period  # Minimum motif length (bp) to search for
        self.max_period = max_period  # Maximum motif length (bp) to search for
        self.show_progress = show_progress  # Flag for whether to print progress info
        self.enabled_tiers = self._normalize_tiers(enabled_tiers)  # Normalize the set of tiers to enable

        # Ensure array length lower/upper bounds are non-negative
        self.min_array_bp = max(0, min_array_bp) if min_array_bp else None
        self.max_array_bp = max(0, max_array_bp) if max_array_bp else None
        if self.min_array_bp and self.max_array_bp and self.min_array_bp > self.max_array_bp:
            # Swap to keep bounds consistent
            # If lower bound exceeds upper bound, swap them for consistency
            self.min_array_bp, self.max_array_bp = self.max_array_bp, self.min_array_bp

        # Initialize BWT Core
        if show_progress:
            # Notify user that BWT index construction is starting
            print(f"  [{chromosome}] Building BWT and FM-index...", flush=True)
        t0 = time.time()  # Record BWT construction start time
        # Ensure sentinel
        if not sequence.endswith('$'):
            sequence += '$'  # Append sentinel character '$' to end of sequence for BWT construction

        self.bwt = BWTCore(sequence, sa_sample_rate=1)  # Build FM-index (suffix array, BWT, occurrence arrays)
        if show_progress:
            # Print BWT construction completion and elapsed time
            print(f"  [{chromosome}] BWT built in {time.time() - t0:.2f}s", flush=True)

        # Initialize Tiers
        self.tier1 = None  # Initialize Tier 1 finder (default None)
        tier1_min = max(1, min_period)  # Minimum motif length for Tier 1 (at least 1 bp)
        tier1_max = min(9, max_period)  # Maximum motif length for Tier 1 (at most 9 bp)
        if "tier1" in self.enabled_tiers and tier1_min <= tier1_max:
            # Only create instance when Tier 1 is enabled and has a valid range
            self.tier1 = Tier1STRFinder(
                self.bwt.text_arr,          # Sequence converted to byte array
                self.bwt,                   # FM-index object
                max_motif_length=tier1_max, # Maximum motif length
                min_motif_length=tier1_min, # Minimum motif length
                allowed_mismatch_rate=0.2,  # Allowed mismatch rate 20%
                allowed_indel_rate=0.1,     # Allowed indel rate 10%
                show_progress=show_progress # Pass progress display flag
            )

        self.tier2 = None  # Initialize Tier 2 finder (default None)
        if "tier2" in self.enabled_tiers:
            # Only create instance when Tier 2 is enabled
            self.tier2 = Tier2LCPFinder(
                self.bwt,                   # FM-index object
                min_period=min_period,      # Minimum motif length
                max_period=max_period,      # Maximum motif length
                allowed_mismatch_rate=0.2,  # Allowed mismatch rate 20%
                allowed_indel_rate=0.1,     # Allowed indel rate 10%
                show_progress=show_progress # Pass progress display flag
            )

        # Only create Tier 3 instance if enabled, otherwise None
        self.tier3 = Tier3LongReadFinder(self.bwt, mode=tier3_mode) if "tier3" in self.enabled_tiers else None

        # Satellite gap-fill tuning (env-overridable). Defaults are the improved
        # operating point that recovers divergent-alpha-sat interior gaps which
        # the 3-tier pipeline leaves uncovered between large-HOR-motif calls.
        # SAT_FILL_MIN_IDENTITY: min autocorrelation identity to accept a gap as
        #   satellite (0.45 captures ~46%-identity divergent alpha-sat; ~2.5x
        #   above the ~0.18 autocorrelation floor of non-repetitive DNA).
        # SAT_ANCHOR_MIN_MOTIF / SAT_ANCHOR_MAX_MOTIF: motif-length window for a
        #   repeat to count as a satellite anchor whose neighbourhood is scanned
        #   for uncovered gaps. MAX=0 means no upper bound, so large HOR-unit
        #   calls (motif >> 300bp) also anchor gap scanning.
        self.sat_fill_min_identity = float(os.environ.get("SAT_FILL_MIN_IDENTITY", "0.45"))
        self.sat_anchor_min_motif = int(os.environ.get("SAT_ANCHOR_MIN_MOTIF", "100"))
        self.sat_anchor_max_motif = int(os.environ.get("SAT_ANCHOR_MAX_MOTIF", "0"))
        # Period window for the gap autocorrelation scan. Default 100-360 covers
        # the alpha-satellite monomer (171bp) and the dimeric HOR period
        # (2x171=342) where highly divergent arrays (e.g. chr3, ~46% monomer
        # identity) carry their strongest periodic signal. The 0.45 identity
        # floor stays ~2.5x above the ~0.18 autocorrelation of non-repetitive DNA.
        self.sat_fill_min_period = int(os.environ.get("SAT_FILL_MIN_PERIOD", "100"))
        self.sat_fill_max_period = int(os.environ.get("SAT_FILL_MAX_PERIOD", "360"))

    def find_all(self) -> List[TandemRepeat]:
        """Execute the full 3-tier finding pipeline."""
        all_repeats = []  # List to collect repeats found across all tiers
        tier1_seen: Set[Tuple[int, int]] = set()  # Set of (start, end) coordinates found by Tier 1 (for deduplication)
        tier2_seen: Set[Tuple[int, int]] = set()  # Set of (start, end) coordinates found by Tier 2 (for deduplication)

        # --- Tier 1: Short Perfect Repeats ---
        if self.tier1:
            if self.show_progress:
                # Print Tier 1 start notification
                print(f"  [{self.chromosome}] Running Tier 1 (Short Perfect)...", flush=True)
            t0 = time.time()  # Record Tier 1 start time
            tier1_repeats = self.tier1.find_strs(self.chromosome)  # Run Tier 1: find short perfect repeats

            accepted = 0  # Counter for repeats that pass the filter
            for r in tier1_repeats:
                # Register each repeat in the global list (with array length range filter)
                if self._register_repeat(r, all_repeats, tier1_seen):
                    accepted += 1  # Increment counter on successful registration

            if self.show_progress:
                # Print Tier 1 completion and results summary
                print(f"  [{self.chromosome}] Tier 1 found {accepted} repeats in {time.time() - t0:.2f}s", flush=True)
        elif self.show_progress:
            # Notify if Tier 1 is disabled or out of requested range
            print(f"  [{self.chromosome}] Skipping Tier 1 (disabled or out of requested range)", flush=True)

        # --- Tier 2: Imperfect & Medium Repeats (>=10bp by design) ---
        if self.tier2:
            if self.show_progress:
                # Print Tier 2 start notification
                print(f"  [{self.chromosome}] Running Tier 2 (Imperfect & Medium)...", flush=True)
            t0 = time.time()  # Record Tier 2 start time
            # 2a. Long unit repeats (strict adjacency, >=20bp units)
            long_unit_repeats = []  # List for long unit repeat results
            long_kept = 0  # Count of registered long unit repeats
            min_unit = max(20, self.min_period)  # Minimum motif length for Tier 2 long unit repeats (at least 20 bp)
            if self.max_period >= min_unit:
                # Only run long unit repeat search when max period is at least min_unit
                long_unit_repeats = self.tier2.find_long_unit_repeats_strict(
                    self.chromosome,          # Chromosome name
                    min_unit_len=min_unit,    # Minimum unit length
                    max_unit_len=self.max_period,  # Maximum unit length
                    min_copies=2              # Minimum copy count of 2
                )

                for r in long_unit_repeats:
                    is_new = True  # Initialize flag for whether this is a new repeat
                    for start, end in tier1_seen:
                        # Check if it overlaps with positions already found by Tier 1
                        if not (r.end <= start or r.start >= end):
                            is_new = False  # Not new if overlapping
                            break
                    if is_new and self._register_repeat(r, all_repeats, tier2_seen):
                        long_kept += 1  # Increment counter if new and successfully registered

            # 2b. General scanning for medium/long repeats up to requested max
            medium_repeats = []  # List for medium-length repeat results
            medium_kept = 0  # Count of registered medium-length repeats
            # Force Tier2 to ignore classic microsatellites: start from period 10bp
            scan_lower = max(10, self.min_period)   # Scan lower bound: at least 10 bp to exclude microsatellites
            scan_upper = min(50, self.max_period)   # Scan upper bound: up to 50 bp (BWT seed method range)
            if scan_upper >= scan_lower:
                # Only run when there is a valid scan range
                combined_seen = tier1_seen.union(tier2_seen)  # Union of positions already found by Tier 1 and Tier 2
                medium_repeats = self.tier2.find_long_repeats(
                    self.chromosome,          # Chromosome name
                    combined_seen,            # Already-found position set for deduplication
                    max_scan_period=scan_upper  # Maximum scan period
                )

                for r in medium_repeats:
                    # Register medium-length repeats
                    if self._register_repeat(r, all_repeats, tier2_seen):
                        medium_kept += 1  # Increment counter on successful registration
            if self.show_progress:
                total_kept = long_kept + medium_kept  # Total of long unit and medium-length repeats
                # Print Tier 2 completion and results summary
                print(f"  [{self.chromosome}] Tier 2 processed {total_kept} repeats (>=10bp motifs) in {time.time() - t0:.2f}s", flush=True)
        elif self.show_progress:
            # Notify if Tier 2 is disabled
            print(f"  [{self.chromosome}] Skipping Tier 2 (disabled)", flush=True)

        # --- Tier 3: Long Reads (Optional/Advanced) ---
        if self.tier3:
            if self.show_progress:
                # Print Tier 3 start notification
                print(f"  [{self.chromosome}] Running Tier 3 (Long Reads)...", flush=True)
            t0 = time.time()  # Record Tier 3 start time

            combined_seen = tier1_seen.union(tier2_seen)  # Union of Tier 1 + Tier 2 found positions (for Tier 3 deduplication)
            tier3_repeats = self.tier3.find_long_repeats(
                self.chromosome,  # Chromosome name
                tier1_seen,       # Tier 1 found position set
                tier2_seen        # Tier 2 found position set
            )

            accepted = 0  # Counter for repeats registered from Tier 3
            for r in tier3_repeats:
                # Register Tier 3 results in the global list
                if self._register_repeat(r, all_repeats, combined_seen):
                    accepted += 1  # Increment counter on successful registration

            if self.show_progress:
                # Print Tier 3 completion and results summary
                print(f"  [{self.chromosome}] Tier 3 found {accepted} repeats in {time.time() - t0:.2f}s", flush=True)
        elif self.show_progress:
            # Notify if Tier 3 is disabled
            print(f"  [{self.chromosome}] Skipping Tier 3 (disabled)", flush=True)

        # --- Post-processing ---
        all_repeats.sort(key=lambda x: x.start)

        if self.show_progress:
            t0 = time.time()
        all_repeats = self._merge_adjacent_repeats(all_repeats)
        if self.show_progress:
            print(f"  [{self.chromosome}] Merge: {time.time() - t0:.2f}s, {len(all_repeats)} repeats", flush=True)
            t0 = time.time()
        final_repeats = self._filter_overlaps(all_repeats)
        if self.show_progress:
            print(f"  [{self.chromosome}] Filter: {time.time() - t0:.2f}s, {len(final_repeats)} repeats", flush=True)

        # Satellite DNA scanner (multiple passes)
        for pass_num in range(2):
            if self.show_progress:
                t0 = time.time()
            prev_count = len(final_repeats)
            final_repeats = self._fill_satellite_gaps(final_repeats)
            final_repeats = self._merge_adjacent_repeats(final_repeats)
            if self.show_progress:
                print(f"  [{self.chromosome}] Satellite pass {pass_num+1}: {time.time() - t0:.2f}s, {len(final_repeats)} repeats (+{len(final_repeats) - prev_count})", flush=True)
            if len(final_repeats) == prev_count:
                break

        final_repeats = [r for r in final_repeats if self._repeat_within_bounds(r)]  # Filter to only results within user-specified bounds

        return final_repeats  # Return the final list of repeats

    def _filter_overlaps(self, repeats: List[TandemRepeat]) -> List[TandemRepeat]:
        """Filter overlapping repeats, keeping the one with higher score/length."""
        if not repeats:
            return []  # Return empty list if input is empty

        # Sort by start position
        repeats.sort(key=lambda x: x.start)  # Sort by start position

        filtered = [repeats[0]]  # Add the first repeat to the result list

        for current in repeats[1:]:
            prev = filtered[-1]  # Last retained repeat so far

            # Check overlap
            if current.start < prev.end:
                # Two repeats overlap; calculate overlap amount
                # Calculate overlap amount
                overlap = min(prev.end, current.end) - max(prev.start, current.start)  # Actual overlapping bp count
                overlap_ratio = overlap / min(prev.length, current.length)  # Overlap ratio relative to the shorter one

                if overlap_ratio > 0.5:  # Significant overlap
                    # If overlap exceeds 50%, select the one with the higher score
                    # Keep the better one
                    # Criteria: Length * (1 - mismatch_rate)
                    prev_score = prev.length * (1.0 - prev.mismatch_rate)   # Calculate score for previous repeat
                    curr_score = current.length * (1.0 - current.mismatch_rate)  # Calculate score for current repeat

                    if curr_score > prev_score:
                        filtered[-1] = current  # If current is better, replace previous with current
                else:
                    # Small overlap, keep both (maybe compound?)
                    # Small overlap; keep both repeats (may be a compound repeat)
                    filtered.append(current)
            else:
                filtered.append(current)  # No overlap; add as-is

        return filtered  # Return filtered list of repeats

    def _merge_adjacent_repeats(self, repeats: List[TandemRepeat]) -> List[TandemRepeat]:
        """Merge adjacent repeats with the same motif."""
        if not repeats:
            return []  # Return empty list if input is empty

        # Sort by start position
        repeats.sort(key=lambda x: x.start)  # Sort by start position

        merged = [repeats[0]]  # Start merge result list with the first repeat

        for current in repeats[1:]:
            prev = merged[-1]  # Last merged item so far

            # Check if they are adjacent or overlapping
            # Allow a small gap (e.g., up to period length or 10bp)
            gap = max(0, current.start - prev.end)  # Gap between two repeats (clamped to non-negative)

            # Check if motifs are compatible (same canonical motif)
            motif1 = prev.consensus_motif or prev.motif    # Consensus motif or original motif of the previous item
            motif2 = current.consensus_motif or current.motif  # Consensus motif or original motif of the current item

            canon1, strand1 = MotifUtils.get_canonical_motif_stranded(motif1)  # Canonical form and strand direction of previous motif
            canon2, strand2 = MotifUtils.get_canonical_motif_stranded(motif2)  # Canonical form and strand direction of current motif

            # Allow merge if canonical motifs match and gap is small
            # For long-period repeats (satellite DNA), allow larger gaps
            # since individual repeat units can be ~178bp with imperfect boundaries
            period_len = len(canon1)
            if period_len >= 100:
                max_gap = period_len * 100  # e.g. 17.8kb for CEN180
            elif period_len >= 20:
                max_gap = period_len * 10
            else:
                max_gap = max(10, period_len)
            # For long-period repeats, use fuzzy canonical motif matching
            # (satellite DNA consensus can drift between adjacent regions)
            motifs_match = (canon1 == canon2)
            if not motifs_match and len(canon1) == len(canon2) and len(canon1) >= 50:
                hamming = sum(1 for a, b in zip(canon1, canon2) if a != b)
                motifs_match = (hamming / len(canon1)) <= 0.10  # 10% tolerance
            if motifs_match and gap <= max_gap:
                # Trial merge: check if combined region quality is acceptable
                new_start = min(prev.start, current.start)
                new_end = max(prev.end, current.end)
                avg_mm = (prev.mismatch_rate + current.mismatch_rate) / 2

                # Quality check for merge
                text_arr = self.bwt.text_arr
                actual_motif = motif1
                mlen = len(actual_motif)
                merge_ok = False

                if mlen >= 100 and gap > mlen:
                    # Satellite DNA: check periodicity of gap region via autocorrelation
                    gap_start_pos = prev.end
                    gap_end_pos = current.start
                    gap_seq = text_arr[gap_start_pos:gap_end_pos]
                    gap_len = len(gap_seq)
                    if gap_len >= mlen * 2:
                        total = gap_len - mlen
                        matches = int(np.sum(gap_seq[:total] == gap_seq[mlen:mlen + total]))
                        identity = matches / total if total > 0 else 0
                        merge_ok = identity >= 0.55
                else:
                    # Short repeats: use direct motif comparison
                    motif_arr = np.frombuffer(actual_motif.encode('ascii'), dtype=np.uint8)
                    trial_mismatches = 0
                    trial_total = 0
                    for pos in range(new_start, min(new_end, len(text_arr) - mlen), mlen):
                        window = text_arr[pos:pos + mlen]
                        if len(window) == mlen:
                            trial_mismatches += np.sum(window != motif_arr)
                            trial_total += mlen
                    trial_mm = trial_mismatches / trial_total if trial_total > 0 else 0
                    max_acceptable_mm = max(avg_mm * 2, 0.15)
                    merge_ok = trial_mm <= max_acceptable_mm

                if merge_ok:
                    prev.start = new_start
                    prev.end = new_end
                    prev.length = new_end - new_start
                    prev.copies = prev.length / len(canon1)
                    # Skip expensive stats recomputation for very large regions
                    if prev.length <= 50000:
                        self._recompute_stats(prev)
                else:
                    # Mismatch too high after merge — keep as separate repeats
                    merged.append(current)

            else:
                merged.append(current)  # Merge conditions not met; add as a separate item

        return merged  # Return the merged list of repeats

    def _recompute_stats(self, repeat: TandemRepeat):
        """Recompute statistics for a repeat (e.g. after merging)."""
        text_arr = self.bwt.text_arr  # Reference the byte array sequence from the FM-index
        motif = repeat.consensus_motif or repeat.motif  # Use consensus motif or original motif
        motif_len = len(motif)  # Calculate motif length

        if motif_len == 0:
            return  # Cannot compute statistics with zero-length motif; return immediately

        # Re-derive consensus and mismatch rate from the full merged region
        # Recompute consensus and mismatch rate from the entire merged region
        consensus_arr, mm_rate, max_mm = MotifUtils.build_consensus_motif_array(
            text_arr, repeat.start, motif_len, int(repeat.copies)
        )

        if consensus_arr.size > 0:
            # Decode consensus array to ASCII string
            consensus_str = consensus_arr.tobytes().decode('ascii', errors='replace')
            repeat.consensus_motif = consensus_str  # Update consensus motif
            repeat.motif = consensus_str # Update motif to new consensus
            repeat.mismatch_rate = mm_rate  # Update mismatch rate
            repeat.max_mismatches_per_copy = max_mm  # Update maximum mismatches per copy

            # Recalculate TRF stats
            # Recalculate TRF-compatible statistics
            (percent_matches, percent_indels, score, composition,
             entropy, actual_sequence) = MotifUtils.calculate_trf_statistics(
                text_arr, repeat.start, repeat.end, consensus_str, int(repeat.copies), mm_rate
            )

            repeat.percent_matches = percent_matches  # Update match percentage
            repeat.percent_indels = percent_indels    # Update indel percentage
            repeat.score = score                      # Update TRF score
            repeat.composition = composition          # Update base composition
            repeat.entropy = entropy                  # Update Shannon entropy
            repeat.actual_sequence = actual_sequence  # Update actual sequence string

            # Update variations
            # Update per-copy variation summary
            variations = MotifUtils.summarize_variations_array(
                text_arr, repeat.start, repeat.end, motif_len, consensus_arr
            )
            repeat.variations = variations  # Update variation information

    def _fill_satellite_gaps(self, repeats: List[TandemRepeat]) -> List[TandemRepeat]:
        """Scan for satellite DNA regions not covered by existing detections.

        Uses autocorrelation on uncovered regions to find periodic satellite DNA
        (e.g., CEN180) that may have been missed by the 3-tier pipeline due to
        high inter-copy divergence.
        """
        text_arr = self.bwt.text_arr
        text_str = self.sequence
        n = len(text_arr)

        if n < 1000:
            return repeats

        # Build coverage mask from existing repeats
        covered = np.zeros(n, dtype=bool)
        for r in repeats:
            covered[r.start:min(r.end, n)] = True

        # Find uncovered blocks >= 300bp using numpy
        transitions = np.diff(covered.astype(np.int8))
        # Starts of uncovered regions: transition from True(1) to False(0) = -1
        # Ends of uncovered regions: transition from False(0) to True(1) = +1
        gap_starts = np.where(transitions == -1)[0] + 1  # Position after last covered
        gap_ends = np.where(transitions == 1)[0] + 1     # First covered position
        # Handle boundaries
        if not covered[0]:
            gap_starts = np.concatenate(([0], gap_starts))
        if not covered[-1]:
            gap_ends = np.concatenate((gap_ends, [n]))
        uncovered_blocks = [(int(s), int(e)) for s, e in zip(gap_starts, gap_ends) if e - s >= 300]

        if not uncovered_blocks:
            return repeats

        # Collect satellite motifs and positions from existing long-period repeats
        satellite_positions = []
        for r in repeats:
            m = r.consensus_motif or r.motif
            if not m or len(m) < self.sat_anchor_min_motif:
                continue
            if self.sat_anchor_max_motif and len(m) > self.sat_anchor_max_motif:
                continue
            satellite_positions.append((r.start, r.end))

        # Build a proximity mask: only scan blocks near satellite regions
        # This avoids scanning the entire non-centromeric sequence
        near_satellite = np.zeros(n, dtype=bool)
        proximity = 50000  # 50kb proximity window
        for sat_start, sat_end in satellite_positions:
            near_satellite[max(0, sat_start - proximity):min(n, sat_end + proximity)] = True

        new_repeats = []
        for block_start, block_end in uncovered_blocks:
            block_size = block_end - block_start
            if block_size > 100000 or block_size < 300:
                continue

            # Skip blocks not near existing satellite detections
            if not np.any(near_satellite[block_start:block_end]):
                continue

            # Autocorrelation-based satellite detection
            # Scan multiple windows: start, middle, end of block
            windows = [(block_start, min(block_start + 5000, block_end))]
            if block_size > 5000:
                mid = block_start + block_size // 2 - 2500
                windows.append((mid, min(mid + 5000, block_end)))
                windows.append((max(block_start, block_end - 5000), block_end))

            best_period = 0
            best_identity = 0.0
            best_w_start = block_start

            for w_start, w_end in windows:
                w_region = text_arr[w_start:w_end]
                w_size = len(w_region)
                if w_size < 300:
                    continue

                for p in range(self.sat_fill_min_period, min(self.sat_fill_max_period + 1, w_size // 2)):
                    total = w_size - p
                    if total <= 0:
                        continue
                    matches = int(np.sum(w_region[:total] == w_region[p:p + total]))
                    identity = matches / total
                    if identity > best_identity:
                        best_identity = identity
                        best_period = p
                        best_w_start = w_start
                        if identity > 0.80:
                            break
                if best_identity > 0.80:
                    break

            if best_identity < self.sat_fill_min_identity or best_period < 50:
                continue

            use_motif = text_arr[best_w_start:best_w_start + best_period].tobytes().decode('ascii', errors='replace')
            copies = block_size / best_period

            new_tr = TandemRepeat(
                chrom=self.chromosome,
                start=block_start, end=block_end,
                motif=use_motif, copies=copies,
                length=block_size, consensus_motif=use_motif,
                mismatch_rate=1.0 - best_identity, tier="satellite",
            )
            new_repeats.append(new_tr)

        if new_repeats:
            filled = list(repeats) + new_repeats
            filled.sort(key=lambda x: x.start)
            return filled

        return repeats

    def cleanup(self):
        """Release resources."""
        self.bwt.clear()  # Release memory used by the BWT/FM-index

    def _normalize_tiers(self, tiers: Optional[Set[str]]) -> Set[str]:
        if not tiers:
            return set(self.VALID_TIERS)  # If no tiers specified, enable all valid tiers

        normalized: Set[str] = set()  # Set of normalized tier names
        for tier in tiers:
            if not tier:
                continue  # Skip empty strings
            name = tier.strip().lower()  # Normalize by stripping whitespace and converting to lowercase
            if name == "all":
                return set(self.VALID_TIERS)  # If "all", return all valid tiers
            if name in self.VALID_TIERS:
                normalized.add(name)  # Add to set if it is a valid tier name

        return normalized if normalized else set(self.VALID_TIERS)  # If no valid tiers found, return all tiers

    def _register_repeat(self, repeat: TandemRepeat, store: List[TandemRepeat],
                         seen: Optional[Set[Tuple[int, int]]] = None) -> bool:
        if not self._repeat_within_bounds(repeat):
            return False  # Reject repeats outside user-specified bounds

        store.append(repeat)  # Add repeat to the global list
        if seen is not None:
            seen.add((repeat.start, repeat.end))  # Register (start, end) coordinates in deduplication set
        return True  # Return registration success

    def _repeat_within_bounds(self, repeat: TandemRepeat) -> bool:
        motif = repeat.motif or repeat.consensus_motif  # Use motif or consensus motif
        motif_len = len(motif) if motif else 0  # Calculate motif length (0 if absent)
        if motif_len <= 0:
            motif_len = max(1, repeat.length)  # If motif length is 0, use array length instead

        if motif_len < self.min_period or motif_len > self.max_period:
            return False  # False if motif length is outside allowed range

        length = repeat.length if repeat.length else repeat.end - repeat.start  # Calculate total array length

        if self.min_array_bp and length < self.min_array_bp:
            return False  # False if array length is below minimum bp
        if self.max_array_bp and length > self.max_array_bp:
            return False  # False if array length exceeds maximum bp

        return True  # True if all conditions are met
