#!/usr/bin/env python3
"""
Advanced BWT-based Tandem Repeat Finder for Genomics

Implements three-tier approach:
1. Short tandem repeats (1-10bp) with FM-index
2. Medium/long repeats (10s-1000s bp) with LCP arrays  
3. Very long repeats (kb+) with long read evidence
"""

import numpy as np
from typing import List, Tuple, Dict, Iterator, Optional, Set
import argparse
from dataclasses import dataclass

# Optional: JIT acceleration with numba when available
HAVE_NUMBA = False
try:
    import numba as _nb  # type: ignore
    HAVE_NUMBA = True
except Exception:
    _nb = None  # type: ignore

if HAVE_NUMBA:
    @_nb.njit(cache=True)
    def _count_equal_range(arr: np.ndarray, start: int, end: int, code: int) -> int:  # type: ignore
        c = 0
        for i in range(start, end):
            if arr[i] == code:
                c += 1
        return c

    @_nb.njit(cache=True)
    def _kasai_lcp_uint8(text_codes: np.ndarray, sa: np.ndarray) -> np.ndarray:  # type: ignore
        n = text_codes.size
        lcp = np.zeros(n, dtype=np.int32)
        rank = np.zeros(n, dtype=np.int32)
        for i in range(n):
            rank[sa[i]] = i
        h = 0
        for i in range(n):
            r = rank[i]
            if r > 0:
                j = sa[r - 1]
                while i + h < n and j + h < n and text_codes[i + h] == text_codes[j + h]:
                    h += 1
                lcp[r] = h
                if h > 0:
                    h -= 1
        return lcp
else:
    def _count_equal_range(arr: np.ndarray, start: int, end: int, code: int) -> int:
        # Pure-python/numpy fallback
        return int(np.count_nonzero(arr[start:end] == code))

    def _kasai_lcp_uint8(text_codes: np.ndarray, sa: np.ndarray) -> np.ndarray:
        # Fallback non-jitted Kasai
        n = text_codes.size
        lcp = np.zeros(n, dtype=np.int32)
        rank = np.zeros(n, dtype=np.int32)
        for i in range(n):
            rank[sa[i]] = i
        h = 0
        for i in range(n):
            r = rank[i]
            if r > 0:
                j = sa[r - 1]
                while i + h < n and j + h < n and text_codes[i + h] == text_codes[j + h]:
                    h += 1
                lcp[r] = h
                if h > 0:
                    h -= 1
        return lcp


class BWTCore:
    """Core BWT construction and FM-index operations.

    Performance notes:
    - Uses pydivsufsort for suffix array when available (C backend).
    - Stores text and BWT as NumPy uint8 arrays (ASCII codes) for vectorized rank.
    - Keeps original text string for compatibility with higher tiers.
    """

    def __init__(self, text: str, sa_sample_rate: int = 32, occ_sample_rate: int = 128):
        """
        Initialize BWT with FM-index.

        Args:
            text: Input text (should end with a single '$' sentinel not present elsewhere)
            sa_sample_rate: Sample every nth suffix array position for space efficiency
            occ_sample_rate: Occurrence checkpoints every nth position to reduce memory
        """
        # Keep string for external consumers, but build numeric views for speed
        self.text: str = text
        self.n = len(text)
        self.sa_sample_rate = sa_sample_rate   
        self.occ_sample_rate = occ_sample_rate

        # Numeric views (ASCII codes)
        self.text_arr: np.ndarray = np.frombuffer(text.encode('ascii'), dtype=np.uint8)

        # Build suffix array and BWT (memory-efficient)
        self.suffix_array = self._build_suffix_array()
        # BWT stored as uint8 array; also expose lightweight string view on demand
        self.bwt_arr: np.ndarray = self._build_bwt_array()

        # Build alphabet/mappings from the original text
        self.alphabet = sorted(set(text))
        # Char<->code maps (ASCII)
        self.char_to_code: Dict[str, int] = {c: ord(c) for c in self.alphabet}
        self.code_to_char: Dict[int, str] = {ord(c): c for c in self.alphabet}

        # FM-index components
        self.char_counts, self.char_totals = self._build_char_counts()
        # Code-keyed variants for internal numeric operations
        self.char_counts_code: Dict[int, int] = {ord(k): v for k, v in self.char_counts.items()}
        self.char_totals_code: Dict[int, int] = {ord(k): v for k, v in self.char_totals.items()}

        # Occurrence checkpoints for rank queries (code -> np.ndarray)
        self.occ_checkpoints = self._build_occurrence_checkpoints()

        # Sample suffix array for efficient locating
        self.sampled_sa = self._sample_suffix_array()

    def clear(self):
        """Release heavy memory structures to let GC reclaim memory."""
        # Replace large attributes with minimal stubs
        self.text = ""
        self.text_arr = np.array([], dtype=np.uint8)
        self.bwt_arr = np.array([], dtype=np.uint8)
        self.suffix_array = np.array([], dtype=np.int32)
        self.sampled_sa = {}
        self.occ_checkpoints = {}
        self.char_counts = {}
        self.char_totals = {}
        self.alphabet = []
        self.char_to_code = {}
        self.code_to_char = {}
        self.char_counts_code = {}
        self.char_totals_code = {}
    
    def _build_suffix_array(self) -> np.ndarray:
        """Build suffix array, preferring pydivsufsort (C backend) with a NumPy fallback.

        Fallback uses prefix-doubling with NumPy lexsort (significantly faster than
        Python list.sort + lambdas). Complexity ~O(n log n) sorts.
        """
        # Prefer fast C implementation when available
        s = self.text
        try:
            import pydivsufsort  # type: ignore
            sa_list = pydivsufsort.divsufsort(s)
            return np.array(sa_list, dtype=np.int32)
        except Exception:
            print("[WARN] pydivsufsort not available; using NumPy-based suffix array fallback.")

        n = self.n
        if n == 0:
            return np.array([], dtype=np.int32)

        # Initial rank from character codes
        codes = self.text_arr.astype(np.int32, copy=False)
        # Compress codes to 0..sigma-1 for stability
        uniq_codes, inv = np.unique(codes, return_inverse=True)
        rank = inv.astype(np.int32, copy=False)
        sa = np.arange(n, dtype=np.int32)

        k = 1
        tmp_rank = np.empty(n, dtype=np.int32)
        idx = np.arange(n, dtype=np.int32)
        while k < n:
            # secondary key is rank[i+k] else -1
            ipk = idx + k
            key2 = np.where(ipk < n, rank[ipk], -1)
            # Sort by (rank[i], key2[i]) using lexsort with primary last
            sa = np.lexsort((key2, rank))
            # Compute new ranks
            r_sa = rank[sa]
            k2_sa = key2[sa]
            # mark changes
            change = np.empty(n, dtype=np.int32)
            change[0] = 0
            change[1:] = (r_sa[1:] != r_sa[:-1]) | (k2_sa[1:] != k2_sa[:-1])
            new_rank_ordered = np.cumsum(change, dtype=np.int32)
            # remap to original index order
            tmp_rank[sa] = new_rank_ordered
            rank, tmp_rank = tmp_rank, rank
            if rank[sa[-1]] == n - 1:
                break
            k <<= 1
        return sa.astype(np.int32, copy=False)
    
    def _build_bwt_array(self) -> np.ndarray:
        """Build BWT from suffix array as uint8 NumPy array (ASCII codes)."""
        if self.n == 0:
            return np.array([], dtype=np.uint8)
        sa = self.suffix_array.astype(np.int64, copy=False)
        # previous index (sa-1) % n
        prev_idx = (sa - 1) % self.n
        # Gather from numeric text array
        return self.text_arr[prev_idx]
    
    def _build_char_counts(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Count character frequencies and compute cumulative counts C[char]."""
        totals: Dict[str, int] = {c: 0 for c in self.alphabet}
        for ch in self.text:
            totals[ch] += 1
        counts: Dict[str, int] = {}
        cumulative = 0
        for char in self.alphabet:
            counts[char] = cumulative
            cumulative += totals[char]
        return counts, totals
    
    def _build_occurrence_checkpoints(self) -> Dict[int, np.ndarray]:
        """Build checkpointed occurrence counts for efficient rank queries with low memory.

        Returns a mapping from ASCII code -> np.ndarray of counts at positions m*k
        (prefix length), with cp[0] = 0. If the last block is partial, a final
        checkpoint with the total count at n is appended to mirror previous behavior.
        """
        bwt = self.bwt_arr
        n = bwt.size
        k = int(self.occ_sample_rate)
        if n == 0:
            return {}

        checkpoints: Dict[int, np.ndarray] = {}
        # Precompute full cumsum once per distinct code as we have small alphabets
        distinct_codes = np.unique(bwt)
        # indices where boundaries end (1-based length m*k corresponds to index m*k-1)
        block_ends = np.arange(k - 1, n, k, dtype=np.int64)

        for code in distinct_codes.tolist():
            mask = (bwt == code)
            csum = np.cumsum(mask, dtype=np.int32)
            # cp[0]=0, then take counts at each block end
            cp_list = [0]
            if block_ends.size:
                cp_list.extend(csum[block_ends].tolist())
            # Optionally append final count for partial block remainder
            if n % k != 0:
                cp_list.append(int(csum[-1]))
            checkpoints[int(code)] = np.asarray(cp_list, dtype=np.int32)
        # Ensure every alphabet character has a checkpoint array (even if absent)
        for c in self.alphabet:
            code = ord(c)
            if code not in checkpoints:
                # Build an all-zeros checkpoint array of same length as others
                # Determine representative length from any existing array
                any_cp = next(iter(checkpoints.values())) if checkpoints else np.array([0], dtype=np.int32)
                checkpoints[code] = np.zeros_like(any_cp)
        return checkpoints
    
    def _sample_suffix_array(self) -> Dict[int, int]:
        """Sample suffix array positions for space-efficient locating."""
        sampled = {}
        for i in range(0, self.n, self.sa_sample_rate):
            sampled[i] = self.suffix_array[i]
        return sampled
    
    def rank(self, char: str | int, pos: int) -> int:
        """Count occurrences of `char` in bwt[0:pos]. Vectorized with checkpoints.

        Args:
            char: character (str) or ASCII code (int) to count
            pos: count occurrences in bwt[0:pos] (pos can be 0..n)
        """
        if pos <= 0:
            return 0
        if pos > self.n:
            pos = self.n
        code = ord(char) if isinstance(char, str) else int(char)
        cp = self.occ_checkpoints.get(code)
        if cp is None:
            return 0
        k = int(self.occ_sample_rate)
        cp_idx = pos // k
        cp_pos = cp_idx * k
        base = int(cp[cp_idx])
        # Fast remainder scan (Numba-accelerated if available)
        if pos > cp_pos:
            base += int(_count_equal_range(self.bwt_arr, cp_pos, pos, code))
        return base
    
    def backward_search(self, pattern: str) -> Tuple[int, int]:
        """
        Find suffix array interval for pattern using backward search.
        
        Returns:
            (start, end) interval in suffix array, or (-1, -1) if not found
        """
        if not pattern:
            return (0, self.n - 1)
        
        # Initialize with character range
        char = pattern[-1]
        if char not in self.char_counts:
            return (-1, -1)
        # sp inclusive, ep inclusive
        sp = self.char_counts[char]
        ep = sp + self.char_totals[char] - 1
        
        # Process pattern right to left
        for i in range(len(pattern) - 2, -1, -1):
            char = pattern[i]
            if char not in self.char_counts:
                return (-1, -1)

            sp = self.char_counts[char] + self.rank(char, sp)
            ep = self.char_counts[char] + self.rank(char, ep + 1) - 1

            if sp > ep:
                return (-1, -1)
        
        return (sp, ep)
    
    def count_occurrences(self, pattern: str) -> int:
        """Count pattern occurrences in text."""
        sp, ep = self.backward_search(pattern)
        if sp == -1:
            return 0
        return ep - sp + 1
    
    def locate_positions(self, pattern: str) -> List[int]:
        """
        Locate all positions of pattern in text.
        Uses sampled suffix array for efficiency.
        """
        sp, ep = self.backward_search(pattern)
        if sp == -1:
            return []

        # Directly read positions from the suffix array (much faster than LF walking)
        positions = self.suffix_array[sp:ep + 1].tolist()
        positions.sort()
        return positions
    
    def _get_suffix_position(self, sa_index: int) -> int:
        """Recover original text position from SA index using sampling."""
        if sa_index in self.sampled_sa:
            return self.sampled_sa[sa_index]
        
        # Walk using LF mapping until we hit a sampled position
        steps = 0
        current_idx = sa_index
        
        while current_idx not in self.sampled_sa:
            code = int(self.bwt_arr[current_idx])
            current_idx = self.char_counts_code[code] + self.rank(code, current_idx)
            steps += 1
        
        return (self.sampled_sa[current_idx] + steps) % self.n


@dataclass
class TandemRepeat:
    """Represents a tandem repeat finding."""
    chrom: str
    start: int
    end: int
    motif: str
    copies: float
    length: int
    tier: int
    confidence: float = 1.0
    
    def to_bed(self) -> str:
        """Convert to BED format."""
        return f"{self.chrom}\t{self.start}\t{self.end}\t{self.motif}\t{self.copies:.1f}\t{self.tier}"
    
    def to_vcf_info(self) -> str:
        """Convert to VCF INFO field."""
        return f"MOTIF={self.motif};COPIES={self.copies:.1f};TIER={self.tier};CONF={self.confidence:.2f}"


class MotifUtils:
    """Utilities for canonical motif handling."""
    
    @staticmethod
    def get_canonical_motif(motif: str) -> str:
        """Get lexicographically smallest rotation of motif."""
        if not motif:
            return motif
        
        rotations = [motif[i:] + motif[:i] for i in range(len(motif))]
        return min(rotations)
    
    @staticmethod
    def is_primitive_motif(motif: str) -> bool:
        """Check if motif is not a repetition of a shorter motif."""
        n = len(motif)
        for i in range(1, n):
            if n % i == 0:
                period = motif[:i]
                if period * (n // i) == motif:
                    return False
        return True
    
    @staticmethod
    def enumerate_motifs(k: int, alphabet: str = "ACGT") -> Iterator[str]:
        """Generate all canonical primitive motifs of length k."""
        def generate_strings(length, current=""):
            if length == 0:
                canonical = MotifUtils.get_canonical_motif(current)
                if canonical == current and MotifUtils.is_primitive_motif(current):
                    yield current
                return
            
            for char in alphabet:
                yield from generate_strings(length - 1, current + char)
        
        yield from generate_strings(k)


# Implementation continues with Tier 1, 2, 3 classes...


class Tier1STRFinder:
    """Tier 1: Short Tandem Repeat Finder using FM-index."""
    
    def __init__(self, bwt_core: BWTCore, max_motif_length: int = 6, show_progress: bool = False):
        self.bwt = bwt_core
        self.max_motif_length = max_motif_length
        self.min_copies = 2
        self.show_progress = show_progress
    
    def find_strs(self, chromosome: str) -> List[TandemRepeat]:
        """Find short tandem repeats (1-10bp motifs)."""
        repeats = []

        for k in range(1, self.max_motif_length + 1):
            motif_count = 0

            for motif in MotifUtils.enumerate_motifs(k):
                motif_count += 1
                # Quiet: avoid verbose progress prints
                # Fast counting with backward search
                count = self.bwt.count_occurrences(motif)
                
                if count >= self.min_copies:
                    # Get positions and check for tandem structure
                    positions = self.bwt.locate_positions(motif)
                    tandem_repeats = self._find_tandems_in_positions(
                        positions, motif, chromosome, k
                    )
                    repeats.extend(tandem_repeats)
        
        return repeats
    
    def _find_tandems_in_positions(self, positions: List[int], motif: str, 
                                 chromosome: str, motif_len: int) -> List[TandemRepeat]:
        """Find tandem repeats from motif positions."""
        repeats = []
        if not positions:
            return repeats
        
        positions.sort()
        i = 0
        
        while i < len(positions):
            start_pos = positions[i]
            copies = 1
            current_pos = start_pos
            
            # Extend run of consecutive tandem copies
            j = i + 1
            while j < len(positions):
                expected_pos = current_pos + motif_len
                if positions[j] == expected_pos:
                    copies += 1
                    current_pos = positions[j]
                    j += 1
                else:
                    break
            
            if copies >= self.min_copies:
                # Check maximality
                end_pos = start_pos + copies * motif_len
                if self._is_maximal_repeat(start_pos, end_pos, motif, motif_len):
                    repeat = TandemRepeat(
                        chrom=chromosome,
                        start=start_pos,
                        end=end_pos,
                        motif=motif,
                        copies=copies,
                        length=copies * motif_len,
                        tier=1,
                        confidence=1.0
                    )
                    repeats.append(repeat)
            
            i = j if j > i + 1 else i + 1
        
        return repeats
    
    def _is_maximal_repeat(self, start: int, end: int, motif: str, motif_len: int) -> bool:
        """Check if repeat is maximal (cannot be extended)."""
        # Check left extension
        if start > 0:
            left_char = self.bwt.text[start - 1]
            expected_left = motif[-1]  # Last char of motif
            if left_char == expected_left:
                return False
        
        # Check right extension
        if end < len(self.bwt.text):
            right_char = self.bwt.text[end]
            expected_right = motif[0]  # First char of motif
            if right_char == expected_right:
                return False
        
        return True


class Tier2LCPFinder:
    """Tier 2: Medium/Long Tandem Repeat Finder using LCP arrays."""
    
    def __init__(self, bwt_core: BWTCore, min_period: int = 10, max_period: int = 1000, show_progress: bool = False):
        self.bwt = bwt_core
        self.min_period = min_period
        self.max_period = max_period
        self.min_copies = 2
        self.show_progress = show_progress
    
    def find_long_repeats(self, chromosome: str) -> List[TandemRepeat]:
        """Find medium to long tandem repeats using a lightweight period scan.

        This avoids building large LCP structures and is fast for moderate sequences.
        """
        return self._find_repeats_simple(chromosome)
    
    def _compute_lcp_array(self) -> np.ndarray:
        """Compute LCP array using Kasai over uint8 codes (Numba-accelerated when available)."""
        n = self.bwt.n
        if n == 0:
            return np.zeros(0, dtype=np.int32)
        # Use the text codes directly for fast comparisons
        text_codes = self.bwt.text_arr
        sa = self.bwt.suffix_array.astype(np.int32, copy=False)
        return _kasai_lcp_uint8(text_codes, sa)
    
    def _detect_lcp_plateaus(self, lcp_array: np.ndarray, chromosome: str) -> List[TandemRepeat]:
        """Detect tandem repeats from LCP plateaus."""
        repeats = []
        n = len(lcp_array)
        if n == 0:
            return repeats
        # Choose a single conservative threshold: max(min_period, 20), but <= max LCP and <= max_period
        lcp_max = int(np.max(lcp_array))
        if lcp_max < self.min_period:
            return repeats
        threshold = min(self.max_period, lcp_max)
        threshold = max(self.min_period, min(threshold, 20))

        i = 0
        while i < n:
            if lcp_array[i] >= threshold:
                j = i
                while j < n and lcp_array[j] >= threshold:
                    j += 1
                tandem_repeats = self._analyze_sa_interval_for_tandems(
                    i, j, threshold, chromosome
                )
                repeats.extend(tandem_repeats)
                i = j
            else:
                i += 1

        return repeats

    def _smallest_period(self, s: str) -> int:
        """Return the length of the smallest period of s via prefix-function (KMP)."""
        n = len(s)
        pi = [0] * n
        for i in range(1, n):
            j = pi[i - 1]
            while j > 0 and s[i] != s[j]:
                j = pi[j - 1]
            if s[i] == s[j]:
                j += 1
            pi[i] = j
        p = n - pi[-1]
        return p if p != 0 and n % p == 0 else n
    
    def _smallest_period_codes(self, arr: np.ndarray) -> int:
        """Smallest period for a uint8 array using prefix-function (no strings)."""
        n = int(arr.size)
        if n == 0:
            return 0
        pi = np.zeros(n, dtype=np.int32)
        j = 0
        for i in range(1, n):
            while j > 0 and arr[i] != arr[j]:
                j = int(pi[j - 1])
            if arr[i] == arr[j]:
                j += 1
            pi[i] = j
        p = n - int(pi[-1])
        return p if p != 0 and n % p == 0 else n

    def _find_repeats_simple(self, chromosome: str) -> List[TandemRepeat]:
        """Simple scanning detector for tandem repeats using candidate periods.

        Designed for correctness on small/medium sequences without heavy indexes.
        """
        s_arr = self.bwt.text_arr
        n = int(s_arr.size)
        # Exclude trailing sentinel if present ('$' == 36)
        if n > 0 and s_arr[n - 1] == 36:
            n -= 1
        max_p = min(self.max_period, max(64, self.min_period), max(1, n // 2))
        min_p = self.min_period
        results: List[TandemRepeat] = []
        seen: Set[Tuple[int, int, str]] = set()

        for p in range(min_p, max_p + 1):
            i = 0
            while i + 2 * p <= n:
                motif_view = s_arr[i:i + p]
                # Skip motifs containing '$'(36) or 'N'(78)
                if np.any(motif_view == 36) or np.any(motif_view == 78):
                    i += 1
                    continue
                copies = 1
                j = i + p
                # Extend with array comparisons (views only, no string slicing)
                while j + p <= n and np.array_equal(s_arr[j:j + p], motif_view):
                    copies += 1
                    j += p
                if copies >= self.min_copies:
                    # Normalize motif to its primitive period
                    prim = self._smallest_period_codes(motif_view)
                    if prim < p:
                        motif_view = motif_view[:prim]
                        # adjust p to primitive
                        p_eff = prim
                    else:
                        p_eff = p
                    # Maximality: extend left/right by whole motifs
                    start = i
                    end = j
                    while start - p_eff >= 0 and np.array_equal(s_arr[start - p_eff:start], motif_view):
                        start -= p_eff
                        copies += 1
                    while end + p_eff <= n and np.array_equal(s_arr[end:end + p_eff], motif_view):
                        end += p_eff
                        copies += 1
                    motif_str = motif_view.tobytes().decode('ascii')
                    key = (start, end, motif_str)
                    if key not in seen:
                        seen.add(key)
                        results.append(
                            TandemRepeat(
                                chrom=chromosome,
                                start=start,
                                end=end,
                                motif=motif_str,
                                copies=float((end - start) // len(motif_view)),
                                length=end - start,
                                tier=2,
                                confidence=0.95,
                            )
                        )
                    i = end  # jump past this repeat
                else:
                    i += 1

        return results
    
    def _analyze_sa_interval_for_tandems(self, start_idx: int, end_idx: int, 
                                       period: int, chromosome: str) -> List[TandemRepeat]:
        """Analyze suffix array interval for tandem structure."""
        repeats = []
        
        # Get suffix positions in this interval
        positions = []
        for i in range(start_idx, end_idx):
            positions.append(self.bwt.suffix_array[i])
        
        positions.sort()
        
        # Look for arithmetic progressions with difference = period
        for i in range(len(positions)):
            copies = 1
            start_pos = positions[i]
            
            # Count consecutive positions differing by period
            j = i + 1
            while j < len(positions):
                if positions[j] == start_pos + copies * period:
                    copies += 1
                    j += 1
                else:
                    break
            
            if copies >= self.min_copies:
                # Extract and validate motif using arrays
                motif_start = start_pos
                motif_end = motif_start + period
                if motif_end <= int(self.bwt.text_arr.size):
                    motif_arr = self.bwt.text_arr[motif_start:motif_end]
                    total_length = copies * period
                    rep_end = start_pos + total_length
                    repeat_arr = self.bwt.text_arr[start_pos:rep_end]
                    if self._validate_periodicity_arr(repeat_arr, motif_arr, period):
                        motif_str = motif_arr.tobytes().decode('ascii')
                        repeat = TandemRepeat(
                            chrom=chromosome,
                            start=start_pos,
                            end=rep_end,
                            motif=motif_str,
                            copies=copies,
                            length=total_length,
                            tier=2,
                            confidence=0.9  # High confidence from LCP
                        )
                        repeats.append(repeat)
        
        return repeats
    
    def _validate_periodicity_arr(self, text_arr: np.ndarray, motif_arr: np.ndarray, period: int) -> bool:
        """Validate periodic structure by vectorized uint8 comparison."""
        m = text_arr.size
        if m < 2 * period:
            return False
        idx = np.arange(m, dtype=np.int32) % period
        # Compare each position to motif at idx
        matches = np.count_nonzero(text_arr == motif_arr[idx])
        similarity = matches / m if m > 0 else 0.0
        return bool(similarity >= 0.8)


class Tier3LongReadFinder:
    """Tier 3: Very Long Tandem Repeat Finder using long read evidence."""
    
    def __init__(self, bwt_core: BWTCore, show_progress: bool = False):
        self.bwt = bwt_core
        self.min_read_length = 1000  # Minimum read length for analysis
        self.min_span_length = 100   # Minimum repeat length to analyze
        self.show_progress = show_progress
    
    def find_very_long_repeats(self, long_reads: List[str], chromosome: str) -> List[TandemRepeat]:
        """Find very long tandem repeats using long read evidence."""
        repeats = []
        
        for read_idx, read in enumerate(long_reads):
            if len(read) < self.min_read_length:
                continue
            
            # Find potential repeat regions in read
            read_repeats = self._analyze_read_for_repeats(read, chromosome, read_idx)
            repeats.extend(read_repeats)
        
        # Consolidate overlapping repeat calls
        return self._consolidate_repeat_calls(repeats)
    
    def _analyze_read_for_repeats(self, read: str, chromosome: str, read_idx: int) -> List[TandemRepeat]:
        """Analyze a single long read for tandem repeats."""
        repeats = []
        
        # Sliding window approach to find repetitive regions
        window_size = 500
        step_size = 100
        
        for start in range(0, len(read) - window_size, step_size):
            window = read[start:start + window_size]
            
            # Check if window contains repetitive structure
            repeat_info = self._detect_repetitive_structure(window)
            
            if repeat_info:
                motif, copies, confidence = repeat_info
                
                # Map read position to reference (simplified - would need actual alignment)
                ref_start = self._map_read_to_reference(read, start, chromosome)
                
                if ref_start >= 0:
                    repeat = TandemRepeat(
                        chrom=chromosome,
                        start=ref_start,
                        end=ref_start + len(motif) * copies,
                        motif=motif,
                        copies=copies,
                        length=len(motif) * copies,
                        tier=3,
                        confidence=confidence
                    )
                    repeats.append(repeat)
        
        return repeats
    
    def _detect_repetitive_structure(self, sequence: str) -> Optional[Tuple[str, int, float]]:
        """Detect repetitive structure in sequence using autocorrelation."""
        if len(sequence) < 50:
            return None
        
        # Simple approach: try different period lengths
        best_period = None
        best_copies = 0
        best_score = 0
        
        for period_len in range(10, len(sequence) // 3):
            motif = sequence[:period_len]
            score = self._score_periodicity(sequence, motif, period_len)
            
            if score > best_score and score > 0.7:
                best_score = score
                best_period = motif
                best_copies = len(sequence) // period_len
        
        if best_period and best_copies >= 3:
            return (best_period, best_copies, best_score)
        
        return None
    
    def _score_periodicity(self, sequence: str, motif: str, period: int) -> float:
        """Score how well sequence matches periodic pattern."""
        matches = 0
        total = 0
        
        for i in range(len(sequence)):
            motif_pos = i % period
            if motif_pos < len(motif):
                total += 1
                if sequence[i] == motif[motif_pos]:
                    matches += 1
        
        return matches / total if total > 0 else 0
    
    def _map_read_to_reference(self, read: str, position: int, chromosome: str) -> int:
        """Map read position to reference coordinates (simplified)."""
        # This is a simplified implementation
        # In practice, would use proper read alignment
        
        # Try to find a unique anchor sequence around the position
        anchor_len = 50
        anchor_start = max(0, position - anchor_len)
        anchor = read[anchor_start:position]
        
        if len(anchor) >= 20:
            positions = self.bwt.locate_positions(anchor)
            if len(positions) == 1:  # Unique hit
                return positions[0] + (position - anchor_start)
        
        return -1  # Could not map
    
    def _consolidate_repeat_calls(self, repeats: List[TandemRepeat]) -> List[TandemRepeat]:
        """Consolidate overlapping repeat calls from multiple reads."""
        if not repeats:
            return repeats
        
        # Sort by position
        repeats.sort(key=lambda r: (r.chrom, r.start, r.end))
        
        consolidated = []
        current = repeats[0]
        
        for repeat in repeats[1:]:
            # Check for overlap
            if (repeat.chrom == current.chrom and 
                repeat.start <= current.end and 
                repeat.motif == current.motif):
                
                # Merge overlapping repeats
                current = TandemRepeat(
                    chrom=current.chrom,
                    start=min(current.start, repeat.start),
                    end=max(current.end, repeat.end),
                    motif=current.motif,
                    copies=(current.copies + repeat.copies) / 2,  # Average
                    length=max(current.end, repeat.end) - min(current.start, repeat.start),
                    tier=current.tier,
                    confidence=min(current.confidence, repeat.confidence)
                )
            else:
                consolidated.append(current)
                current = repeat
        
        consolidated.append(current)
        return consolidated


class TandemRepeatFinder:
    """Main class coordinating all three tiers of tandem repeat finding."""
    
    def __init__(self, reference_file: str, sa_sample_rate: int = 32, show_progress: bool = False):
        """
        Initialize the tandem repeat finder.
        
        Args:
            reference_file: Path to reference genome FASTA
            sa_sample_rate: Suffix array sampling rate for space efficiency
        """
        self.reference_file = reference_file
        self.sa_sample_rate = sa_sample_rate
        self.bwt_cores = {}  # Store BWT for each chromosome
        self.show_progress = show_progress
    
    def load_reference(self) -> Dict[str, str]:
        """Load reference sequences from FASTA file."""
        sequences = {}
        current_chrom = None
        current_seq = []

        with open(self.reference_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_chrom:
                        sequences[current_chrom] = ''.join(current_seq)
                    
                    current_chrom = line[1:].split()[0]  # Extract chromosome name
                    current_seq = []
                elif line:
                    current_seq.append(line.upper())
        
        if current_chrom:
            sequences[current_chrom] = ''.join(current_seq)
        
        return sequences
    
    def build_indices(self, sequences: Dict[str, str]):
        """Build BWT and FM-index for each chromosome."""
        items = list(sequences.items())
        for chrom, seq in items:
            # Add a single sentinel character at the end (must not appear elsewhere)
            seq_with_sentinel = seq + "$"

            # Build BWT core
            bwt_core = BWTCore(seq_with_sentinel, self.sa_sample_rate)
            self.bwt_cores[chrom] = bwt_core
    
    def find_tandem_repeats(self, enable_tier1: bool = True, enable_tier2: bool = True, 
                          enable_tier3: bool = False, long_reads: Optional[List[str]] = None) -> List[TandemRepeat]:
        """
        Find tandem repeats using enabled tiers.
        
        Args:
            enable_tier1: Enable short tandem repeat finding
            enable_tier2: Enable medium/long repeat finding  
            enable_tier3: Enable very long repeat finding
            long_reads: Long reads for tier 3 analysis
        """
        all_repeats = []
        
        items = list(self.bwt_cores.items())
        for chrom, bwt_core in items:
            print(f"\nAnalyzing chromosome {chrom}...")
            
            if enable_tier1:
                tier1 = Tier1STRFinder(bwt_core, show_progress=self.show_progress)
                tier1_repeats = tier1.find_strs(chrom)
                all_repeats.extend(tier1_repeats)
                print(f"  Tier 1: {len(tier1_repeats)} STRs")
            
            if enable_tier2:
                tier2 = Tier2LCPFinder(bwt_core, show_progress=self.show_progress)
                tier2_repeats = tier2.find_long_repeats(chrom)
                all_repeats.extend(tier2_repeats)
                print(f"  Tier 2: {len(tier2_repeats)} LTRs")
            
            if enable_tier3 and long_reads:
                tier3 = Tier3LongReadFinder(bwt_core, show_progress=self.show_progress)
                tier3_repeats = tier3.find_very_long_repeats(long_reads, chrom)
                all_repeats.extend(tier3_repeats)
                print(f"  Tier 3: {len(tier3_repeats)} VLTRs")

            # Free per-chromosome memory ASAP
            bwt_core.clear()
        
        return all_repeats
    
    def save_results(self, repeats: List[TandemRepeat], output_file: str, format_type: str = "bed"):
        """Save tandem repeat results to file."""
        with open(output_file, 'w') as f:
            if format_type == "bed":
                f.write("# Tandem Repeats (BED format)\n")
                f.write("# chrom\tstart\tend\tmotif\tcopies\ttier\n")
                for repeat in repeats:
                    f.write(repeat.to_bed() + "\n")
            
            elif format_type == "vcf":
                f.write("##fileformat=VCFv4.2\n")
                f.write("##INFO=<ID=MOTIF,Number=1,Type=String,Description=\"Repeat motif\">\n")
                f.write("##INFO=<ID=COPIES,Number=1,Type=Float,Description=\"Number of copies\">\n")
                f.write("##INFO=<ID=TIER,Number=1,Type=Integer,Description=\"Detection tier\">\n")
                f.write("##INFO=<ID=CONF,Number=1,Type=Float,Description=\"Confidence score\">\n")
                f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
                
                for i, repeat in enumerate(repeats):
                    f.write(f"{repeat.chrom}\t{repeat.start + 1}\tTR{i}\t.\t<TR>\t.\tPASS\t{repeat.to_vcf_info()}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Advanced BWT-based Tandem Repeat Finder")
    parser.add_argument("reference", help="Reference genome FASTA file")
    parser.add_argument("-o", "--output", default="tandem_repeats.bed", help="Output file")
    parser.add_argument("--format", choices=["bed", "vcf"], default="bed", help="Output format")
    parser.add_argument("--tier1", action="store_true", help="Enable tier 1 (short repeats)")
    parser.add_argument("--tier2", action="store_true", help="Enable tier 2 (medium/long repeats)")
    parser.add_argument("--tier3", action="store_true", help="Enable tier 3 (very long repeats)")
    parser.add_argument("--long-reads", help="Long reads file for tier 3")
    parser.add_argument("--sa-sample", type=int, default=32, help="Suffix array sampling rate")
    parser.add_argument("--progress", action="store_true", help="Show progress bars where applicable")
    
    args = parser.parse_args()
    
    # Initialize finder
    finder = TandemRepeatFinder(args.reference, args.sa_sample, show_progress=args.progress)
    
    # Load reference and build indices
    sequences = finder.load_reference()
    finder.build_indices(sequences)
    
    # Load long reads if provided
    long_reads = []
    if args.long_reads and args.tier3:
        print(f"Loading long reads from {args.long_reads}...")
        # Simple FASTA/FASTQ reader
        with open(args.long_reads, 'r') as f:
            seq = ""
            for line in f:
                line = line.strip()
                if line.startswith('>') or line.startswith('@'):
                    if seq:
                        long_reads.append(seq)
                        seq = ""
                elif not line.startswith('+'):
                    seq += line.upper()
            if seq:
                long_reads.append(seq)
        print(f"Loaded {len(long_reads)} long reads")
    
    # Find tandem repeats
    # If no tiers were explicitly enabled, default to Tier1+Tier2 for convenience
    enable_tier1 = args.tier1
    enable_tier2 = args.tier2
    enable_tier3 = args.tier3
    if not (enable_tier1 or enable_tier2 or enable_tier3):
        enable_tier1 = True
        enable_tier2 = True

    repeats = finder.find_tandem_repeats(
        enable_tier1=enable_tier1,
        enable_tier2=enable_tier2,
        enable_tier3=enable_tier3,
        long_reads=long_reads if enable_tier3 and long_reads else None
    )
    
    # Save results
    finder.save_results(repeats, args.output, args.format)
    
    print(f"\nCompleted! Found {len(repeats)} total tandem repeats.")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()