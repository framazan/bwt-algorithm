#!/usr/bin/env python3
"""
Advanced BWT-based Tandem Repeat Finder for Genomics

Implements three-tier approach:
1. Short tandem repeats (1-9bp) with FM-index
2. Medium/long repeats (10-1000bp) with LCP arrays
3. Very long repeats (kb+) with long read evidence
"""

import numpy as np
from typing import List, Tuple, Dict, Iterator, Optional, Set, Union
import argparse
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
import time
import re
import math
from collections import Counter


def _natural_sort_key(value: str):
    """Return a tuple usable for natural sorting (e.g., chr2 before chr10)."""
    if value is None:
        return ()

    parts = re.split(r'(\d+)', str(value))
    key_parts: List[Tuple[int, object]] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key_parts.append((0, int(part)))
        else:
            key_parts.append((1, part.lower()))
    return tuple(key_parts)

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
    """

    # Base encoding for bit-masking (bcftools-inspired)
    BASE_TO_BITS = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}  # 2 bits per base
    BITS_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

    def __init__(self, text: str, sa_sample_rate: int = 32, occ_sample_rate: int = 128):
        """
        Initialize BWT with FM-index.

        Args:
            text: Input text (should end with a single '$' sentinel not present elsewhere)
            sa_sample_rate: Sample every nth suffix array position for space efficiency
            occ_sample_rate: Occurrence checkpoints every nth position to reduce memory
        """
        self.text: str = text
        self.n = len(text)
        self.sa_sample_rate = sa_sample_rate
        self.occ_sample_rate = occ_sample_rate


        self.text_arr = np.frombuffer(text.encode('utf-8'), dtype=np.uint8)

        # Build k-mer hash for fast lookups (performance optimization)
        self._build_kmer_hash()

        # Build suffix array and BWT (memory-efficient)
        self.suffix_array = self._build_suffix_array()
        self.bwt_arr = self._build_bwt_array()
        self.alphabet = sorted(set(text))
        self.char_to_code = {c: ord(c) for c in self.alphabet}
        self.code_to_char = {ord(c): c for c in self.alphabet}
        self.char_counts, self.char_totals = self._build_char_counts()
        self.char_counts_code = {ord(k): v for k, v in self.char_counts.items()}
        self.char_totals_code = {ord(k): v for k, v in self.char_totals.items()}
        self.occ_checkpoints = self._build_occurrence_checkpoints()
        self.sampled_sa = self._sample_suffix_array()

    def _build_kmer_hash(self, k: int = 8):
        """Build hash table for k-mer positions (bcftools-inspired optimization).

        Uses bit-masking for fast k-mer encoding (2 bits per base).
        """
        self.kmer_hash = {}  # hash -> list of positions
        if self.n < k:
            return

        # Encode first k-mer
        mask = (1 << (2 * k)) - 1  # k bases × 2 bits
        w = 0
        valid_count = 0

        for i in range(min(k, self.n)):
            base = self.text[i].upper()
            if base in self.BASE_TO_BITS:
                w = ((w << 2) | self.BASE_TO_BITS[base]) & mask
                valid_count += 1

        if valid_count == k:
            if w not in self.kmer_hash:
                self.kmer_hash[w] = []
            self.kmer_hash[w].append(0)

        # Rolling window
        for i in range(k, self.n):
            base = self.text[i].upper()
            if base in self.BASE_TO_BITS:
                w = ((w << 2) | self.BASE_TO_BITS[base]) & mask

                if w not in self.kmer_hash:
                    self.kmer_hash[w] = []
                self.kmer_hash[w].append(i - k + 1)

    def get_kmer_positions(self, kmer: str) -> List[int]:
        """Get positions of k-mer using hash table (O(1) lookup).

        Args:
            kmer: k-mer sequence (must be valid DNA bases)

        Returns:
            List of positions where k-mer occurs
        """
        if len(kmer) > 8 or not self.kmer_hash:
            # Fall back to FM-index for longer k-mers
            return self.locate_positions(kmer)

        # Encode k-mer to hash
        w = 0
        for base in kmer.upper():
            if base not in self.BASE_TO_BITS:
                return []
            w = (w << 2) | self.BASE_TO_BITS[base]

        return self.kmer_hash.get(w, [])

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
        except (ImportError, Exception):
            # Silently fall back to NumPy implementation
            pass

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
            # Use safe indexing: clip ipk to valid range, then apply condition
            ipk_safe = np.clip(ipk, 0, n - 1)
            key2 = np.where(ipk < n, rank[ipk_safe], -1)
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
    
    def rank(self, char: Union[str, int], pos: int) -> int:
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
    consensus_motif: Optional[str] = None  # Consensus motif from all copies
    mismatch_rate: float = 0.0  # Overall mismatch rate across all copies
    max_mismatches_per_copy: int = 0  # Maximum mismatches in any single copy
    n_copies_evaluated: int = 0  # Number of copies used in consensus
    strand: str = "+"  # Strand information
    # TRF-compatible fields
    percent_matches: float = 0.0  # Percent matches (100 - mismatch_rate*100)
    percent_indels: float = 0.0  # Percent indels (we use 0 for Hamming-based)
    score: int = 0  # Alignment score (calculated from matches/mismatches)
    composition: Optional[Dict[str, float]] = None  # A, C, G, T percentages
    entropy: float = 0.0  # Shannon entropy (0-2 bits)
    actual_sequence: Optional[str] = None  # The actual repeat sequence from genome
    variations: Optional[List[str]] = None  # Per-copy variation annotations

    def to_bed(self) -> str:
        """Convert to BED format."""
        cons = self.consensus_motif or self.motif
        return f"{self.chrom}\t{self.start}\t{self.end}\t{cons}\t{self.copies:.1f}\t{self.tier}\t{self.mismatch_rate:.3f}\t{self.strand}"

    def to_vcf_info(self) -> str:
        """Convert to VCF INFO field."""
        cons = self.consensus_motif or self.motif
        info_parts = [
            f"MOTIF={self.motif}",
            f"CONS_MOTIF={cons}",
            f"COPIES={self.copies:.1f}",
            f"TIER={self.tier}",
            f"CONF={self.confidence:.2f}",
            f"MM_RATE={self.mismatch_rate:.3f}",
            f"MAX_MM_PER_COPY={self.max_mismatches_per_copy}",
            f"N_COPIES_EVAL={self.n_copies_evaluated}",
            f"STRAND={self.strand}"
        ]
        return ";".join(info_parts)

    def to_trf_table(self) -> str:
        """Convert to TRF table format (tab-delimited).

        Format: Indices Period CopyNumber ConsensusSize PercentMatches PercentIndels Score A C G T Entropy
        """
        cons = self.consensus_motif or self.motif
        period = len(cons)
        consensus_size = len(cons)

        # Get composition
        comp = self.composition or {'A': 25.0, 'C': 25.0, 'G': 25.0, 'T': 25.0}

        indices = f"{self.start}--{self.end}"

        return (f"{indices}\t{period}\t{self.copies:.1f}\t{consensus_size}\t"
                f"{self.percent_matches:.0f}\t{self.percent_indels:.0f}\t{self.score}\t"
                f"{comp['A']:.0f}\t{comp['C']:.0f}\t{comp['G']:.0f}\t{comp['T']:.0f}\t"
                f"{self.entropy:.2f}")

    def to_trf_dat(self) -> str:
        """Convert to TRF DAT format (space-delimited, includes consensus and sequence).

        Format: Start End Period CopyNumber ConsensusSize PercentMatches PercentIndels Score
                A C G T Entropy ConsensusPattern Sequence
        """
        cons = self.consensus_motif or self.motif
        period = len(cons)
        consensus_size = len(cons)

        # Get composition
        comp = self.composition or {'A': 25.0, 'C': 25.0, 'G': 25.0, 'T': 25.0}

        # Get actual sequence (or use consensus repeated)
        sequence = self.actual_sequence or (cons * int(self.copies))

        return (f"{self.start} {self.end} {period} {self.copies:.1f} {consensus_size} "
                f"{self.percent_matches:.0f} {self.percent_indels:.0f} {self.score} "
                f"{comp['A']:.0f} {comp['C']:.0f} {comp['G']:.0f} {comp['T']:.0f} "
                f"{self.entropy:.2f} {cons} {sequence}")

    def to_strfinder(self, marker_name: Optional[str] = None,
                     flanking_left: str = "", flanking_right: str = "") -> str:
        """Convert to STRfinder-compatible CSV format (includes variation summary).

        Follows the STRfinder format specification:
        STR_marker, STR_position, STR_motif, STR_genotype_structure, STR_genotype,
        STR_core_seq, Allele_coverage, Alleles_ratio, Reads_Distribution, STR_depth, Full_seq, Variations
        """
        # Check if this is a compound repeat
        is_compound = hasattr(self, 'is_compound') and self.is_compound and hasattr(self, 'compound_partner')

        if is_compound:
            partner = self.compound_partner
            cons1 = self.consensus_motif or self.motif
            cons2 = partner.consensus_motif or partner.motif

            # Compound repeat formatting
            marker = marker_name or f"STR_{self.chrom}_{self.start}"
            position = f"{self.chrom}:{self.start + 1}-{partner.end}"
            str_motif = f"[{cons1}]n+[{cons2}]n"

            copies1 = int(round(self.copies))
            copies2 = int(round(partner.copies))
            motif_len1 = len(cons1)
            motif_len2 = len(cons2)

            genotype_struct = f"{motif_len1}[{cons1}]{copies1};{motif_len2}[{cons2}]{copies2},0"
            genotype = f"{copies1}/{copies2}"

            core_seq1 = self.actual_sequence or (cons1 * copies1)
            core_seq2 = partner.actual_sequence or (cons2 * copies2)
            core_seq = core_seq1 + core_seq2

            allele_coverage = "100%"
            alleles_ratio = "-"
            reads_dist = f"{copies1}:{copies2}"
            str_depth = str(copies1 + copies2)

            if flanking_left or flanking_right:
                full_seq = flanking_left + core_seq + flanking_right
            else:
                full_seq = core_seq

            variation_str = "-"

            return (f"{marker}\t{position}\t{str_motif}\t{genotype_struct}\t{genotype}\t"
                    f"{core_seq}\t{allele_coverage}\t{alleles_ratio}\t{reads_dist}\t"
                    f"{str_depth}\t{full_seq}\t{variation_str}")

        # Regular (non-compound) repeat handling
        cons = self.consensus_motif or self.motif

        # STR_marker - use provided name or generate from position
        marker = marker_name or f"STR_{self.chrom}_{self.start}"

        # STR_position - chr:start-end format (1-BASED COORDINATES)
        # Convert from 0-based internal to 1-based output
        position = f"{self.chrom}:{self.start + 1}-{self.end}"

        # STR_motif - [MOTIF]n format
        str_motif = f"[{cons}]n"

        # STR_genotype_structure - format as motif_length[MOTIF]copies,truncated
        # Calculate truncated bases (remainder after complete copies)
        motif_len = len(cons)
        total_length = self.end - self.start

        complete_copies = int(math.floor(self.copies + 1e-6))
        complete_length = motif_len * complete_copies
        truncated = total_length - complete_length
        genotype_struct = f"{motif_len}[{cons}]{complete_copies},{truncated}"

        # STR_genotype - repeat number(s)
        if abs(self.copies - round(self.copies)) < 1e-6:
            genotype = str(int(round(self.copies)))
        else:
            genotype = f"{self.copies:.2f}".rstrip('0').rstrip('.')

        # STR_core_seq - the actual core sequence
        # Use actual sequence if available, otherwise reconstruct
        if self.actual_sequence:
            core_seq_full = self.actual_sequence
        else:
            core_seq_full = cons * int(self.copies)
        # Truncate long sequences with ellipsis notation
        if len(core_seq_full) > 150:
            # Show first ~70 chars + " ... (xN)" where N is the number of copies
            truncate_len = 70
            core_seq = f"{core_seq_full[:truncate_len]}... (x{complete_copies})"
        else:
            core_seq = core_seq_full

        # Allele_coverage - percentage (use percent_matches if available, else confidence)
        if hasattr(self, 'percent_matches') and self.percent_matches is not None:
            allele_coverage = f"{self.percent_matches:.0f}%"
        else:
            allele_coverage = f"{self.confidence * 100:.0f}%"

        # Alleles_ratio - for diploid; use "-" for haploid
        alleles_ratio = "-"

        # Reads_Distribution - simplified format showing copy numbers
        # Format: 7:0,8:150,9:0,10:0,11:200,12:0 (copy_number:read_count)
        reads_dist = f"{complete_copies}:{self.n_copies_evaluated}"

        # STR_depth - use n_copies_evaluated as proxy
        str_depth = str(self.n_copies_evaluated)

        # Variation summary (list variants differing from consensus)
        variation_str = ";".join(self.variations) if self.variations else "-"

        # Full_seq - flanking + CORE + flanking (simple concatenation)
        # Use full sequence (not truncated) for Full_seq, but truncate if too long
        if flanking_left or flanking_right:
            full_seq_complete = flanking_left + core_seq_full + flanking_right
        else:
            full_seq_complete = core_seq_full

        # Truncate Full_seq if extremely long (keep reasonable size)
        if len(full_seq_complete) > 500:
            full_seq = f"{full_seq_complete[:250]}...{full_seq_complete[-200:]}"
        else:
            full_seq = full_seq_complete

        return (f"{marker}\t{position}\t{str_motif}\t{genotype_struct}\t{genotype}\t"
                f"{core_seq}\t{allele_coverage}\t{alleles_ratio}\t{reads_dist}\t"
                f"{str_depth}\t{full_seq}\t{variation_str}")


@dataclass
class AlignmentResult:
    """Per-copy alignment outcome against the consensus motif template."""
    consumed: int
    unit_sequence: str
    mismatch_count: int
    insertion_length: int
    deletion_length: int
    operations: List[Tuple]  # ('sub', pos, ref, alt) | ('ins', pos, seq) | ('del', pos, length)
    observed_bases: List[Tuple[int, str]]  # (motif_index, base) observations for consensus tally
    edit_distance: int

    @property
    def error_count(self) -> int:
        return self.mismatch_count + self.insertion_length + self.deletion_length


@dataclass
class RepeatAlignmentSummary:
    """Aggregate result for aligning a tandem repeat block."""
    consensus: str
    motif_len: int
    copies: int
    consumed_length: int
    mismatch_rate: float
    max_errors_per_copy: int
    variations: List[str]
    copy_sequences: List[str]
    total_insertions: int
    total_deletions: int
    error_counts: List[int]
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
    def reverse_complement(seq: str) -> str:
        """Get reverse complement of DNA sequence."""
        complement_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        return ''.join(complement_map.get(b, b) for b in reversed(seq))

    @staticmethod
    def get_canonical_motif_stranded(motif: str) -> Tuple[str, str]:
        """Get canonical motif considering both strands.

        Returns:
            (canonical_motif, strand) where strand is '+' or '-'
        """
        if not motif:
            return motif, '+'

        # Get all rotations of forward strand
        forward_rotations = [motif[i:] + motif[:i] for i in range(len(motif))]
        forward_canonical = min(forward_rotations)

        # Get all rotations of reverse complement
        rc = MotifUtils.reverse_complement(motif)
        rc_rotations = [rc[i:] + rc[:i] for i in range(len(rc))]
        rc_canonical = min(rc_rotations)

        # Return lexicographically smallest
        if forward_canonical <= rc_canonical:
            return forward_canonical, '+'
        else:
            return rc_canonical, '-'

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
    def calculate_entropy(seq: str) -> float:
        """Calculate Shannon entropy of sequence (bits per base)."""
        if not seq:
            return 0.0

        from collections import Counter
        counts = Counter(seq)
        n = len(seq)
        entropy = 0.0

        for count in counts.values():
            if count > 0:
                p = count / n
                entropy -= p * np.log2(p)

        return entropy

    @staticmethod
    def is_transition(base1: str, base2: str) -> bool:
        """Check if a base change is a transition (A↔G or C↔T).

        Transitions are more common than transversions in biology.
        Purines: A, G (transitions within purines: A↔G)
        Pyrimidines: C, T (transitions within pyrimidines: C↔T)
        """
        if base1 == base2:
            return True  # No change

        transitions = {
            ('A', 'G'), ('G', 'A'),  # Purine transitions
            ('C', 'T'), ('T', 'C')   # Pyrimidine transitions
        }
        return (base1, base2) in transitions

    @staticmethod
    def hamming_distance(s1: str, s2: str) -> int:
        """Calculate Hamming distance between two strings of equal length."""
        if len(s1) != len(s2):
            return max(len(s1), len(s2))

        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    @staticmethod
    def hamming_distance_array(arr1: np.ndarray, arr2: np.ndarray) -> int:
        """Calculate Hamming distance between two uint8 arrays."""
        if arr1.size != arr2.size:
            return max(arr1.size, arr2.size)

        return int(np.count_nonzero(arr1 != arr2))

    @staticmethod
    def count_transversions_array(arr1: np.ndarray, arr2: np.ndarray) -> int:
        """Count transversions (non-transition mismatches) between two uint8 arrays.

        Returns number of transversion changes (A↔C, A↔T, G↔C, G↔T).
        """
        if arr1.size != arr2.size:
            return max(arr1.size, arr2.size)

        # ASCII codes: A=65, C=67, G=71, T=84
        transversions = 0
        for i in range(arr1.size):
            b1, b2 = arr1[i], arr2[i]
            if b1 != b2:
                # Convert to characters for transition check
                c1 = chr(b1) if 65 <= b1 <= 84 else 'N'
                c2 = chr(b2) if 65 <= b2 <= 84 else 'N'
                if not MotifUtils.is_transition(c1, c2):
                    transversions += 1

        return transversions

    @staticmethod
    def edit_distance(a: str, b: str) -> int:
        """Compute Levenshtein edit distance between two short strings."""
        la, lb = len(a), len(b)
        if la == 0:
            return lb
        if lb == 0:
            return la

        prev = list(range(lb + 1))
        curr = [0] * (lb + 1)

        for i in range(1, la + 1):
            curr[0] = i
            ai = a[i - 1]
            for j in range(1, lb + 1):
                cost = 0 if ai == b[j - 1] else 1
                curr[j] = min(
                    prev[j] + 1,      # deletion
                    curr[j - 1] + 1,  # insertion
                    prev[j - 1] + cost  # substitution
                )
            prev, curr = curr, prev

        return prev[lb]

    @staticmethod
    def _align_unit_to_window(motif: str, window: str, max_indel: int,
                              mismatch_tolerance: int) -> Optional[AlignmentResult]:
        """Align motif to a window allowing mismatches and small indels."""
        m = len(motif)
        n = len(window)

        if m == 0 or n == 0:
            return None

        max_indel = max(0, max_indel)
        mismatch_tolerance = max(0, mismatch_tolerance)

        lower = max(0, m - max_indel)
        upper = min(n, m + max_indel)
        if lower > upper:
            return None

        inf = m + n + 10
        dp = [[inf] * (n + 1) for _ in range(m + 1)]
        ptr = [[''] * (n + 1) for _ in range(m + 1)]

        dp[0][0] = 0
        for j in range(1, n + 1):
            dp[0][j] = j
            ptr[0][j] = 'I'
        for i in range(1, m + 1):
            dp[i][0] = i
            ptr[i][0] = 'D'

        band_extra = max_indel + 2

        for i in range(1, m + 1):
            j_min = max(1, i - band_extra)
            j_max = min(n, i + band_extra)
            for j in range(j_min, j_max + 1):
                sub_cost = dp[i - 1][j - 1] + (motif[i - 1] != window[j - 1])
                del_cost = dp[i - 1][j] + 1
                ins_cost = dp[i][j - 1] + 1

                best_cost = sub_cost
                best_ptr = 'M' if motif[i - 1] == window[j - 1] else 'S'

                if del_cost < best_cost:
                    best_cost = del_cost
                    best_ptr = 'D'
                if ins_cost < best_cost:
                    best_cost = ins_cost
                    best_ptr = 'I'

                dp[i][j] = best_cost
                ptr[i][j] = best_ptr

        best_j = -1
        best_cost = inf
        for j in range(lower, upper + 1):
            cost = dp[m][j]
            if cost < best_cost:
                best_cost = cost
                best_j = j

        if best_j <= 0 or best_cost >= inf:
            return None

        aligned_ref = []
        aligned_query = []
        i, j = m, best_j
        while i > 0 or j > 0:
            op = ptr[i][j]
            if op in ('M', 'S'):
                aligned_ref.append(motif[i - 1])
                aligned_query.append(window[j - 1])
                i -= 1
                j -= 1
            elif op == 'D':
                aligned_ref.append(motif[i - 1])
                aligned_query.append('-')
                i -= 1
            elif op == 'I':
                aligned_ref.append('-')
                aligned_query.append(window[j - 1])
                j -= 1
            else:  # Should only occur at origin
                break

        aligned_ref.reverse()
        aligned_query.reverse()

        operations: List[Tuple] = []
        observed_bases: List[Tuple[int, str]] = []
        mismatch_count = 0
        insertion_len = 0
        deletion_len = 0

        ref_pos = 0
        pending_ins: List[str] = []
        pending_ins_pos = 0
        pending_del_len = 0
        pending_del_pos = 0

        for r, q in zip(aligned_ref, aligned_query):
            if r == '-':
                if not pending_ins:
                    pending_ins_pos = ref_pos
                pending_ins.append(q)
                continue

            if pending_ins:
                ins_seq = ''.join(pending_ins)
                operations.append(('ins', pending_ins_pos, ins_seq))
                insertion_len += len(ins_seq)
                pending_ins = []
                pending_ins_pos = 0

            ref_pos += 1

            if q == '-':
                if pending_del_len == 0:
                    pending_del_pos = ref_pos
                pending_del_len += 1
                continue

            if pending_del_len:
                operations.append(('del', pending_del_pos, pending_del_len))
                deletion_len += pending_del_len
                pending_del_len = 0

            observed_bases.append((ref_pos - 1, q))
            if r != q:
                operations.append(('sub', ref_pos, r, q))
                mismatch_count += 1

        if pending_ins:
            ins_seq = ''.join(pending_ins)
            operations.append(('ins', pending_ins_pos, ins_seq))
            insertion_len += len(ins_seq)

        if pending_del_len:
            operations.append(('del', pending_del_pos, pending_del_len))
            deletion_len += pending_del_len

        if mismatch_count > mismatch_tolerance:
            return None
        if insertion_len > max_indel or deletion_len > max_indel:
            return None

        return AlignmentResult(
            consumed=best_j,
            unit_sequence=window[:best_j],
            mismatch_count=mismatch_count,
            insertion_length=insertion_len,
            deletion_length=deletion_len,
            operations=operations,
            observed_bases=observed_bases,
            edit_distance=best_cost
        )

    @staticmethod
    def _consensus_from_counts(counts: List[Counter], fallback: str) -> str:
        """Build consensus string from per-position base counts."""
        consensus = []
        for idx, counter in enumerate(counts):
            if counter:
                base, _ = counter.most_common(1)[0]
                consensus.append(base)
            else:
                consensus.append(fallback[idx] if idx < len(fallback) else 'N')
        return ''.join(consensus)

    @staticmethod
    def align_repeat_region(sequence: str, start: int, end: int, motif_template: str,
                            mismatch_fraction: float = 0.1,
                            max_indel: Optional[int] = None,
                            min_copies: int = 3) -> Optional[RepeatAlignmentSummary]:
        """Align sequential copies of a motif within a sequence region."""
        if not motif_template:
            return None

        seq_len = len(sequence)
        if seq_len == 0:
            return None

        start = max(0, start)
        end = min(seq_len, end if end > start else seq_len)

        motif_len = len(motif_template)
        if motif_len == 0:
            return None

        tolerance = max(1, int(math.floor(motif_len * mismatch_fraction)))
        if max_indel is None:
            max_indel = max(1, min(10, motif_len // 2 if motif_len >= 4 else 1))
        else:
            max_indel = max(0, max_indel)

        position_counts: List[Counter] = [Counter() for _ in range(motif_len)]
        copy_sequences: List[str] = []
        operations_by_copy: List[List[Tuple]] = []
        error_counts: List[int] = []

        total_insertions = 0
        total_deletions = 0

        current_motif = motif_template
        pos = start
        safety_limit = min(seq_len, max(end, start + motif_len * min_copies) + max(motif_len * 3, max_indel * 4))

        while pos < safety_limit:
            window_end = min(seq_len, pos + motif_len + max_indel)
            window = sequence[pos:window_end]
            if len(window) < motif_len - max_indel:
                break

            result = MotifUtils._align_unit_to_window(current_motif, window, max_indel, tolerance)
            if result is None or result.consumed == 0:
                break

            copy_sequences.append(result.unit_sequence)
            operations_by_copy.append(result.operations)
            error_counts.append(result.error_count)
            total_insertions += result.insertion_length
            total_deletions += result.deletion_length

            for motif_idx, base in result.observed_bases:
                if 0 <= motif_idx < motif_len:
                    position_counts[motif_idx][base] += 1

            pos += result.consumed
            current_motif = MotifUtils._consensus_from_counts(position_counts, current_motif)

        copies = len(copy_sequences)
        if copies < min_copies:
            return None

        consumed_len = pos - start
        if consumed_len <= 0:
            return None

        consensus = MotifUtils._consensus_from_counts(position_counts, current_motif)
        total_errors = sum(error_counts)
        denom = copies * motif_len
        mismatch_rate = total_errors / denom if denom > 0 else 0.0
        max_errors_per_copy = max(error_counts) if error_counts else 0

        variations: List[str] = []
        for idx, ops in enumerate(operations_by_copy, 1):
            for op in ops:
                if not op:
                    continue
                kind = op[0]
                if kind == 'sub':
                    _, pos_idx, ref_base, alt_base = op
                    variations.append(f"{idx}:{pos_idx}:{ref_base}>{alt_base}")
                elif kind == 'ins':
                    _, pos_idx, inserted = op
                    if inserted:
                        variations.append(f"{idx}:{pos_idx}:ins({inserted})")
                elif kind == 'del':
                    _, pos_idx, length = op
                    if length > 0:
                        variations.append(f"{idx}:{pos_idx}:del({length})")

        return RepeatAlignmentSummary(
            consensus=consensus,
            motif_len=motif_len,
            copies=copies,
            consumed_length=consumed_len,
            mismatch_rate=mismatch_rate,
            max_errors_per_copy=max_errors_per_copy,
            variations=variations,
            copy_sequences=copy_sequences,
            total_insertions=total_insertions,
            total_deletions=total_deletions,
            error_counts=error_counts
        )

    @staticmethod
    def is_insertion_variant(candidate: str, consensus: str) -> bool:
        """Return True if removing a single base from candidate yields consensus."""
        if len(candidate) != len(consensus) + 1:
            return False
        for i in range(len(candidate)):
            if candidate[:i] + candidate[i + 1:] == consensus:
                return True
        return False

    @staticmethod
    def is_deletion_variant(candidate: str, consensus: str) -> bool:
        """Return True if inserting a single base into candidate yields consensus."""
        if len(candidate) + 1 != len(consensus):
            return False
        for i in range(len(consensus)):
            if consensus[:i] + consensus[i + 1:] == candidate:
                return True
        return False

    @staticmethod
    def smallest_period_str(s: str) -> int:
        """Return length of the smallest period of string s."""
        if not s:
            return 0
        n = len(s)
        for p in range(1, n + 1):
            if n % p == 0 and s == s[:p] * (n // p):
                return p
        return n

    @staticmethod
    def normalize_variant(candidate: str, consensus: str) -> str:
        """Rotate variant to best align with consensus (minimal edit distance)."""
        if not candidate:
            return candidate

        best = candidate
        best_cost = MotifUtils.edit_distance(candidate, consensus)

        if len(candidate) == 1:
            return candidate

        for shift in range(1, len(candidate)):
            rotated = candidate[shift:] + candidate[:shift]
            cost = MotifUtils.edit_distance(rotated, consensus)
            if cost < best_cost or (cost == best_cost and rotated < best):
                best = rotated
                best_cost = cost

        return best

    @staticmethod
    def rotate_deletion_variant(candidate: str, consensus: str) -> str:
        """Rotate shorter variant so it no longer begins with the consensus prefix."""
        if not candidate:
            return candidate

        rotated = candidate
        for _ in range(len(candidate)):
            if rotated[0] != consensus[0]:
                return rotated
            rotated = rotated[1:] + rotated[:1]
        return rotated

    @staticmethod
    def build_consensus_motif(sequences: List[str]) -> Tuple[str, float]:
        """Build consensus motif from multiple aligned sequences using majority vote.

        Returns:
            (consensus, avg_mismatch_rate) - consensus sequence and average mismatch rate
        """
        if not sequences:
            return "", 0.0

        if len(sequences) == 1:
            return sequences[0], 0.0

        motif_len = len(sequences[0])
        consensus = []
        total_mismatches = 0

        for pos in range(motif_len):
            bases = [seq[pos] for seq in sequences if pos < len(seq)]
            if not bases:
                consensus.append('N')
                continue

            # Majority vote
            from collections import Counter
            counts = Counter(bases)
            most_common = counts.most_common(1)[0][0]
            consensus.append(most_common)

            # Count mismatches at this position
            mismatches = len(bases) - counts[most_common]
            total_mismatches += mismatches

        total_bases = len(sequences) * motif_len
        avg_mismatch_rate = total_mismatches / total_bases if total_bases > 0 else 0.0

        return ''.join(consensus), avg_mismatch_rate

    @staticmethod
    def build_consensus_motif_array(text_arr: np.ndarray, start: int, motif_len: int,
                                   n_copies: int) -> Tuple[np.ndarray, float, int]:
        """Build consensus motif from array copies using majority vote.

        Returns:
            (consensus_array, mismatch_rate, max_mismatches_per_copy)
        """
        if n_copies == 0 or motif_len == 0:
            return np.array([], dtype=np.uint8), 0.0, 0

        consensus = np.zeros(motif_len, dtype=np.uint8)
        total_mismatches = 0
        max_mismatches_per_copy = 0

        # Collect all copies
        copies = []
        for i in range(n_copies):
            copy_start = start + i * motif_len
            copy_end = copy_start + motif_len
            if copy_end > text_arr.size:
                break
            copy_arr = text_arr[copy_start:copy_end]
            copies.append(copy_arr)

        if not copies:
            return np.array([], dtype=np.uint8), 0.0, 0

        # Build consensus by majority vote at each position
        for pos in range(motif_len):
            bases = [copy[pos] for copy in copies if pos < len(copy)]
            if not bases:
                consensus[pos] = ord('N')
                continue

            # Find most common base
            unique, counts = np.unique(bases, return_counts=True)
            most_common_idx = np.argmax(counts)
            consensus[pos] = unique[most_common_idx]

        # Calculate mismatch statistics
        for copy in copies:
            mismatches = MotifUtils.hamming_distance_array(copy, consensus)
            total_mismatches += mismatches
            max_mismatches_per_copy = max(max_mismatches_per_copy, mismatches)

        total_bases = len(copies) * motif_len
        mismatch_rate = total_mismatches / total_bases if total_bases > 0 else 0.0

        return consensus, mismatch_rate, max_mismatches_per_copy

    @staticmethod
    def summarize_variations_array(text_arr: np.ndarray, start: int, end: int, motif_len: int,
                                   consensus_arr: np.ndarray) -> List[str]:
        """Summarize per-copy variations relative to consensus, allowing small indels."""
        if text_arr.size == 0 or motif_len <= 0:
            return []

        sequence = text_arr.tobytes().decode('ascii', errors='replace')
        start = max(0, start)
        end = min(len(sequence), end if end > start else len(sequence))

        if end <= start:
            return []

        if consensus_arr.size > 0:
            motif_template = consensus_arr.tobytes().decode('ascii', errors='replace')
        else:
            motif_template = sequence[start:start + motif_len]

        summary = MotifUtils.align_repeat_region(
            sequence,
            start,
            end,
            motif_template,
            mismatch_fraction=0.1,
            min_copies=1
        )
        if not summary:
            return []
        return summary.variations

    @staticmethod
    def calculate_composition(sequence: str) -> Dict[str, float]:
        """Calculate nucleotide composition as percentages.

        Returns:
            Dictionary with A, C, G, T percentages
        """
        if not sequence:
            return {'A': 0.0, 'C': 0.0, 'G': 0.0, 'T': 0.0}

        from collections import Counter
        counts = Counter(sequence.upper())
        total = len(sequence)

        composition = {
            'A': (counts.get('A', 0) / total) * 100.0,
            'C': (counts.get('C', 0) / total) * 100.0,
            'G': (counts.get('G', 0) / total) * 100.0,
            'T': (counts.get('T', 0) / total) * 100.0,
        }

        return composition

    @staticmethod
    def calculate_trf_score(consensus: str, copies: int, mismatch_rate: float, length: int) -> int:
        """Calculate TRF-style alignment score.

        TRF uses match/mismatch/indel scoring. We approximate:
        - Match: +2 points
        - Mismatch: -7 points
        - Indel: -7 points (we use 0 indels for Hamming distance)

        Returns:
            Alignment score (integer)
        """
        total_bases = length
        matches = total_bases * (1.0 - mismatch_rate)
        mismatches = total_bases * mismatch_rate

        # TRF scoring parameters (approximately)
        match_score = 2
        mismatch_penalty = 7

        score = int((matches * match_score) - (mismatches * mismatch_penalty))
        return max(0, score)  # Don't allow negative scores

    @staticmethod
    def calculate_trf_statistics(text_arr: np.ndarray, start: int, end: int,
                                 consensus_motif: str, copies: int,
                                 mismatch_rate: float) -> Tuple[float, float, int, Dict[str, float], float, str]:
        """Calculate TRF-compatible statistics for a repeat.

        Returns:
            (percent_matches, percent_indels, score, composition, entropy, actual_sequence)
        """
        # Extract actual sequence
        if end <= text_arr.size:
            actual_sequence = text_arr[start:end].tobytes().decode('ascii', errors='replace')
        else:
            actual_sequence = consensus_motif * int(copies)

        # Percent matches (inverse of mismatch rate)
        percent_matches = (1.0 - mismatch_rate) * 100.0

        # Percent indels (we use Hamming distance, so 0 indels)
        percent_indels = 0.0

        # Composition
        composition = MotifUtils.calculate_composition(consensus_motif)

        # Entropy (already calculated, but recalculate for consistency)
        entropy = MotifUtils.calculate_entropy(consensus_motif)

        # Score
        length = end - start
        score = MotifUtils.calculate_trf_score(consensus_motif, copies, mismatch_rate, length)

        return percent_matches, percent_indels, score, composition, entropy, actual_sequence

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
    """Tier 1: Short Perfect Tandem Repeat Finder using optimized sliding window (1-9bp).

    Fast method for detecting perfect tandem repeats with adaptive sampling.
    No BWT/FM-index required - just optimized direct scanning with smart skipping.
    """

    def __init__(self, text_arr: np.ndarray, max_motif_length: int = 9, show_progress: bool = False):
        self.text_arr = text_arr
        self.max_motif_length = max_motif_length
        self.min_copies = 3  # Require at least 3 copies to reduce noise
        self.min_array_length = 6  # Minimum total array length in bp
        self.min_entropy = 1.0  # Minimum Shannon entropy to avoid low-complexity
        self.show_progress = show_progress

    def _get_max_mismatches_for_array(self, motif_len: int, n_copies: int) -> int:
        """Calculate maximum allowed mismatches for full array.

        Args:
            motif_len: Length of the motif/period
            n_copies: Number of tandem copies

        Returns:
            Maximum allowed mismatches across entire repeat array
        """
        total_length = motif_len * n_copies

        # For single nucleotide repeats (homopolymers), NO mismatches allowed
        if motif_len == 1:
            return 0

        # For short motifs (2-6bp), be very conservative
        if motif_len <= 6:
            # Allow at most 5% mismatches (1 in 20 bases)
            return max(1, int(np.ceil(0.05 * total_length)))

        # For longer motifs, allow up to 8% mismatches
        return max(1, int(np.ceil(0.08 * total_length)))
    
    def _find_simple_tandems_kmer(self, chromosome: str) -> List[TandemRepeat]:
        """Find simple perfect tandem repeats using optimized sliding window.

        This uses a FAST sliding window approach with early skipping.
        No k-mer index needed - direct scanning is faster for tandem repeat detection.
        """
        repeats = []
        text_arr = self.text_arr
        n = text_arr.size
        seen_regions: Set[Tuple[int, int]] = set()

        # Create bitmap for fast region checking (O(1) instead of O(n))
        seen_mask = np.zeros(n, dtype=bool)

        # For very large chromosomes, use adaptive sampling
        if n > 10_000_000:  # > 10 Mbp
            position_step = 50  # Skip positions for speed
            if self.show_progress:
                print(f"  [{chromosome}] Large sequence ({n:,} bp) - using fast sampling mode (step={position_step})")
        elif n > 5_000_000:  # > 5 Mbp
            position_step = 20
        else:
            position_step = 1

        # Process each motif length (1-9bp) in REVERSE order (longest first)
        # This ensures we detect [AT]n before [A]n, [GCG]n before [GC]n, etc.
        for motif_len in range(min(self.max_motif_length, 9), 0, -1):
            i = 0
            while i < n - motif_len:
                # Skip if already in a found region (O(1) bitmap check)
                if seen_mask[i]:
                    i += position_step
                    continue

                # Get potential motif
                motif_arr = text_arr[i:i + motif_len]
                motif = motif_arr.tobytes().decode('ascii', errors='replace')

                # Skip if contains N or other non-ACGT
                if not all(c in 'ACGT' for c in motif):
                    i += position_step
                    continue

                # Count consecutive perfect repeats
                copies = 1
                check_pos = i + motif_len

                while check_pos + motif_len <= n:
                    next_motif_arr = text_arr[check_pos:check_pos + motif_len]
                    if np.array_equal(motif_arr, next_motif_arr):
                        copies += 1
                        check_pos += motif_len
                    else:
                        break

                # If enough copies found, create a repeat
                if copies >= self.min_copies:
                    end_pos = i + copies * motif_len
                    length = end_pos - i

                    # Check entropy
                    entropy = MotifUtils.calculate_entropy(motif)
                    if entropy < self.min_entropy and length < 10:
                        i += position_step
                        continue

                    if length >= self.min_array_length:
                        # Calculate statistics
                        actual_sequence = text_arr[i:end_pos].tobytes().decode('ascii', errors='replace')
                        (percent_matches, percent_indels, score, composition,
                         entropy_val, _) = MotifUtils.calculate_trf_statistics(
                            text_arr, i, end_pos, motif, copies, 0.0
                        )

                        repeat = TandemRepeat(
                            chrom=chromosome,
                            start=i,
                            end=end_pos,
                            motif=motif,
                            copies=float(copies),
                            length=length,
                            tier=1,
                            confidence=1.0,
                            consensus_motif=motif,
                            mismatch_rate=0.0,
                            max_mismatches_per_copy=0,
                            n_copies_evaluated=copies,
                            strand='+',
                            percent_matches=percent_matches,
                            percent_indels=percent_indels,
                            score=score,
                            composition=composition,
                            entropy=entropy_val,
                            actual_sequence=actual_sequence,
                            variations=None
                        )
                        repeats.append(repeat)
                        seen_regions.add((i, end_pos))
                        # Update bitmap
                        seen_mask[i:end_pos] = True
                        i = end_pos  # Jump past the repeat
                        continue

                i += position_step

        return repeats

    def find_strs(self, chromosome: str) -> List[TandemRepeat]:
        """Find perfect short tandem repeats (1-9bp) using optimized sliding window.

        This is Tier 1: fast sliding window with adaptive sampling, perfect match only.
        """
        return self._find_simple_tandems_kmer(chromosome)

    def _find_tandems_with_mismatches(self, positions: List[int], motif: str,
                                     chromosome: str, motif_len: int,
                                     seen_regions: Set[Tuple[int, int]]) -> List[TandemRepeat]:
        """Find tandem repeats allowing mismatches using seed-and-extend strategy."""
        repeats = []
        if not positions:
            return repeats

        positions.sort()
        max_mm = 0  # Parameter not used by _extend_tandem_array (kept for backward compatibility)
        text_arr = self.bwt.text_arr

        for seed_pos in positions:
            # Skip if this position is already part of a found repeat
            if any(start <= seed_pos < end for start, end in seen_regions):
                continue

            # Skip seeds whose window would exceed the chromosome boundary
            if seed_pos + motif_len > text_arr.size:
                continue

            # Explore all rotations that could anchor this seed by sliding up to one motif length
            best_start = None
            best_end = None
            best_copies = 0

            max_left_shift = min(motif_len, seed_pos + 1)
            for shift in range(max_left_shift):
                candidate_seed = seed_pos - shift
                candidate_end = candidate_seed + motif_len

                if candidate_seed < 0 or candidate_end > text_arr.size:
                    continue

                # Skip if candidate seed already claimed by another repeat
                if any(start <= candidate_seed < end for start, end in seen_regions):
                    continue

                candidate_arr = text_arr[candidate_seed:candidate_end]
                candidate_motif = candidate_arr.tobytes().decode('ascii', errors='replace')

                start_pos_candidate, end_pos_candidate, copies_candidate = self._extend_tandem_array(
                    text_arr, candidate_seed, candidate_motif, motif_len, max_mm
                )

                # Ensure the original seed is covered by this candidate extension
                if not (start_pos_candidate <= seed_pos < end_pos_candidate):
                    continue

                if (copies_candidate > best_copies or
                        (copies_candidate == best_copies and
                         (best_start is None or start_pos_candidate < best_start))):
                    best_start = start_pos_candidate
                    best_end = end_pos_candidate
                    best_copies = copies_candidate

            if best_start is None:
                continue

            start_pos, end_pos, copies = best_start, best_end, best_copies

            array_length = end_pos - start_pos

            if copies >= self.min_copies and (end_pos - start_pos) >= self.min_array_length:
                # Build consensus motif from all copies
                consensus_arr, mm_rate, max_mm_per_copy = MotifUtils.build_consensus_motif_array(
                    text_arr, start_pos, motif_len, copies
                )

                if consensus_arr.size == 0:
                    continue

                consensus_str = consensus_arr.tobytes().decode('ascii', errors='replace')

                primitive_len = MotifUtils.smallest_period_str(consensus_str)
                if primitive_len < len(consensus_str):
                    motif_len = primitive_len
                    copies = max(1, (end_pos - start_pos) // motif_len)
                    end_pos = start_pos + copies * motif_len
                    consensus_arr, mm_rate, max_mm_per_copy = MotifUtils.build_consensus_motif_array(
                        text_arr, start_pos, motif_len, copies
                    )
                    if consensus_arr.size == 0:
                        continue
                    consensus_str = consensus_arr.tobytes().decode('ascii', errors='replace')

                # Get canonical motif considering both strands
                canonical, strand = MotifUtils.get_canonical_motif_stranded(consensus_str)

                # Check maximality
                if self._is_maximal_repeat_approx(start_pos, end_pos, consensus_arr, motif_len, max_mm):
                    # Calculate confidence based on mismatch rate
                    confidence = max(0.5, 1.0 - mm_rate)

                    # Calculate TRF-compatible statistics
                    (percent_matches, percent_indels, score, composition,
                     entropy, actual_sequence) = MotifUtils.calculate_trf_statistics(
                        text_arr, start_pos, end_pos, consensus_str, copies, mm_rate
                    )

                    variations = MotifUtils.summarize_variations_array(
                        text_arr, start_pos, end_pos, motif_len, consensus_arr
                    )

                    repeat = TandemRepeat(
                        chrom=chromosome,
                        start=start_pos,
                        end=end_pos,
                        motif=consensus_str,  # Use observed consensus, not search motif
                        copies=copies,
                        length=end_pos - start_pos,
                        tier=1,
                        confidence=confidence,
                        consensus_motif=consensus_str,  # Store observed consensus
                        mismatch_rate=mm_rate,
                        max_mismatches_per_copy=max_mm_per_copy,
                        n_copies_evaluated=copies,
                        strand=strand,
                        percent_matches=percent_matches,
                        percent_indels=percent_indels,
                        score=score,
                        composition=composition,
                        entropy=entropy,
                        actual_sequence=actual_sequence,
                        variations=variations if variations else None
                    )
                    repeats.append(repeat)
                    seen_regions.add((start_pos, end_pos))

        return repeats

    def _extend_tandem_array(self, text_arr: np.ndarray, seed_pos: int,
                            motif: str, motif_len: int, max_mismatches: int) -> Tuple[int, int, int]:
        """Extend tandem array left and right from seed position allowing mismatches.

        Note: max_mismatches parameter is no longer used; kept for backward compatibility.
        Mismatch tolerance is now calculated as 10% of full array length.

        Returns:
            (start_pos, end_pos, copy_count)
        """
        motif_arr = np.frombuffer(motif.encode('ascii'), dtype=np.uint8)
        n = text_arr.size

        # Start from seed position
        start = seed_pos
        end = seed_pos + motif_len
        copies = 1

        # Build initial consensus from first copy
        consensus = motif_arr.copy()

        # Helper function to calculate total mismatches across all copies
        def get_total_mismatches(start_pos, end_pos, consensus_arr, motif_length):
            num_copies = (end_pos - start_pos) // motif_length
            total_mm = 0
            for i in range(num_copies):
                copy_start = start_pos + i * motif_length
                copy_end = copy_start + motif_length
                if copy_end <= n:
                    copy = text_arr[copy_start:copy_end]
                    total_mm += MotifUtils.hamming_distance_array(copy, consensus_arr)
            return total_mm

        # Extend right
        while end + motif_len <= n:
            next_copy = text_arr[end:end + motif_len]

            # Tentatively add this copy
            temp_copies = copies + 1
            temp_end = end + motif_len

            # Collect all copies including the new one
            all_copies = []
            for i in range(temp_copies):
                copy_start = start + i * motif_len
                copy_end = copy_start + motif_len
                if copy_end <= n:
                    all_copies.append(text_arr[copy_start:copy_end])

            # Build temporary consensus
            temp_consensus = np.zeros(motif_len, dtype=np.uint8)
            for pos in range(motif_len):
                bases = [copy[pos] for copy in all_copies if pos < len(copy)]
                if bases:
                    unique, counts = np.unique(bases, return_counts=True)
                    temp_consensus[pos] = unique[np.argmax(counts)]

            # Calculate total mismatches with new copy
            total_mm = get_total_mismatches(start, temp_end, temp_consensus, motif_len)
            total_length = temp_copies * motif_len
            max_mm_for_array = self._get_max_mismatches_for_array(motif_len, temp_copies)

            if total_mm <= max_mm_for_array:
                # Accept the new copy
                copies = temp_copies
                end = temp_end
                consensus = temp_consensus
            else:
                break

        # Extend left
        while start - motif_len >= 0:
            prev_copy = text_arr[start - motif_len:start]

            # Tentatively add this copy
            temp_copies = copies + 1
            temp_start = start - motif_len

            # Collect all copies including the new one
            all_copies = []
            for i in range(temp_copies):
                copy_start = temp_start + i * motif_len
                copy_end = copy_start + motif_len
                if copy_end <= n:
                    all_copies.append(text_arr[copy_start:copy_end])

            # Build temporary consensus
            temp_consensus = np.zeros(motif_len, dtype=np.uint8)
            for pos in range(motif_len):
                bases = [copy[pos] for copy in all_copies if pos < len(copy)]
                if bases:
                    unique, counts = np.unique(bases, return_counts=True)
                    temp_consensus[pos] = unique[np.argmax(counts)]

            # Calculate total mismatches with new copy
            total_mm = get_total_mismatches(temp_start, end, temp_consensus, motif_len)
            total_length = temp_copies * motif_len
            max_mm_for_array = self._get_max_mismatches_for_array(motif_len, temp_copies)

            if total_mm <= max_mm_for_array:
                # Accept the new copy
                copies = temp_copies
                start = temp_start
                consensus = temp_consensus
            else:
                break

        return start, end, copies

    def _is_maximal_repeat_approx(self, start: int, end: int, consensus: np.ndarray,
                                  motif_len: int, max_mismatches: int) -> bool:
        """Check if repeat is maximal (cannot be extended) with mismatch tolerance."""
        text_arr = self.bwt.text_arr
        n = text_arr.size

        # Check left extension
        if start >= motif_len:
            left_copy = text_arr[start - motif_len:start]
            if MotifUtils.hamming_distance_array(left_copy, consensus) <= max_mismatches:
                return False

        # Check right extension
        if end + motif_len <= n:
            right_copy = text_arr[end:end + motif_len]
            if MotifUtils.hamming_distance_array(right_copy, consensus) <= max_mismatches:
                return False

        return True
    
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
    """Tier 2: BWT/FM-index based repeat finder for ALL motif lengths with imperfect repeat support.

    Handles both short repeats (with mismatches) and medium/long repeats using:
    - FM-index backward search for motif occurrences
    - LCP arrays for longer period detection
    - Seed-and-extend with mismatch tolerance
    """

    def __init__(self, bwt_core: BWTCore, min_period: int = 1, max_period: int = 1000,
                 max_short_motif: int = 9, allow_mismatches: bool = True, show_progress: bool = False):
        self.bwt = bwt_core
        self.min_period = min_period  # Now starts at 1bp
        self.max_period = max_period
        self.max_short_motif = max_short_motif  # For FM-index search (1-9bp)
        self.min_copies = 3  # Require at least 3 copies
        self.min_array_length = 6  # Minimum total array length (for short repeats)
        self.min_entropy = 1.0  # Minimum Shannon entropy
        self.allow_mismatches = allow_mismatches
        self.show_progress = show_progress
        self.period_step = 1  # Step size for period scanning (increase to speed up)

    def _hamming_distance(self, arr1: np.ndarray, arr2: np.ndarray) -> int:
        """Calculate Hamming distance between two arrays."""
        return int(np.sum(arr1 != arr2))

    def find_long_unit_repeats_strict(self, chromosome: str, min_unit_len: int = 20,
                                      max_unit_len: int = 120, max_mismatch: int = 2,
                                      min_copies: int = 3) -> List[TandemRepeat]:
        """Find long-unit tandem repeats using strict adjacency checking.

        This detects biologically meaningful long repeats (20-120bp units) and avoids
        reporting nested short-motif periodicities inside them.

        Args:
            chromosome: Chromosome name
            min_unit_len: Minimum unit length to consider (default 20bp)
            max_unit_len: Maximum unit length to scan (default 120bp)
            max_mismatch: Maximum Hamming distance per unit comparison (default 2)
            min_copies: Minimum number of adjacent copies required (default 3)

        Returns:
            List of long-unit tandem repeats
        """
        repeats = []
        text_arr = self.bwt.text_arr
        n = int(text_arr.size)
        # print(f"[DEBUG] Searching chrom={chromosome}, seq_len={n}, min_unit={min_unit_len}, max_unit={max_unit_len}")

        # Exclude sentinel
        if n > 0 and text_arr[n - 1] == 36:  # '$' = 36
            n -= 1

        # For each candidate unit length - PROCESS IN REVERSE ORDER (longest first)
        # This ensures we detect [AT]n before [A]n, [GCG]n before [GC]n, etc.
        max_possible_unit = min(max_unit_len, n // min_copies)
        for unit_len in range(max_possible_unit, min_unit_len - 1, -1):
            i = 0
            while i + unit_len * min_copies <= n:
                # Test if there's at least one adjacency starting at i
                count = 1
                start_pos = i

                # Extend right while adjacency holds
                while True:
                    a_start = i + (count - 1) * unit_len
                    a_end = i + count * unit_len
                    b_start = i + count * unit_len
                    b_end = b_start + unit_len

                    if b_end > n:
                        break

                    a = text_arr[a_start:a_end]
                    b = text_arr[b_start:b_end]

                    if self._hamming_distance(a, b) <= max_mismatch:
                        count += 1
                    else:
                        break

                # If we found enough copies, create a repeat
                if count >= min_copies:
                    end_pos = i + count * unit_len
                    length = end_pos - i

                    # Get the first unit as the motif
                    motif_arr = text_arr[i:i + unit_len]
                    motif = motif_arr.tobytes().decode('ascii', errors='replace')

                    # Reduce to primitive period (e.g., 105bp -> 36bp if 105 is periodic)
                    primitive_period = MotifUtils.smallest_period_str(motif)
                    if primitive_period < len(motif):
                        # Use the primitive period instead
                        motif = motif[:primitive_period]
                        # Recalculate count based on primitive period
                        count = length // primitive_period

                    # Calculate consensus and statistics
                    (percent_matches, percent_indels, score, composition,
                     entropy, actual_sequence) = MotifUtils.calculate_trf_statistics(
                        text_arr, i, end_pos, motif, count, 0.0
                    )

                    # Calculate actual mismatches (perfect repeats have 100% matches)
                    actual_mismatches_per_copy = 0 if percent_matches >= 99.9 else max_mismatch

                    repeat = TandemRepeat(
                        chrom=chromosome,
                        start=i,
                        end=end_pos,
                        motif=motif,
                        copies=float(count),
                        length=length,
                        tier=2,  # Tier 2 for long units
                        confidence=0.95,
                        consensus_motif=motif,
                        mismatch_rate=0.0,
                        max_mismatches_per_copy=actual_mismatches_per_copy,
                        n_copies_evaluated=count,
                        strand='+',
                        percent_matches=percent_matches,
                        percent_indels=percent_indels,
                        score=score,
                        composition=composition,
                        entropy=entropy,
                        actual_sequence=actual_sequence,
                        variations=None
                    )
                    # if 'test8' in chromosome or 'test10' in chromosome:
                    #     print(f"[DEBUG] Found repeat: chrom={chromosome}, start={i}, end={end_pos}, motif={motif[:20]}{'...' if len(motif)>20 else ''}, unit_len={unit_len}, copies={count}")
                    repeats.append(repeat)
                    i = end_pos  # Jump past this repeat
                else:
                    i += 1

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

        # For single nucleotide repeats (homopolymers), NO mismatches allowed
        if motif_len == 1:
            return 0

        # For short motifs (2-6bp), be very conservative
        if motif_len <= 6:
            # Allow at most 5% mismatches (1 in 20 bases)
            return max(1, int(np.ceil(0.05 * total_length)))

        # For longer motifs, allow up to 8% mismatches
        return max(1, int(np.ceil(0.08 * total_length)))
    
    def find_short_imperfect_repeats(self, chromosome: str, tier1_seen: Set[Tuple[int, int]]) -> List[TandemRepeat]:
        """Find short imperfect tandem repeats (1-9bp) using FM-index with mismatch tolerance.

        This is called after Tier1 to find imperfect repeats that the perfect-match sliding window missed.

        Args:
            chromosome: Chromosome name
            tier1_seen: Set of (start, end) regions already found by Tier1

        Returns:
            List of imperfect tandem repeats
        """
        repeats = []
        seen_regions = tier1_seen.copy()  # Don't overlap with Tier1 results

        # Use the BWT-based methods from old Tier1
        text_arr = self.bwt.text_arr
        n = text_arr.size

        # IMPORTANT: Skip BWT search for large chromosomes to prevent stalling
        # Tier 1 already found perfect repeats, and BWT search is too expensive for large sequences
        if n > 1_000_000:  # > 1 Mbp
            if self.show_progress:
                print(f"  [{chromosome}] Tier 2 short imperfect repeats: SKIPPED (>{n:,} bp, too expensive)")
            return repeats  # Return empty list

        for k in range(self.min_period, min(self.max_short_motif + 1, 10)):
            # Check entropy threshold for motif length
            for motif in MotifUtils.enumerate_motifs(k):
                entropy = MotifUtils.calculate_entropy(motif)
                if entropy < self.min_entropy:
                    continue  # Skip low-complexity motifs

                # Search for all rotations of the canonical motif AND its reverse complement
                all_positions = []

                # Generate rotations of forward canonical motif
                forward_rotations = [motif[i:] + motif[:i] for i in range(len(motif))]

                # Generate rotations of reverse complement
                rc_motif = MotifUtils.reverse_complement(motif)
                rc_rotations = [rc_motif[i:] + rc_motif[:i] for i in range(len(rc_motif))]

                # Combine and deduplicate (some motifs are palindromic)
                rotations = list(set(forward_rotations + rc_rotations))

                for rotation in rotations:
                    # Fast k-mer lookup for short motifs
                    if k <= 8:
                        rot_positions = self.bwt.get_kmer_positions(rotation)
                    else:
                        rot_count = self.bwt.count_occurrences(rotation)
                        rot_positions = self.bwt.locate_positions(rotation) if rot_count > 0 else []

                    all_positions.extend(rot_positions)

                # Remove duplicates and sort
                positions = sorted(set(all_positions))

                if len(positions) >= self.min_copies:
                    if self.allow_mismatches:
                        # Use the seed-and-extend with mismatches logic
                        # (We'll need to copy those methods from old Tier1)
                        tandem_repeats = self._find_tandems_fm_with_mismatches(
                            positions, motif, chromosome, k, seen_regions
                        )
                        repeats.extend(tandem_repeats)

        return repeats

    def find_long_repeats(self, chromosome: str, tier1_seen: Optional[Set[Tuple[int, int]]] = None) -> List[TandemRepeat]:
        """Find medium to long tandem repeats using a lightweight period scan.

        This avoids building large LCP structures and is fast for moderate sequences.

        Args:
            chromosome: Chromosome name
            tier1_seen: Set of (start, end) regions already found by Tier1 (to skip)
        """
        return self._find_repeats_simple(chromosome, tier1_seen or set())
    
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

    def _find_repeats_simple(self, chromosome: str, tier1_seen: Set[Tuple[int, int]]) -> List[TandemRepeat]:
        """Simple scanning detector for tandem repeats with imperfect repeat support.

        Optimized for long sequences with adaptive scanning.

        Args:
            chromosome: Chromosome name
            tier1_seen: Set of (start, end) regions already found by Tier1 (to skip)
        """
        s_arr = self.bwt.text_arr
        n = int(s_arr.size)
        # Exclude trailing sentinel if present ('$' == 36)
        if n > 0 and s_arr[n - 1] == 36:
            n -= 1
        # AGGRESSIVE max_period limits to prevent stalling
        max_p = min(self.max_period, max(1, n // 2))
        if n > 100_000:  # >100 Kbp
            max_p = min(max_p, 30)  # Very conservative for large sequences
        elif n > 10_000:  # >10 Kbp
            max_p = min(max_p, 50)
        elif n > 1_000:  # >1 Kbp
            max_p = min(max_p, 100)
        else:
            max_p = min(max_p, 200)  # Small sequences can have longer periods
        min_p = min(self.min_period, max_p)
        results: List[TandemRepeat] = []
        seen: Set[Tuple[int, int, str]] = set()

        # Optimize masking: create a bitmap for fast lookups
        # This prevents O(n) checks on every iteration
        tier1_mask = np.zeros(n, dtype=bool)
        for start, end in tier1_seen:
            tier1_mask[start:min(end, n)] = True

        # AGGRESSIVE adaptive scanning to prevent stalling
        # The _extend_with_mismatches is VERY expensive, so we need aggressive sampling
        if n > 10_000_000:  # >10 Mbp
            position_step = 500  # Very aggressive
            period_step = 20
        elif n > 5_000_000:  # >5 Mbp
            position_step = 200
            period_step = 10
        elif n > 1_000_000:  # >1 Mbp
            position_step = 100
            period_step = 5
        elif n > 100_000:  # >100 Kbp
            position_step = 50
            period_step = 2
        elif n > 10_000:  # >10 Kbp
            position_step = 20
            period_step = 1
        else:
            position_step = 10  # Even tiny sequences need sampling
            period_step = 1

        if self.show_progress and (position_step > 1 or period_step > 1):
            print(f"    Tier 2 adaptive mode: position_step={position_step}, period_step={period_step} (faster, may miss some repeats)")

        # Safety counter to prevent infinite loops and timeout
        max_iterations = 100_000  # Reduced from 1M - be very aggressive
        iteration_count = 0
        start_time = time.time()
        max_time_seconds = 30  # Maximum 30 seconds per chromosome for Tier 2

        for p in range(min_p, max_p + 1, period_step):
            i = 0
            while i + 2 * p <= n:
                iteration_count += 1

                # Check both iteration limit AND time limit
                if iteration_count > max_iterations:
                    if self.show_progress:
                        print(f"  [{chromosome}] WARNING: Hit iteration limit ({max_iterations:,}), stopping Tier 2 scan")
                    return results  # Return what we have so far

                if iteration_count % 1000 == 0:  # Check time every 1000 iterations
                    elapsed = time.time() - start_time
                    if elapsed > max_time_seconds:
                        if self.show_progress:
                            print(f"  [{chromosome}] WARNING: Tier 2 timeout ({elapsed:.1f}s > {max_time_seconds}s), stopping scan")
                        return results

                # MASKING: Skip regions already found by Tier 1 (O(1) bitmap lookup)
                if i < n and tier1_mask[i]:
                    i += position_step
                    continue

                motif_view = s_arr[i:i + p]

                # Skip motifs containing '$'(36) or 'N'(78)
                if np.any(motif_view == 36) or np.any(motif_view == 78):
                    i += position_step
                    continue

                # Check entropy to avoid low-complexity regions
                motif_str_temp = motif_view.tobytes().decode('ascii', errors='replace')
                if MotifUtils.calculate_entropy(motif_str_temp) < self.min_entropy:
                    i += position_step
                    continue

                allow_mm = self.allow_mismatches and p <= 64
                array_start, array_end, copies_full, full_start, full_end = self._extend_with_mismatches(
                    s_arr, i, p, n, allow_mismatches=allow_mm
                )

                array_length = array_end - array_start
                if array_length < self.min_array_length:
                    i += position_step
                    continue

                full_length = copies_full * p
                partial_length = max(0, array_length - full_length)
                partial_fraction = (partial_length / p) if p > 0 else 0.0
                effective_copies_int = copies_full + (1 if partial_fraction >= 0.75 else 0)
                effective_copies_float = copies_full + partial_fraction

                if (copies_full >= self.min_copies or effective_copies_int >= self.min_copies):
                    # Normalize motif to its primitive period
                    motif_view = s_arr[full_start:full_start + p]
                    prim = self._smallest_period_codes(motif_view)
                    if prim < p:
                        motif_view = motif_view[:prim]
                        p_eff = prim
                    else:
                        p_eff = p

                    # Recalculate with primitive period
                    allow_mm_eff = self.allow_mismatches and p_eff <= 64
                    array_start, array_end, copies_full, full_start, full_end = self._extend_with_mismatches(
                        s_arr, full_start, p_eff, n, allow_mismatches=allow_mm_eff
                    )
                    array_length = array_end - array_start
                    full_length = copies_full * p_eff
                    partial_length = max(0, array_length - full_length)
                    partial_fraction = (partial_length / p_eff) if p_eff > 0 else 0.0
                    effective_copies_int = copies_full + (1 if partial_fraction >= 0.75 else 0)
                    effective_copies_float = copies_full + partial_fraction

                    if (copies_full < self.min_copies and effective_copies_int < self.min_copies):
                        i += position_step
                        continue

                    # Build consensus motif
                    consensus_arr, mm_rate, max_mm_per_copy = MotifUtils.build_consensus_motif_array(
                        s_arr, full_start, p_eff, copies_full
                    )

                    if consensus_arr.size == 0:
                        i += position_step
                        continue

                    consensus_str = consensus_arr.tobytes().decode('ascii', errors='replace')

                    prim_len = MotifUtils.smallest_period_str(consensus_str)
                    if prim_len < len(consensus_str):
                        p_eff = prim_len
                        copies_full = max(1, (array_end - array_start) // p_eff)
                        array_end = array_start + copies_full * p_eff
                        consensus_arr, mm_rate, max_mm_per_copy = MotifUtils.build_consensus_motif_array(
                            s_arr, array_start, p_eff, copies_full
                        )
                        if consensus_arr.size == 0:
                            i += position_step
                            continue
                        consensus_str = consensus_arr.tobytes().decode('ascii', errors='replace')

                    # Get canonical motif considering both strands (deduplication only)
                    canonical, strand = MotifUtils.get_canonical_motif_stranded(consensus_str)

                    key = (array_start, array_end, canonical)
                    if key not in seen:
                        seen.add(key)
                        # Calculate confidence based on mismatch rate
                        confidence = max(0.5, 0.95 - mm_rate)

                        # Calculate TRF-compatible statistics
                        (percent_matches, percent_indels, score, composition,
                         entropy, actual_sequence) = MotifUtils.calculate_trf_statistics(
                            s_arr, array_start, array_end, consensus_str, effective_copies_float, mm_rate
                        )

                        variations = MotifUtils.summarize_variations_array(
                            s_arr, array_start, array_end, p_eff, consensus_arr
                        )
                        observed_copies = float(copies_full)
                        results.append(
                            TandemRepeat(
                                chrom=chromosome,
                                start=array_start,
                                end=array_end,
                                motif=consensus_str,
                                copies=observed_copies,
                                length=array_end - array_start,
                                tier=2,
                                confidence=confidence,
                                consensus_motif=consensus_str,
                                mismatch_rate=mm_rate,
                                max_mismatches_per_copy=max_mm_per_copy,
                                n_copies_evaluated=copies_full,
                                strand=strand,
                                percent_matches=percent_matches,
                                percent_indels=percent_indels,
                                score=score,
                                composition=composition,
                                entropy=entropy,
                                actual_sequence=actual_sequence,
                                variations=variations if variations else None
                            )
                        )
                    i = array_end  # jump past this repeat
                else:
                    i += position_step

        return results

    def _extend_with_mismatches(self, s_arr: np.ndarray, start_pos: int,
                               period: int, n: int, allow_mismatches: bool = True
                               ) -> Tuple[int, int, int, int, int]:
        """Extend tandem array with mismatch tolerance (10% of full array length).

        Returns:
            (array_start, array_end, copies, full_start, full_end)
        """
        motif = s_arr[start_pos:start_pos + period].copy()
        start = start_pos
        end = start_pos + period
        copies = 1
        consensus = motif.copy()

        def get_total_mismatches(start_pos_inner, end_pos_inner, consensus_arr, period_len):
            num_copies_inner = (end_pos_inner - start_pos_inner) // period_len
            total_mm = 0
            for i in range(num_copies_inner):
                copy_start = start_pos_inner + i * period_len
                copy_end = copy_start + period_len
                if copy_end <= n:
                    copy = s_arr[copy_start:copy_end]
                    total_mm += MotifUtils.hamming_distance_array(copy, consensus_arr)
            return total_mm

        # Extend right with complete copies
        while end + period <= n:
            temp_copies = copies + 1
            temp_end = end + period

            all_copies = []
            for i in range(temp_copies):
                copy_start = start + i * period
                copy_end = copy_start + period
                if copy_end <= n:
                    all_copies.append(s_arr[copy_start:copy_end])

            temp_consensus = np.zeros(period, dtype=np.uint8)
            for pos in range(period):
                bases = [copy[pos] for copy in all_copies if pos < len(copy)]
                if bases:
                    unique, counts = np.unique(bases, return_counts=True)
                    temp_consensus[pos] = unique[np.argmax(counts)]

            total_mm = get_total_mismatches(start, temp_end, temp_consensus, period)
            max_mm_for_array = (self._get_max_mismatches_for_array(period, temp_copies)
                                if allow_mismatches else 0)

            if total_mm <= max_mm_for_array:
                copies = temp_copies
                end = temp_end
                consensus = temp_consensus
            else:
                break

        # Extend left with complete copies
        while start - period >= 0:
            temp_copies = copies + 1
            temp_start = start - period

            all_copies = []
            for i in range(temp_copies):
                copy_start = temp_start + i * period
                copy_end = copy_start + period
                if copy_end <= n:
                    all_copies.append(s_arr[copy_start:copy_end])

            temp_consensus = np.zeros(period, dtype=np.uint8)
            for pos in range(period):
                bases = [copy[pos] for copy in all_copies if pos < len(copy)]
                if bases:
                    unique, counts = np.unique(bases, return_counts=True)
                    temp_consensus[pos] = unique[np.argmax(counts)]

            total_mm = get_total_mismatches(temp_start, end, temp_consensus, period)
            max_mm_for_array = (self._get_max_mismatches_for_array(period, temp_copies)
                                if allow_mismatches else 0)

            if total_mm <= max_mm_for_array:
                copies = temp_copies
                start = temp_start
                consensus = temp_consensus
            else:
                break

        full_start = start
        full_end = end

        # Extend right with partial copy (exact matches only)
        partial_right = 0
        while partial_right < period and full_end + partial_right < n:
            consensus_idx = partial_right % period
            if s_arr[full_end + partial_right] != consensus[consensus_idx]:
                break
            partial_right += 1
        array_end = full_end + partial_right

        # Extend left with partial copy (exact matches only)
        partial_left = 0
        while partial_left < period and full_start - partial_left - 1 >= 0:
            consensus_idx = (period - 1 - (partial_left % period))
            if s_arr[full_start - partial_left - 1] != consensus[consensus_idx]:
                break
            partial_left += 1
        array_start = full_start - partial_left

        return array_start, array_end, copies, full_start, full_end
    
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

    def _find_tandems_fm_with_mismatches(self, positions: List[int], motif: str,
                                         chromosome: str, motif_len: int,
                                         seen_regions: Set[Tuple[int, int]]) -> List[TandemRepeat]:
        """Find tandem repeats allowing mismatches using seed-and-extend strategy (for FM-index results)."""
        repeats = []
        if not positions:
            return repeats

        positions_sorted = sorted(positions)
        max_mm = 0  # Parameter not used
        text_arr = self.bwt.text_arr

        for seed_pos in positions_sorted:
            # Skip if this position is already part of a found repeat
            if any(start <= seed_pos < end for start, end in seen_regions):
                continue

            # Skip seeds whose window would exceed the chromosome boundary
            if seed_pos + motif_len > text_arr.size:
                continue

            # Explore all rotations that could anchor this seed
            best_start = None
            best_end = None
            best_copies = 0

            max_left_shift = min(motif_len, seed_pos + 1)
            for shift in range(max_left_shift):
                candidate_seed = seed_pos - shift
                candidate_end = candidate_seed + motif_len

                if candidate_seed < 0 or candidate_end > text_arr.size:
                    continue

                if any(start <= candidate_seed < end for start, end in seen_regions):
                    continue

                candidate_arr = text_arr[candidate_seed:candidate_end]
                candidate_motif = candidate_arr.tobytes().decode('ascii', errors='replace')

                start_pos_candidate, end_pos_candidate, copies_candidate = self._extend_tandem_fm(
                    text_arr, candidate_seed, candidate_motif, motif_len, max_mm
                )

                if not (start_pos_candidate <= seed_pos < end_pos_candidate):
                    continue

                if (copies_candidate > best_copies or
                        (copies_candidate == best_copies and
                         (best_start is None or start_pos_candidate < best_start))):
                    best_start = start_pos_candidate
                    best_end = end_pos_candidate
                    best_copies = copies_candidate

            if best_start is None:
                continue

            start_pos, end_pos, copies = best_start, best_end, best_copies
            array_length = end_pos - start_pos

            if copies >= self.min_copies and array_length >= self.min_array_length:
                consensus_arr, mm_rate, max_mm_per_copy = MotifUtils.build_consensus_motif_array(
                    text_arr, start_pos, motif_len, copies
                )

                if consensus_arr.size == 0:
                    continue

                consensus_str = consensus_arr.tobytes().decode('ascii', errors='replace')

                # Check for primitive motif
                primitive_len = MotifUtils.smallest_period_str(consensus_str)
                if primitive_len < len(consensus_str):
                    motif_len = primitive_len
                    copies = max(1, (end_pos - start_pos) // motif_len)
                    end_pos = start_pos + copies * motif_len
                    consensus_arr, mm_rate, max_mm_per_copy = MotifUtils.build_consensus_motif_array(
                        text_arr, start_pos, motif_len, copies
                    )
                    if consensus_arr.size == 0:
                        continue
                    consensus_str = consensus_arr.tobytes().decode('ascii', errors='replace')

                canonical, strand = MotifUtils.get_canonical_motif_stranded(consensus_str)

                # Check maximality
                if self._is_maximal_fm(start_pos, end_pos, consensus_arr, motif_len, max_mm):
                    confidence = max(0.5, 1.0 - mm_rate)

                    (percent_matches, percent_indels, score, composition,
                     entropy, actual_sequence) = MotifUtils.calculate_trf_statistics(
                        text_arr, start_pos, end_pos, consensus_str, copies, mm_rate
                    )

                    # CRITICAL: Reject repeats with low match rates (garbage detection)
                    # For Tier2, we need at least 90% matches for short motifs, 85% for longer ones
                    min_match_threshold = 90.0 if motif_len <= 6 else 85.0
                    if percent_matches < min_match_threshold:
                        continue  # Skip this garbage repeat

                    # Also reject if too many indels (indicates poor alignment)
                    if percent_indels > 5.0:
                        continue

                    variations = MotifUtils.summarize_variations_array(
                        text_arr, start_pos, end_pos, motif_len, consensus_arr
                    )

                    repeat = TandemRepeat(
                        chrom=chromosome,
                        start=start_pos,
                        end=end_pos,
                        motif=consensus_str,
                        copies=copies,
                        length=end_pos - start_pos,
                        tier=2,  # This is now Tier2
                        confidence=confidence,
                        consensus_motif=consensus_str,
                        mismatch_rate=mm_rate,
                        max_mismatches_per_copy=max_mm_per_copy,
                        n_copies_evaluated=copies,
                        strand=strand,
                        percent_matches=percent_matches,
                        percent_indels=percent_indels,
                        score=score,
                        composition=composition,
                        entropy=entropy,
                        actual_sequence=actual_sequence,
                        variations=variations if variations else None
                    )
                    repeats.append(repeat)
                    seen_regions.add((start_pos, end_pos))

        return repeats

    def _extend_tandem_fm(self, text_arr: np.ndarray, seed_pos: int,
                         motif: str, motif_len: int, max_mismatches: int) -> Tuple[int, int, int]:
        """Extend tandem array left and right from seed position (FM-index version)."""
        motif_arr = np.frombuffer(motif.encode('ascii'), dtype=np.uint8)
        n = text_arr.size

        start = seed_pos
        end = seed_pos + motif_len
        copies = 1
        consensus = motif_arr.copy()

        def get_total_mismatches(start_pos, end_pos, consensus_arr, motif_length):
            num_copies = (end_pos - start_pos) // motif_length
            total_mm = 0
            for i in range(num_copies):
                copy_start = start_pos + i * motif_length
                copy_end = copy_start + motif_length
                if copy_end <= n:
                    copy = text_arr[copy_start:copy_end]
                    total_mm += MotifUtils.hamming_distance_array(copy, consensus_arr)
            return total_mm

        def get_transversions(start_pos, end_pos, consensus_arr, motif_length):
            """Count transversions (non-transition mismatches) across all copies."""
            num_copies = (end_pos - start_pos) // motif_length
            total_transv = 0
            for i in range(num_copies):
                copy_start = start_pos + i * motif_length
                copy_end = copy_start + motif_length
                if copy_end <= n:
                    copy = text_arr[copy_start:copy_end]
                    total_transv += MotifUtils.count_transversions_array(copy, consensus_arr)
            return total_transv

        # Extend right
        while end + motif_len <= n:
            next_copy = text_arr[end:end + motif_len]

            # Don't extend into homopolymers (likely a different repeat)
            if motif_len > 1 and len(set(next_copy)) == 1:
                break

            temp_copies = copies + 1
            temp_end = end + motif_len

            all_copies = []
            for i in range(temp_copies):
                copy_start = start + i * motif_len
                copy_end = copy_start + motif_len
                if copy_end <= n:
                    all_copies.append(text_arr[copy_start:copy_end])

            temp_consensus = np.zeros(motif_len, dtype=np.uint8)
            for pos in range(motif_len):
                bases = [copy[pos] for copy in all_copies if pos < len(copy)]
                if bases:
                    unique, counts = np.unique(bases, return_counts=True)
                    temp_consensus[pos] = unique[np.argmax(counts)]

            total_mm = get_total_mismatches(start, temp_end, temp_consensus, motif_len)
            total_transv = get_transversions(start, temp_end, temp_consensus, motif_len)
            max_mm_for_array = self._get_max_mismatches_for_array(motif_len, temp_copies)

            # Reject if: too many total mismatches OR any transversions present
            if total_mm <= max_mm_for_array and total_transv == 0:
                copies = temp_copies
                end = temp_end
                consensus = temp_consensus
            else:
                break

        # Extend left
        while start - motif_len >= 0:
            prev_copy = text_arr[start - motif_len:start]

            # Don't extend into homopolymers (likely a different repeat)
            if motif_len > 1 and len(set(prev_copy)) == 1:
                break

            temp_copies = copies + 1
            temp_start = start - motif_len

            all_copies = []
            for i in range(temp_copies):
                copy_start = temp_start + i * motif_len
                copy_end = copy_start + motif_len
                if copy_end <= n:
                    all_copies.append(text_arr[copy_start:copy_end])

            temp_consensus = np.zeros(motif_len, dtype=np.uint8)
            for pos in range(motif_len):
                bases = [copy[pos] for copy in all_copies if pos < len(copy)]
                if bases:
                    unique, counts = np.unique(bases, return_counts=True)
                    temp_consensus[pos] = unique[np.argmax(counts)]

            total_mm = get_total_mismatches(temp_start, end, temp_consensus, motif_len)
            total_transv = get_transversions(temp_start, end, temp_consensus, motif_len)
            max_mm_for_array = self._get_max_mismatches_for_array(motif_len, temp_copies)

            # Reject if: too many total mismatches OR any transversions present
            if total_mm <= max_mm_for_array and total_transv == 0:
                copies = temp_copies
                start = temp_start
                consensus = temp_consensus
            else:
                break

        return start, end, copies

    def _is_maximal_fm(self, start: int, end: int, consensus: np.ndarray,
                      motif_len: int, max_mm: int) -> bool:
        """Check if repeat is maximal (FM-index version)."""
        text_arr = self.bwt.text_arr
        n = text_arr.size

        if start > 0:
            left_char = text_arr[start - 1]
            expected_left = consensus[motif_len - 1]  # Last char of motif
            if left_char == expected_left:
                return False

        if end < n:
            right_char = text_arr[end]
            expected_right = consensus[0]  # First char of motif
            if right_char == expected_right:
                return False

        return True


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
                    motif_len = len(motif)
                    if motif_len == 0:
                        continue

                    text_arr = self.bwt.text_arr
                    max_len = text_arr.size
                    if max_len > 0 and text_arr[max_len - 1] == 36:  # Exclude sentinel '$'
                        max_len -= 1

                    if ref_start >= max_len:
                        continue

                    available_max_copies = max((max_len - ref_start) // motif_len, 0)
                    if available_max_copies == 0:
                        continue

                    copies_int = max(1, min(int(round(copies)), available_max_copies))
                    ref_end = ref_start + motif_len * copies_int

                    consensus_str = motif
                    mm_rate = 0.0
                    max_mm_per_copy = 0
                    percent_matches = 100.0
                    percent_indels = 0.0
                    score = 0
                    composition = MotifUtils.calculate_composition(consensus_str)
                    entropy = MotifUtils.calculate_entropy(consensus_str)
                    actual_sequence = (consensus_str * copies_int)[:max(ref_end - ref_start, 0)]
                    strand = '+'

                    if motif_len > 0 and ref_end > ref_start:
                        consensus_arr, mm_rate, max_mm_per_copy = MotifUtils.build_consensus_motif_array(
                            text_arr, ref_start, motif_len, copies_int
                        )
                        if consensus_arr.size:
                            consensus_str = consensus_arr.tobytes().decode('ascii', errors='replace')
                            motif_len = len(consensus_str)
                        if ref_end <= max_len:
                            (percent_matches, percent_indels, score, composition,
                             entropy, actual_sequence) = MotifUtils.calculate_trf_statistics(
                                text_arr, ref_start, ref_end, consensus_str, copies_int, mm_rate
                            )

                    canonical, strand = MotifUtils.get_canonical_motif_stranded(consensus_str)
                    consensus_arr_for_variation = np.frombuffer(consensus_str.encode('ascii'), dtype=np.uint8)
                    variations = MotifUtils.summarize_variations_array(
                        text_arr, ref_start, ref_end, motif_len, consensus_arr_for_variation
                    )

                    repeat = TandemRepeat(
                        chrom=chromosome,
                        start=ref_start,
                        end=ref_end,
                        motif=consensus_str,
                        copies=float(copies_int),
                        length=ref_end - ref_start,
                        tier=3,
                        confidence=confidence,
                        consensus_motif=consensus_str,
                        mismatch_rate=mm_rate,
                        max_mismatches_per_copy=max_mm_per_copy,
                        n_copies_evaluated=copies_int,
                        strand=strand,
                        percent_matches=percent_matches,
                        percent_indels=percent_indels,
                        score=score,
                        composition=composition,
                        entropy=entropy,
                        actual_sequence=actual_sequence,
                        variations=variations if variations else None
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
        repeats.sort(key=lambda r: (_natural_sort_key(r.chrom), r.start, r.end))
        
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


# Global worker function for multiprocessing
def _process_chromosome_worker(args):
    """Worker function for parallel chromosome processing.

    Args:
        args: Tuple of (chrom, seq, config_dict)

    Returns:
        List of TandemRepeat objects for this chromosome
    """
    chrom, seq, config = args

    try:
        # Build BWT for this chromosome
        seq_with_sentinel = seq + '$'
        bwt_core = BWTCore(seq_with_sentinel, config['sa_sample_rate'])

        repeats = []
        chrom_length = len(seq)

        if config.get('show_progress', False):
            print(f"  [{chrom}] Building indices ({chrom_length:,} bp)...")

        # UNIFIED STRICT ADJACENCY ARCHITECTURE:
        # Use only the strict adjacency detector for ALL repeat sizes (1-120bp)
        # This avoids nested calls and detects biological units cleanly
        tier1_seen = set()  # Keep for compatibility but won't be populated

        # Tier 2: BWT/FM-index for imperfect repeats AND medium/long repeats
        if config.get('enable_tier2', False):
            # Skip very large chromosomes unless forced
            if chrom_length > 50_000_000 and not config.get('show_progress', False):
                if config.get('show_progress', False):
                    print(f"  [{chrom}] Tier 2: SKIPPED (>50 Mbp)")
            else:
                tier2 = Tier2LCPFinder(
                    bwt_core,
                    min_period=1,  # Start from 1bp to handle short imperfect repeats
                    max_period=config['max_period'],
                    max_short_motif=config['max_motif_length'],
                    allow_mismatches=config['allow_mismatches'],
                    show_progress=config.get('show_progress', False)
                )
                tier2.min_copies = config['min_copies']
                tier2.min_entropy = config['min_entropy']

                # PRIORITY 1: Find all-unit repeats (1-max_unit_len bp) with strict adjacency
                # This detects all biologically meaningful repeats from short to long

                # Auto-adjust max_unit_len based on sequence length if needed
                seq_len = len(seq)
                min_copies = config['min_copies']
                base_max_unit = config['max_unit_len']

                # Calculate adaptive max_unit_len: allow motifs up to seq_len / min_copies
                # but cap at a reasonable maximum for performance
                adaptive_max_unit = min(seq_len // min_copies, 1000)  # Cap at 1000bp

                # Use the larger of user-specified or adaptive value
                effective_max_unit = max(base_max_unit, adaptive_max_unit)

                # Use strict perfect matching (max_mismatch=0) to avoid spurious long motifs
                # Allowing mismatches during scanning creates false longer motifs that mask
                # true shorter perfect repeats (e.g., 14bp imperfect masking 7bp perfect)
                tier2_long_unit_repeats = tier2.find_long_unit_repeats_strict(
                    chrom, min_unit_len=1, max_unit_len=effective_max_unit,
                    max_mismatch=0, min_copies=min_copies
                )
                repeats.extend(tier2_long_unit_repeats)

                # DONE - Only use strict adjacency detector
                # No need for other Tier 2 methods as they may create nested calls

                if config.get('show_progress', False):
                    print(f"  [{chrom}] Strict adjacency: {len(tier2_long_unit_repeats)} tandem repeats detected")

        # Filter out imperfect repeats for very short motifs with low copy numbers
        # Rule 1: motif < 5bp AND copies < 30 => must be perfect (no SNPs/indels)
        # This prevents spurious short repeats but allows longer motifs (5-9bp) to have variations
        filtered_repeats = []
        for repeat in repeats:
            motif = repeat.consensus_motif or repeat.motif
            motif_len = len(motif)
            copies = repeat.copies

            # Rule 1: For very short motifs with low copy numbers, require perfect matches
            if motif_len < 5 and copies < 30:
                # Reject if there are any mismatches
                if repeat.mismatch_rate > 0 or repeat.max_mismatches_per_copy > 0:
                    continue  # Skip this imperfect repeat

            filtered_repeats.append(repeat)

        # Clean up
        bwt_core.clear()

        return filtered_repeats

    except Exception as e:
        print(f"ERROR processing chromosome {chrom}: {e}")
        import traceback
        traceback.print_exc()
        return []


class TandemRepeatFinder:
    """Main class coordinating all three tiers of tandem repeat finding."""

    def __init__(self, reference_file: str, sa_sample_rate: int = 32, show_progress: bool = False,
                 allow_mismatches: bool = True, max_motif_length: int = 9,
                 min_period: int = 10, max_period: int = 1000,
                 min_copies: int = 3, min_entropy: float = 1.0,
                 flank_trim: int = 30, max_unit_len: int = 120):
        """
        Initialize the tandem repeat finder.

        Args:
            reference_file: Path to reference genome FASTA
            sa_sample_rate: Suffix array sampling rate for space efficiency
            show_progress: Show progress information
            allow_mismatches: Allow imperfect repeats with SNPs
            max_motif_length: Maximum motif length for tier 1
            min_period: Minimum period for tier 2
            max_period: Maximum period for tier 2
            min_copies: Minimum number of copies required
            min_entropy: Minimum Shannon entropy to filter low-complexity
            flank_trim: Trim N bp from each end before analysis
            max_unit_len: Maximum unit length for tier 2 long repeat detection
        """
        self.reference_file = reference_file
        self.sa_sample_rate = sa_sample_rate
        self.bwt_cores = {}  # Store BWT for each chromosome
        self.sequences = {}  # Store sequences for parallel processing
        self.show_progress = show_progress
        self.allow_mismatches = allow_mismatches
        self.max_motif_length = max_motif_length
        self.min_period = min_period
        self.max_period = max_period
        self.min_copies = min_copies
        self.min_entropy = min_entropy
        self.flank_trim = max(0, flank_trim)
        self.max_unit_len = max_unit_len
        self.trim_offsets: Dict[str, int] = {}
        self.full_sequences: Dict[str, str] = {}
    
    @staticmethod
    def _repeat_sort_key(repeat: TandemRepeat):
        """Key used for sorting repeats by chromosome (natural) and coordinates."""
        return (_natural_sort_key(repeat.chrom), repeat.start, repeat.end)
    
    def _deduplicate_repeats(self, repeats: List[TandemRepeat]) -> List[TandemRepeat]:
        """Remove redundant repeat calls across tiers, keeping the highest-confidence entry."""
        if not repeats:
            return []

        # print(f"[DEBUG] Deduplicating {len(repeats)} repeats...")
        # for i, r in enumerate(repeats):
        #     print(f"[DEBUG]   Input #{i}: chrom={r.chrom}, pos={r.start}-{r.end}, motif={r.motif[:10]}{'...' if len(r.motif)>10 else ''}")

        dedup: Dict[Tuple[str, int, int, str], TandemRepeat] = {}
        for repeat in repeats:
            key = (repeat.chrom, repeat.start, repeat.end, repeat.motif)
            existing = dedup.get(key)
            if existing is None:
                dedup[key] = repeat
                continue

            # Prefer higher confidence; then lower mismatch rate; then lower tier.
            if repeat.confidence > existing.confidence:
                dedup[key] = repeat
            elif repeat.confidence == existing.confidence:
                if repeat.mismatch_rate < existing.mismatch_rate:
                    dedup[key] = repeat
                elif repeat.mismatch_rate == existing.mismatch_rate and repeat.tier < existing.tier:
                    dedup[key] = repeat

        deduped = list(dedup.values())
        deduped.sort(key=self._repeat_sort_key)
        # print(f"[DEBUG] After dedup: {len(deduped)} unique repeats")
        # for i, r in enumerate(deduped):
        #     print(f"[DEBUG]   Output #{i}: chrom={r.chrom}, pos={r.start}-{r.end}, motif={r.motif[:10]}{'...' if len(r.motif)>10 else ''}")
        return deduped
    
    def _merge_adjacent_repeats(self, repeats: List[TandemRepeat]) -> List[TandemRepeat]:
        """Merge overlapping or near-adjacent repeats with matching canonical motifs."""
        if not repeats:
            return []

        merged: List[TandemRepeat] = []
        current = repeats[0]

        for nxt in repeats[1:]:
            if self._should_merge_repeats(current, nxt):
                current = self._merge_repeats(current, nxt)
            else:
                merged.append(current)
                current = nxt

        merged.append(current)
        return merged

    def _should_merge_repeats(self, r1: TandemRepeat, r2: TandemRepeat) -> bool:
        if r1.chrom != r2.chrom:
            return False

        motif1 = r1.consensus_motif or r1.motif
        motif2 = r2.consensus_motif or r2.motif
        motif_len1 = len(motif1)
        motif_len2 = len(motif2)
        if motif_len1 == 0 or motif_len2 == 0:
            return False

        # Don't merge repeats with different canonical motifs
        canon1, _ = MotifUtils.get_canonical_motif_stranded(motif1)
        canon2, _ = MotifUtils.get_canonical_motif_stranded(motif2)
        if canon1 != canon2:
            return False

        min_len = min(motif_len1, motif_len2)
        gap = max(0, r2.start - r1.end)
        if gap > min_len + 1:
            return False

        merged_start = min(r1.start, r2.start)
        merged_end = max(r1.end, r2.end)
        initial_len = max(1, min_len)

        try:
            merged = self._recompute_repeat(
                r1.chrom,
                merged_start,
                merged_end,
                initial_len,
                tier_hint=min(r1.tier, r2.tier)
            )
        except ValueError:
            return False

        if merged.copies < self.min_copies:
            return False

        baseline_mismatch = max(r1.mismatch_rate, r2.mismatch_rate, 0.01)
        return merged.mismatch_rate <= baseline_mismatch + 0.2

    def _merge_repeats(self, r1: TandemRepeat, r2: TandemRepeat) -> TandemRepeat:
        chrom = r1.chrom
        start = min(r1.start, r2.start)
        end = max(r1.end, r2.end)
        motif_len = len(r1.consensus_motif or r1.motif)
        merged = self._recompute_repeat(chrom, start, end, motif_len, tier_hint=min(r1.tier, r2.tier))
        return merged

    def _refine_repeats(self, repeats: List[TandemRepeat]) -> List[TandemRepeat]:
        """Recompute consensus, copy counts, and variations for each repeat block."""
        refined: List[TandemRepeat] = []
        for repeat in repeats:
            # Skip refinement for perfect repeats (mismatch_rate = 0.0)
            # Recomputation can introduce errors by extending into adjacent regions
            if repeat.mismatch_rate == 0.0:
                refined.append(repeat)
                continue

            motif_source = repeat.consensus_motif or repeat.motif
            motif_len = len(motif_source)
            if motif_len <= 0:
                motif_len = max(1, repeat.length // max(1, int(round(repeat.copies)) or 1))
            refined_repeat = self._recompute_repeat(
                repeat.chrom,
                repeat.start,
                repeat.end,
                motif_len,
                tier_hint=repeat.tier
            )
            refined.append(refined_repeat)
        refined.sort(key=self._repeat_sort_key)
        return refined

    def _restore_reference_coordinates(self, repeats: List[TandemRepeat]) -> None:
        """Map repeat coordinates back to the original reference positions (undo flank trimming)."""
        for repeat in repeats:
            offset = self.trim_offsets.get(repeat.chrom, 0)
            repeat.start += offset
            repeat.end += offset
            repeat.length = repeat.end - repeat.start
            full_sequence = self.full_sequences.get(repeat.chrom)
            if full_sequence:
                repeat.actual_sequence = full_sequence[repeat.start:repeat.end]

    def _should_collapse_duplicates(self, r1: TandemRepeat, r2: TandemRepeat) -> bool:
        if r1.chrom != r2.chrom:
            return False

        overlap = min(r1.end, r2.end) - max(r1.start, r2.start)
        if overlap <= 0:
            return False

        shorter = min(r1.length, r2.length)
        if shorter <= 0:
            return False

        if overlap / shorter < 0.8:
            return False

        canon1, _ = MotifUtils.get_canonical_motif_stranded(r1.motif)
        canon2, _ = MotifUtils.get_canonical_motif_stranded(r2.motif)
        if canon1 == canon2:
            return True

        # Special case: Collapse homopolymers with longer motifs that fully overlap
        if (len(r1.motif) == 1 or len(r2.motif) == 1) and overlap / shorter >= 0.95:
            return True

        if len(r1.motif) == len(r2.motif) and overlap / shorter >= 0.9:
            return abs(r1.mismatch_rate - r2.mismatch_rate) >= 0.2

        return False

    @staticmethod
    def _prefer_repeat_entry(r1: TandemRepeat, r2: TandemRepeat) -> TandemRepeat:
        # Smart motif length preference:
        # 0. Always deprioritize homopolymers (A, T, G, C) when compared to longer motifs
        # 1. If one motif is a primitive root of the other (e.g., AT vs ATAT, GCG vs GCGGCG),
        #    prefer the primitive (shorter) one
        # 2. Otherwise, prefer longer motifs (more specific patterns like 125bp vs 3bp)

        motif1 = r1.consensus_motif or r1.motif
        motif2 = r2.consensus_motif or r2.motif
        motif1_len = len(motif1)
        motif2_len = len(motif2)

        if motif1_len != motif2_len:
            # Special case: Always prefer non-homopolymers over homopolymers
            if motif1_len == 1 and motif2_len > 1:
                return r2  # Prefer r2 (longer motif)
            if motif2_len == 1 and motif1_len > 1:
                return r1  # Prefer r1 (longer motif)

            # Check if one is a power of the other (e.g., ATAT = AT*2, GCGGCG = GCG*2)
            shorter, longer = (motif1, motif2) if motif1_len < motif2_len else (motif2, motif1)
            shorter_len = len(shorter)
            longer_len = len(longer)

            # Check if longer motif is made of repeated shorter motif
            if longer_len % shorter_len == 0:
                repetitions = longer_len // shorter_len
                if shorter * repetitions == longer:
                    # One is a primitive root of the other - prefer the shorter (primitive)
                    return r1 if motif1_len < motif2_len else r2

            # Not a primitive/composite pair - prefer longer motif (more specific)
            return r1 if motif1_len > motif2_len else r2

        # Then prefer lower mismatch rate
        if r1.mismatch_rate != r2.mismatch_rate:
            return r1 if r1.mismatch_rate < r2.mismatch_rate else r2
        # Then prefer higher confidence
        if r1.confidence != r2.confidence:
            return r1 if r1.confidence > r2.confidence else r2
        # Finally prefer longer total length
        if r1.length != r2.length:
            return r1 if r1.length >= r2.length else r2
        return r1

    def _suppress_nested_short_calls(self, repeats: List[TandemRepeat],
                                      overlap_threshold: float = 0.5) -> List[TandemRepeat]:
        """Suppress nested short-motif calls that overlap with longer-unit repeats.

        This implements the post-filter that prioritizes longer repeat units and
        removes nested short periodicities inside them.

        Args:
            repeats: List of all detected repeats
            overlap_threshold: Fraction of overlap required to suppress (default 0.5 = 50%)

        Returns:
            Filtered list with nested calls removed
        """
        if not repeats:
            return []

        # Group repeats by chromosome
        repeats_by_chrom: Dict[str, List[TandemRepeat]] = {}
        for repeat in repeats:
            if repeat.chrom not in repeats_by_chrom:
                repeats_by_chrom[repeat.chrom] = []
            repeats_by_chrom[repeat.chrom].append(repeat)

        all_kept_repeats = []

        # Process each chromosome independently
        for chrom, chrom_repeats in repeats_by_chrom.items():
            # if 'test8' in chrom or 'test10' in chrom:
            #     print(f"[DEBUG NESTED INPUT] {chrom}: {len(chrom_repeats)} repeats")
            #     for r in chrom_repeats:
            #         print(f"  [{r.motif}] at {r.start}-{r.end}")

            # Sort by: 1) Perfect repeats first (mismatch_rate=0), 2) Then by motif length descending
            # This ensures perfect short repeats are prioritized over imperfect long ones
            sorted_repeats = sorted(chrom_repeats, key=lambda r: (r.mismatch_rate > 0, -len(r.motif)))

            kept_repeats = []
            occupied_spans = []  # (start, end, motif_len) tuples

            for repeat in sorted_repeats:
                r_start, r_end = repeat.start, repeat.end
                r_len = r_end - r_start
                r_motif_len = len(repeat.motif)

                # if 'test8' in chrom or 'test10' in chrom:
                #     print(f"[DEBUG NESTED] Processing [{repeat.motif}] at {r_start}-{r_end}, len={r_motif_len}")

                # Check if this repeat overlaps significantly with any longer-unit repeat
                is_nested = False
                for occ_start, occ_end, occ_motif_len in occupied_spans:
                    # Only suppress if the occupied span has a longer motif
                    if occ_motif_len <= r_motif_len:
                        continue

                    # Calculate overlap
                    overlap_start = max(r_start, occ_start)
                    overlap_end = min(r_end, occ_end)
                    overlap_len = max(0, overlap_end - overlap_start)

                    if overlap_len == 0:
                        continue

                    # Adaptive threshold based on motif size ratio
                    motif_ratio = occ_motif_len / r_motif_len

                    # Special case: Always suppress homopolymers (1bp) if a longer motif (2bp+) overlaps significantly
                    if r_motif_len == 1 and occ_motif_len > 1 and overlap_len / r_len >= 0.8:
                        # Debug: print when suppressing homopolymers
                        # print(f"DEBUG: Suppressing homopolymer [{repeat.motif}] at {r_start}-{r_end} due to overlap with {occ_motif_len}bp motif")
                        is_nested = True
                        break

                    # If the longer motif is >10x bigger, use very aggressive suppression (any overlap)
                    if motif_ratio >= 10:
                        adaptive_threshold = 0.1  # Suppress if >10% overlap
                    # If the longer motif is >5x bigger, use aggressive suppression
                    elif motif_ratio >= 5:
                        adaptive_threshold = 0.3  # Suppress if >30% overlap
                    else:
                        adaptive_threshold = overlap_threshold  # Use default 50%

                    # If this repeat overlaps > threshold with a longer-unit repeat, suppress it
                    if overlap_len / r_len >= adaptive_threshold:
                        is_nested = True
                        break

                if not is_nested:
                    kept_repeats.append(repeat)
                    occupied_spans.append((r_start, r_end, r_motif_len))

            all_kept_repeats.extend(kept_repeats)

        # Restore original order
        all_kept_repeats.sort(key=self._repeat_sort_key)
        return all_kept_repeats

    def _collapse_overlapping_repeats(self, repeats: List[TandemRepeat]) -> List[TandemRepeat]:
        """Collapse redundant repeat calls that represent the same locus."""
        if not repeats:
            return []

        collapsed: List[TandemRepeat] = []
        for repeat in sorted(repeats, key=self._repeat_sort_key):
            if collapsed and self._should_collapse_duplicates(collapsed[-1], repeat):
                preferred = self._prefer_repeat_entry(collapsed[-1], repeat)
                # if 'test8' in repeat.chrom or 'test10' in repeat.chrom:
                #     print(f"[DEBUG COLLAPSE] {repeat.chrom}: Collapsing [{repeat.motif}] with [{collapsed[-1].motif}], keeping [{preferred.motif}]")
                collapsed[-1] = preferred
            else:
                collapsed.append(repeat)
        return collapsed

    def _recompute_repeat(self, chrom: str, start: int, end: int, motif_len: int,
                          tier_hint: int = 1) -> TandemRepeat:
        sequence = self.sequences.get(chrom)
        if sequence is None:
            raise ValueError(f"Sequence for chromosome {chrom} not available for recomputation.")

        seq_len = len(sequence)
        motif_len = max(1, motif_len)
        start = max(0, int(start))
        end = min(seq_len, int(end)) if end > 0 else seq_len
        if end <= start:
            end = min(seq_len, start + motif_len)

        motif_template = sequence[start:start + motif_len]
        if not motif_template:
            alt_start = max(0, start - motif_len)
            motif_template = sequence[alt_start:alt_start + motif_len]
        if not motif_template:
            motif_template = 'N' * motif_len

        summary = MotifUtils.align_repeat_region(
            sequence,
            start,
            end,
            motif_template,
            mismatch_fraction=0.1,
            min_copies=max(1, self.min_copies)
        )
        if summary is None:
            summary = MotifUtils.align_repeat_region(
                sequence,
                start,
                end,
                motif_template,
                mismatch_fraction=0.1,
                min_copies=1
            )

        if summary is None:
            consumed = min(seq_len - start, max(motif_len, end - start))
            actual_sequence = sequence[start:start + consumed]
            copies_int = max(1, consumed // motif_len)
            consensus = motif_template if motif_template else (actual_sequence[:motif_len] or 'N')
            mismatch_rate = 0.0
            max_errors = 0
            percent_indels = 0.0
            variations_out: Optional[List[str]] = None
        else:
            consumed = summary.consumed_length
            actual_sequence = sequence[start:start + consumed]
            copies_int = summary.copies
            consensus = summary.consensus or motif_template
            mismatch_rate = summary.mismatch_rate
            total_copies_bases = summary.copies * summary.motif_len
            indel_rate = ((summary.total_insertions + summary.total_deletions) / total_copies_bases
                          if total_copies_bases > 0 else 0.0)
            percent_indels = indel_rate * 100.0
            max_errors = summary.max_errors_per_copy
            variations_out = summary.variations if summary.variations else None

        total_length = len(actual_sequence)
        motif_len_effective = len(consensus) if consensus else motif_len
        copies_float = float(copies_int)
        if total_length > 0 and motif_len_effective > 0:
            fractional = total_length / motif_len_effective
            # Keep integer if close, otherwise preserve fractional component
            if abs(fractional - round(fractional)) < 1e-6:
                copies_float = float(round(fractional))
            else:
                copies_float = fractional

        percent_matches = max(0.0, 100.0 - mismatch_rate * 100.0)
        score = MotifUtils.calculate_trf_score(consensus, max(1, copies_int), mismatch_rate, total_length)
        composition = MotifUtils.calculate_composition(consensus)
        entropy = MotifUtils.calculate_entropy(consensus)
        _, strand = MotifUtils.get_canonical_motif_stranded(consensus)
        confidence = max(0.3, 1.0 - mismatch_rate)

        return TandemRepeat(
            chrom=chrom,
            start=start,
            end=start + total_length,
            motif=consensus,
            copies=copies_float,
            length=total_length,
            tier=tier_hint,
            confidence=confidence,
            consensus_motif=consensus,
            mismatch_rate=mismatch_rate,
            max_mismatches_per_copy=max_errors,
            n_copies_evaluated=max(1, copies_int),
            strand=strand,
            percent_matches=percent_matches,
            percent_indels=percent_indels,
            score=score,
            composition=composition,
            entropy=entropy,
            actual_sequence=actual_sequence,
            variations=variations_out
        )
    
    def _scan_repeats_simple(self, chrom: str, seq: str) -> List[TandemRepeat]:
        """Lightweight scanner for STRs allowing limited mismatches."""
        results: List[TandemRepeat] = []
        n = len(seq)
        offset = self.trim_offsets.get(chrom, 0)
        full_seq = self.full_sequences.get(chrom, seq)

        max_period = min(self.max_period, max(1, n))
        i = 0
        while i < n:
            best_repeat: Optional[Tuple[int, int, int, str]] = None  # (motif_len, copies, end_pos, motif)
            max_len = min(max_period, n - i)

            for motif_len in range(1, max_len + 1):
                if i + motif_len * self.min_copies > n:
                    break

                motif = seq[i:i + motif_len]
                if 'N' in motif:
                    continue
                if MotifUtils.calculate_entropy(motif) < self.min_entropy:
                    continue

                tolerance = max(1, int(motif_len * 0.1))
                pos = i
                copies = 0
                while pos + motif_len <= n:
                    copy_seq = seq[pos:pos + motif_len]
                    mismatches = MotifUtils.hamming_distance(copy_seq, motif)
                    if mismatches <= tolerance:
                        copies += 1
                        pos += motif_len
                    else:
                        break

                if copies >= self.min_copies:
                    primitive_len = MotifUtils.smallest_period_str(motif)
                    motif = motif[:primitive_len]
                    motif_len = primitive_len
                    pos = i + copies * motif_len
                    copies = (pos - i) // motif_len
                    pos = i + copies * motif_len
                    if copies >= self.min_copies:
                        if best_repeat is None or copies > best_repeat[1] or (
                            copies == best_repeat[1] and motif_len < best_repeat[0]
                        ):
                            best_repeat = (motif_len, copies, pos, motif)

            if best_repeat is None:
                i += 1
                continue

            motif_len, copies, end_pos, motif = best_repeat
            canonical, strand = MotifUtils.get_canonical_motif_stranded(motif)
            start_trimmed = i
            end_trimmed = i + copies * motif_len

            start = start_trimmed + offset
            end = end_trimmed + offset

            actual_sequence = full_seq[start:end] if full_seq else seq[start_trimmed:end_trimmed]
            consensus = canonical
            copies_float = float(copies)
            mismatch_rate = 0.0
            percent_matches = 100.0

            composition = MotifUtils.calculate_composition(consensus)
            entropy = MotifUtils.calculate_entropy(consensus)
            score = int(copies * motif_len * 2)

            repeat = TandemRepeat(
                chrom=chrom,
                start=start,
                end=end,
                motif=consensus,
                copies=copies_float,
                length=end - start,
                tier=1,
                confidence=1.0,
                consensus_motif=consensus,
                mismatch_rate=mismatch_rate,
                max_mismatches_per_copy=0,
                n_copies_evaluated=copies,
                strand=strand,
                percent_matches=percent_matches,
                percent_indels=0.0,
                score=score,
                composition=composition,
                entropy=entropy,
                actual_sequence=actual_sequence,
                variations=None
            )
            results.append(repeat)
            i = end_trimmed

        return results

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
                        full_seq = ''.join(current_seq)
                        self.full_sequences[current_chrom] = full_seq
                        if len(full_seq) <= 2 * self.flank_trim:
                            trimmed = full_seq
                            trim_left = 0
                        else:
                            trim_left = self.flank_trim
                            trim_right = self.flank_trim
                            trimmed = full_seq[trim_left:len(full_seq) - trim_right]
                        self.trim_offsets[current_chrom] = trim_left
                        sequences[current_chrom] = trimmed

                    current_chrom = line[1:].split()[0]  # Extract chromosome name
                    current_seq = []
                elif line:
                    current_seq.append(line.upper())

        if current_chrom:
            full_seq = ''.join(current_seq)
            self.full_sequences[current_chrom] = full_seq
            if len(full_seq) <= 2 * self.flank_trim:
                trimmed = full_seq
                trim_left = 0
            else:
                trim_left = self.flank_trim
                trim_right = self.flank_trim
                trimmed = full_seq[trim_left:len(full_seq) - trim_right]
            self.trim_offsets[current_chrom] = trim_left
            sequences[current_chrom] = trimmed

        # Store sequences for parallel processing
        self.sequences = sequences
        return sequences
    
    def build_indices(self, sequences: Dict[str, str]):
        """Build BWT and FM-index for each chromosome."""
        items = list(sequences.items())
        total_chroms = len(items)

        print(f"Building BWT indices...")
        start_time = time.time()

        for idx, (chrom, seq) in enumerate(items, 1):
            # Progress indicator
            progress_pct = (idx - 1) / total_chroms * 100
            bar_length = 40
            filled = int(bar_length * (idx - 1) / total_chroms)
            bar = '█' * filled + '░' * (bar_length - filled)

            elapsed = time.time() - start_time
            elapsed_str = f"{int(elapsed//60)}m {int(elapsed%60)}s" if elapsed >= 60 else f"{int(elapsed)}s"

            print(f"\r[{bar}] {progress_pct:.1f}% Building index for {chrom} ({len(seq):,} bp) - {elapsed_str}", end='', flush=True)

            # Add a single sentinel character at the end (must not appear elsewhere)
            seq_with_sentinel = seq + "$"

            # Build BWT core
            bwt_core = BWTCore(seq_with_sentinel, self.sa_sample_rate)
            self.bwt_cores[chrom] = bwt_core

        # Final progress update
        bar = '█' * bar_length
        elapsed = time.time() - start_time
        elapsed_str = f"{int(elapsed//60)}m {int(elapsed%60)}s" if elapsed >= 60 else f"{int(elapsed)}s"
        print(f"\r[{bar}] 100.0% BWT indices built for {total_chroms} chromosome(s) - {elapsed_str}     ")
        print()
    
    def find_tandem_repeats(self, enable_tier1: bool = True, enable_tier2: bool = True,
                          enable_tier3: bool = False, long_reads: Optional[List[str]] = None) -> List[TandemRepeat]:
        """Sequential repeat discovery using the tiered pipeline (no multiprocessing)."""
        all_repeats: List[TandemRepeat] = []
        items = list(self.sequences.items())
        total = len(items)

        for idx, (chrom, seq) in enumerate(items, 1):
            progress_pct = (idx - 1) / total * 100 if total else 100.0
            bar_length = 40
            filled = int(bar_length * (idx - 1) / total) if total else bar_length
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"\n[{bar}] {progress_pct:.1f}% ({idx}/{total})")
            print(f"Scanning chromosome {chrom} ({len(seq):,} bp)...")

            config = {
                'sa_sample_rate': self.sa_sample_rate,
                'enable_tier1': enable_tier1,
                'enable_tier2': enable_tier2,
                'allow_mismatches': self.allow_mismatches,
                'max_motif_length': self.max_motif_length,
                'min_period': self.min_period,
                'max_period': self.max_period,
                'min_copies': self.min_copies,
                'min_entropy': self.min_entropy,
                'show_progress': self.show_progress,
                'max_unit_len': self.max_unit_len
            }
            repeats = _process_chromosome_worker((chrom, seq, config))
            all_repeats.extend(repeats)
            print(f"  Detected {len(repeats)} STR blocks")

        bar = '█' * 40
        print(f"\n[{bar}] 100.0% ({total}/{total})")

        # STEP 1: Suppress nested short calls (prioritize long units like 36-mers)
        # Use strict threshold to only suppress truly nested calls, not adjacent ones
        filtered_repeats = self._suppress_nested_short_calls(all_repeats, overlap_threshold=0.5)
        if len(filtered_repeats) < len(all_repeats):
            print(f"Nested call suppression: {len(all_repeats)} -> {len(filtered_repeats)} repeats")

        # STEP 2: Standard post-processing
        deduped_repeats = self._deduplicate_repeats(filtered_repeats)
        merged_repeats = self._merge_adjacent_repeats(deduped_repeats)
        refined_repeats = self._refine_repeats(merged_repeats)
        self._restore_reference_coordinates(refined_repeats)
        refined_repeats = self._collapse_overlapping_repeats(refined_repeats)

        # Final filter: remove repeats that don't meet minimum criteria
        filtered_repeats = [
            r for r in refined_repeats
            if r.copies >= self.min_copies and r.length >= 6  # At least 3 copies and 6bp total
        ]
        filtered_repeats.sort(key=self._repeat_sort_key)

        print(f"Analysis complete! Found {len(filtered_repeats)} total repeats.")
        return filtered_repeats

    def find_tandem_repeats_parallel(self, enable_tier1: bool = True, enable_tier2: bool = True,
                                      enable_tier3: bool = False, long_reads: Optional[List[str]] = None,
                                      n_jobs: Optional[int] = None) -> List[TandemRepeat]:
        """
        Find tandem repeats using enabled tiers with multiprocessing.

        Args:
            enable_tier1: Enable short tandem repeat finding
            enable_tier2: Enable medium/long repeat finding
            enable_tier3: Enable very long repeat finding (not parallelized)
            long_reads: Long reads for tier 3 analysis
            n_jobs: Number of parallel jobs (None = all CPU cores)
        """
        if n_jobs is None:
            n_jobs = min(cpu_count(), len(self.sequences))

        print(f"Parallel mode: Using {n_jobs} CPU cores for {len(self.sequences)} chromosomes")
        print()

        # Prepare tasks for parallel processing
        tasks = []
        for chrom, seq in self.sequences.items():
            config = {
                'sa_sample_rate': self.sa_sample_rate,
                'enable_tier1': enable_tier1,
                'enable_tier2': enable_tier2,
                'allow_mismatches': self.allow_mismatches,
                'max_motif_length': self.max_motif_length,
                'min_period': self.min_period,
                'max_period': self.max_period,
                'min_copies': self.min_copies,
                'min_entropy': self.min_entropy,
                'show_progress': self.show_progress,
                'max_unit_len': self.max_unit_len
            }
            tasks.append((chrom, seq, config))

        # Process chromosomes in parallel with real-time progress
        all_repeats = []
        print(f"Processing {len(tasks)} chromosome(s)...")
        print()

        start_time = time.time()

        with Pool(n_jobs) as pool:
            # Use imap_unordered for real-time progress updates
            results_iter = pool.imap_unordered(_process_chromosome_worker, tasks)

            completed = 0
            for result in results_iter:
                all_repeats.extend(result)
                completed += 1

                # Update progress bar in real-time
                progress_pct = completed / len(tasks) * 100
                bar_length = 40
                filled = int(bar_length * completed / len(tasks))
                bar = '█' * filled + '░' * (bar_length - filled)

                elapsed = time.time() - start_time
                elapsed_str = f"{int(elapsed//60)}m {int(elapsed%60)}s" if elapsed >= 60 else f"{int(elapsed)}s"

                print(f"\r[{bar}] {progress_pct:.1f}% ({completed}/{len(tasks)}) chromosomes completed - {elapsed_str}", end='', flush=True)

        print()  # New line after progress bar


        # Tier 3 (not parallelized - requires long reads)
        if enable_tier3 and long_reads:
            print("\nTier 3 processing (serial)...")
            for chrom, bwt_core in self.bwt_cores.items():
                tier3 = Tier3LongReadFinder(bwt_core, show_progress=self.show_progress)
                tier3_repeats = tier3.find_very_long_repeats(long_reads, chrom)
                all_repeats.extend(tier3_repeats)
                print(f"  {chrom}: {len(tier3_repeats)} VLTRs")

        # STEP 1: Suppress nested short calls (prioritize long units like 36-mers)
        # Use strict threshold to only suppress truly nested calls, not adjacent ones
        filtered_repeats = self._suppress_nested_short_calls(all_repeats, overlap_threshold=0.5)
        if len(filtered_repeats) < len(all_repeats):
            print(f"\nNested call suppression: {len(all_repeats)} -> {len(filtered_repeats)} repeats")

        # STEP 2: Standard post-processing
        deduped_repeats = self._deduplicate_repeats(filtered_repeats)
        merged_repeats = self._merge_adjacent_repeats(deduped_repeats)
        refined_repeats = self._refine_repeats(merged_repeats)
        self._restore_reference_coordinates(refined_repeats)
        refined_repeats = self._collapse_overlapping_repeats(refined_repeats)

        # Final filter: remove repeats that don't meet minimum criteria
        filtered_repeats = [
            r for r in refined_repeats
            if r.copies >= self.min_copies and r.length >= 6  # At least 3 copies and 6bp total
        ]
        filtered_repeats.sort(key=self._repeat_sort_key)

        unique_count = len(filtered_repeats)
        duplicate_count = len(all_repeats) - len(deduped_repeats)

        if duplicate_count > 0:
            print(f"Analysis complete! Found {unique_count} unique repeats (deduplicated {duplicate_count}).")
        else:
            print(f"Analysis complete! Found {unique_count} unique repeats.")

        return filtered_repeats

    def _simple_kmer_scan(self, chrom: str, start: int, end: int, k: int = 3, use_full_seq: bool = True) -> List[TandemRepeat]:
        """Simple k-mer based scan for tandem repeats in a specific region."""
        if use_full_seq:
            seq = self.full_sequences.get(chrom, "")
        else:
            seq = self.sequences.get(chrom, "")

        if not seq or start >= end or start < 0 or end > len(seq):
            return []

        region = seq[start:end]
        results = []

        i = 0
        while i < len(region) - k:
            motif = region[i:i+k]
            # Count consecutive copies
            copies = 1
            j = i + k
            while j + k <= len(region) and region[j:j+k] == motif:
                copies += 1
                j += k

            if copies >= 5:  # Minimum 5 copies
                results.append(TandemRepeat(
                    chrom=chrom, start=start + i, end=start + j,
                    motif=motif, copies=float(copies), length=j - i,
                    tier=1, confidence=1.0, consensus_motif=motif,
                    mismatch_rate=0.0, max_mismatches_per_copy=0, n_copies_evaluated=copies,
                    strand='+', percent_matches=100.0, percent_indels=0.0,
                    score=100.0, composition={'A': 0, 'C': 0, 'G': 0, 'T': 0},
                    entropy=1.5, actual_sequence=region[i:j], variations=None
                ))
                i = j
            else:
                i += 1

        return results

    def _detect_compound_repeats(self, repeats: List[TandemRepeat]) -> List[TandemRepeat]:
        """Detect and merge adjacent repeats with different motifs into compound repeats."""
        if not repeats:
            return []

        # Group by chromosome
        by_chrom: Dict[str, List[TandemRepeat]] = {}
        for r in repeats:
            if r.chrom not in by_chrom:
                by_chrom[r.chrom] = []
            by_chrom[r.chrom].append(r)

        # Check for missing repeats using k-mer scan
        for chrom in by_chrom.keys():
            seq = self.full_sequences.get(chrom, "")  # Use full (untrimmed) sequence
            if not seq:
                continue

            # Scan for adjacent repeats that the BWT might have merged
            # Focus on regions around existing repeats
            for r in list(by_chrom[chrom]):
                # Check region after this repeat for a different 3-mer
                if r.motif and len(r.motif) == 3:
                    scan_start = r.end
                    scan_end = min(len(seq), r.end + 50)  # Look ahead 50bp
                    if scan_start < scan_end:
                        kmer_after = self._simple_kmer_scan(chrom, scan_start, scan_end, k=3)
                        for kr in kmer_after:
                            if kr.motif != r.motif:  # Different motif - potential compound
                                by_chrom[chrom].append(kr)

        result = []
        for chrom, chrom_repeats in by_chrom.items():
            chrom_repeats.sort(key=lambda r: r.start)

            # Build index of long motifs (>10bp) to check against
            long_motifs = []
            for r in chrom_repeats:
                if len(r.motif) > 10:
                    long_motifs.append((r.start, r.end, len(r.motif)))

            # Silently check for compound repeats

            i = 0
            while i < len(chrom_repeats):
                current = chrom_repeats[i]

                # Special case: check if a single 3-mer repeat might actually be a compound repeat
                # by examining the actual sequence
                if len(current.motif) == 3 and current.copies >= 10:
                    seq = self.sequences.get(current.chrom, "")
                    if seq:
                        repeat_seq = seq[current.start:current.end]
                        # Try to split into two different 3-mer repeats
                        for split_point in range(len(current.motif), len(repeat_seq) - len(current.motif), len(current.motif)):
                            motif1 = repeat_seq[:len(current.motif)]
                            motif2 = repeat_seq[split_point:split_point + len(current.motif)]

                            if motif1 != motif2:
                                # Check if first part is all motif1
                                copies1 = 0
                                for j in range(0, split_point, len(current.motif)):
                                    if repeat_seq[j:j+len(current.motif)] == motif1:
                                        copies1 += 1
                                    else:
                                        break

                                # Check if second part is all motif2
                                copies2 = 0
                                for j in range(split_point, len(repeat_seq), len(current.motif)):
                                    if repeat_seq[j:j+len(current.motif)] == motif2:
                                        copies2 += 1
                                    else:
                                        break

                                # If we found a valid split with both parts having >=5 copies
                                if copies1 >= 5 and copies2 >= 5 and (copies1 * len(motif1) + copies2 * len(motif2)) >= len(repeat_seq) * 0.9:

                                    # Create two repeats
                                    r1 = TandemRepeat(
                                        chrom=current.chrom, start=current.start, end=current.start + copies1 * len(motif1),
                                        motif=motif1, copies=float(copies1), length=copies1 * len(motif1),
                                        tier=current.tier, confidence=1.0, consensus_motif=motif1,
                                        mismatch_rate=0.0, max_mismatches_per_copy=0, n_copies_evaluated=copies1,
                                        strand='+', percent_matches=100.0, percent_indels=0.0,
                                        score=100.0, composition={'A': 0, 'C': 0, 'G': 0, 'T': 0},
                                        entropy=1.5, actual_sequence=repeat_seq[:copies1*len(motif1)], variations=None
                                    )
                                    r2 = TandemRepeat(
                                        chrom=current.chrom, start=current.start + copies1 * len(motif1), end=current.start + copies1 * len(motif1) + copies2 * len(motif2),
                                        motif=motif2, copies=float(copies2), length=copies2 * len(motif2),
                                        tier=current.tier, confidence=1.0, consensus_motif=motif2,
                                        mismatch_rate=0.0, max_mismatches_per_copy=0, n_copies_evaluated=copies2,
                                        strand='+', percent_matches=100.0, percent_indels=0.0,
                                        score=100.0, composition={'A': 0, 'C': 0, 'G': 0, 'T': 0},
                                        entropy=1.5, actual_sequence=repeat_seq[copies1*len(motif1):copies1*len(motif1)+copies2*len(motif2)], variations=None
                                    )

                                    # Mark as compound
                                    r1.is_compound = True
                                    r1.compound_partner = r2
                                    result.append(r1)
                                    i += 1
                                    continue

                # Check if next repeat is adjacent and has different motif (compound repeat)
                if i + 1 < len(chrom_repeats):
                    next_r = chrom_repeats[i + 1]
                    gap = next_r.start - current.end

                    # Adjacent repeats with small gap (<5bp) and different short motifs
                    if (gap <= 5 and
                        len(current.motif) <= 4 and len(next_r.motif) <= 4 and
                        current.motif != next_r.motif and
                        current.copies >= 5 and next_r.copies >= 5):

                        # Check if there's a longer motif covering this region
                        # If so, skip creating the compound (it's a false positive)
                        compound_start = current.start
                        compound_end = next_r.end
                        is_covered_by_long_motif = False

                        for long_start, long_end, long_len in long_motifs:
                            # Check if the compound region is mostly covered by a long motif
                            overlap_start = max(compound_start, long_start)
                            overlap_end = min(compound_end, long_end)
                            overlap = max(0, overlap_end - overlap_start)
                            compound_len = compound_end - compound_start

                            if overlap / compound_len >= 0.8:  # 80% covered by long motif
                                is_covered_by_long_motif = True
                                break

                        if not is_covered_by_long_motif:
                            # Create compound repeat marker
                            current.is_compound = True
                            current.compound_partner = next_r
                            result.append(current)
                            i += 2  # Skip the next repeat as it's now part of compound
                            continue

                result.append(current)
                i += 1

        return result

    def save_results(self, repeats: List[TandemRepeat], output_file: str, format_type: str = "bed"):
        """Save tandem repeat results to file."""
        # Detect compound repeats for strfinder format
        if format_type == "strfinder":
            repeats = self._detect_compound_repeats(repeats)

        sorted_repeats = sorted(repeats, key=self._repeat_sort_key)

        with open(output_file, 'w') as f:
            if format_type == "bed":
                f.write("# Tandem Repeats (BED format with imperfect repeat support)\n")
                f.write("# chrom\tstart\tend\tconsensus_motif\tcopies\ttier\tmismatch_rate\tstrand\n")
                for repeat in sorted_repeats:
                    f.write(repeat.to_bed() + "\n")

            elif format_type == "vcf":
                f.write("##fileformat=VCFv4.2\n")
                f.write("##INFO=<ID=MOTIF,Number=1,Type=String,Description=\"Original seed motif\">\n")
                f.write("##INFO=<ID=CONS_MOTIF,Number=1,Type=String,Description=\"Consensus motif from all copies\">\n")
                f.write("##INFO=<ID=COPIES,Number=1,Type=Float,Description=\"Number of copies\">\n")
                f.write("##INFO=<ID=TIER,Number=1,Type=Integer,Description=\"Detection tier (1=short, 2=medium/long, 3=very long)\">\n")
                f.write("##INFO=<ID=CONF,Number=1,Type=Float,Description=\"Confidence score\">\n")
                f.write("##INFO=<ID=MM_RATE,Number=1,Type=Float,Description=\"Overall mismatch rate across all copies\">\n")
                f.write("##INFO=<ID=MAX_MM_PER_COPY,Number=1,Type=Integer,Description=\"Maximum mismatches in any single copy\">\n")
                f.write("##INFO=<ID=N_COPIES_EVAL,Number=1,Type=Integer,Description=\"Number of copies evaluated for consensus\">\n")
                f.write("##INFO=<ID=STRAND,Number=1,Type=String,Description=\"Strand of canonical motif (+/-)\">\n")
                f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

                for i, repeat in enumerate(sorted_repeats):
                    f.write(f"{repeat.chrom}\t{repeat.start + 1}\tTR{i}\t.\t<TR>\t.\tPASS\t{repeat.to_vcf_info()}\n")

            elif format_type == "trf_table":
                f.write("# Tandem Repeats Finder Compatible Table Format\n")
                f.write("# Indices\tPeriod\tCopyNumber\tConsensusSize\tPercentMatches\tPercentIndels\t")
                f.write("Score\tA\tC\tG\tT\tEntropy\n")
                for repeat in sorted_repeats:
                    f.write(repeat.to_trf_table() + "\n")

            elif format_type == "trf_dat":
                # TRF DAT format: no header, space-delimited
                for repeat in sorted_repeats:
                    f.write(repeat.to_trf_dat() + "\n")

            elif format_type == "strfinder":
                # STRfinder-compatible CSV format: tab-delimited with headers
                f.write("STR_marker\tSTR_position\tSTR_motif\tSTR_genotype_structure\tSTR_genotype\t")
                f.write("STR_core_seq\tAllele_coverage\tAlleles_ratio\tReads_Distribution(consensused)\t")
                f.write("STR_depth\tFull_seq\tVariations\n")
                for idx, repeat in enumerate(sorted_repeats):
                    marker_name = f"STR_{repeat.chrom}"

                    # Extract flanking sequences (30bp each side)
                    full_sequence = self.full_sequences.get(repeat.chrom, "")
                    flank_size = 30
                    flank_left = full_sequence[max(0, repeat.start - flank_size):repeat.start] if full_sequence else ""
                    flank_right = full_sequence[repeat.end:repeat.end + flank_size] if full_sequence else ""

                    f.write(repeat.to_strfinder(marker_name, flank_left, flank_right) + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Advanced BWT-based Tandem Repeat Finder with Imperfect Repeat Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find all repeats (exact matches only)
  python bwt.py reference.fa --no-mismatches

  # Find imperfect repeats with default settings (recommended)
  python bwt.py reference.fa -o repeats.vcf --format vcf

  # Focus on short tandem repeats only
  python bwt.py reference.fa --tier1 --max-motif-len 10

  # Find medium/long repeats with custom period range
  python bwt.py reference.fa --tier2 --min-period 20 --max-period 500
        """
    )
    parser.add_argument("reference", help="Reference genome FASTA file")
    parser.add_argument("-o", "--output", default="repeat.tab", help="Output file (default: repeat.tab)")
    parser.add_argument("--format", choices=["bed", "vcf", "trf_table", "trf_dat", "strfinder"],
                       default="strfinder",
                       help="Output format (default: strfinder)")
    parser.add_argument("--tier1", action="store_true", help="Enable tier 1 only (short repeats, 1-9bp)")
    parser.add_argument("--tier3", action="store_true", help="Enable tier 3 (very long repeats, kb+)")
    parser.add_argument("--long-reads", help="Long reads file for tier 3")
    parser.add_argument("--sa-sample", type=int, default=32, help="Suffix array sampling rate (default: 32)")
    parser.add_argument("--progress", action="store_true", help="Show progress bars where applicable")
    parser.add_argument("--jobs", type=int, default=4, help="Number of parallel CPU cores (default: 4, 0=use all CPUs, -1=disable parallelism)")

    # Imperfect repeat options
    parser.add_argument("--no-mismatches", action="store_true",
                       help="Disable mismatch tolerance (exact matches only)")
    parser.add_argument("--max-motif-len", type=int, default=9,
                       help="Maximum motif length for tier 1 (default: 9)")
    parser.add_argument("--min-period", type=int, default=10,
                       help="Minimum period for tier 2 (default: 10)")
    parser.add_argument("--max-period", type=int, default=1000,
                       help="Maximum period for tier 2 (default: 1000)")
    parser.add_argument("--max-unit-len", type=int, default=120,
                       help="Maximum unit length for tier 2 long repeat detection (default: 120)")
    parser.add_argument("--min-copies", type=int, default=3,
                       help="Minimum number of copies required (default: 3)")
    parser.add_argument("--min-entropy", type=float, default=1.0,
                       help="Minimum Shannon entropy to avoid low-complexity (default: 1.0)")
    parser.add_argument("--flank-trim", type=int, default=30,
                       help="Trim N bp from each end before analysis (default: 30, use 0 to disable)")

    args = parser.parse_args()

    # Determine mismatch tolerance
    allow_mismatches = not args.no_mismatches

    # Determine tiers
    if args.tier1:
        tiers_enabled = "Tier 1 (short repeats)"
        tier1, tier2 = True, False
    else:
        tiers_enabled = "Tier 1 + Tier 2 (short + medium repeats)"
        tier1, tier2 = True, True
    if args.tier3:
        tiers_enabled += " + Tier 3 (very long repeats)"

    # Determine parallelism
    if args.jobs == 0:
        parallel_info = f"all {cpu_count()} CPU cores"
    elif args.jobs == -1:
        parallel_info = "disabled (sequential)"
    else:
        parallel_info = f"{args.jobs} CPU cores"

    print(f"BWT-based Tandem Repeat Finder")
    print(f"{'=' * 60}")
    print(f"Reference:    {args.reference}")
    print(f"Output:       {args.output} ({args.format} format)")
    print(f"Tiers:        {tiers_enabled}")
    print(f"Parallelism:  {parallel_info}")
    print(f"")
    print(f"Detection Parameters:")
    print(f"  Tier 1 motif length: 1-{args.max_motif_len} bp")
    if tier2:
        print(f"  Tier 2 period range: {args.min_period}-{args.max_period} bp")
    print(f"  Min copies required: {args.min_copies}")
    print(f"  Min entropy (bits):  {args.min_entropy}")
    print(f"  Mismatch tolerance:  {'Enabled (10% of full sequence)' if allow_mismatches else 'Disabled (exact matches only)'}")
    print(f"  SA sampling rate:    {args.sa_sample}")
    print(f"  Flank trimming:      {args.flank_trim} bp from each end")
    print()

    # Initialize finder
    finder = TandemRepeatFinder(
        args.reference,
        args.sa_sample,
        show_progress=args.progress,
        allow_mismatches=allow_mismatches,
        max_motif_length=args.max_motif_len,
        min_period=args.min_period,
        max_period=args.max_period,
        min_copies=args.min_copies,
        min_entropy=args.min_entropy,
        flank_trim=args.flank_trim,
        max_unit_len=args.max_unit_len
    )

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

    # Find tandem repeats (tiers already determined above for banner)
    enable_tier1 = tier1
    enable_tier2 = tier2
    enable_tier3 = args.tier3

    # Determine parallelism: 0=all CPUs, -1=sequential, >0=specific number
    if args.jobs == 0:
        n_jobs = None  # Use all CPUs
    elif args.jobs == -1:
        n_jobs = None  # Will trigger sequential mode below
    else:
        n_jobs = args.jobs

    # Choose parallel or sequential processing
    if args.jobs != -1:
        repeats = finder.find_tandem_repeats_parallel(
            enable_tier1=enable_tier1,
            enable_tier2=enable_tier2,
            enable_tier3=enable_tier3,
            long_reads=long_reads if enable_tier3 and long_reads else None,
            n_jobs=n_jobs
        )
    else:
        repeats = finder.find_tandem_repeats(
            enable_tier1=enable_tier1,
            enable_tier2=enable_tier2,
            enable_tier3=enable_tier3,
            long_reads=long_reads if enable_tier3 and long_reads else None
        )

    # Save results
    finder.save_results(repeats, args.output, args.format)

    print(f"\n{'=' * 60}")
    print(f"Completed! Found {len(repeats)} total tandem repeats.")
    if allow_mismatches and repeats:
        avg_mm_rate = sum(r.mismatch_rate for r in repeats) / len(repeats)
        print(f"Average mismatch rate: {avg_mm_rate:.3f}")
        with_mismatches = sum(1 for r in repeats if r.mismatch_rate > 0)
        print(f"Imperfect repeats: {with_mismatches} ({100*with_mismatches/len(repeats):.1f}%)")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
