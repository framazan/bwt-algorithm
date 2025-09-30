#!/usr/bin/env python3
"""
Advanced BWT-based Tandem Repeat Finder for Genomics

Implements three-tier approach:
1. Short tandem repeats (1-10bp) with FM-index
2. Medium/long repeats (10s-1000s bp) with LCP arrays  
3. Very long repeats (kb+) with long read evidence

Author: AI Assistant
"""

import numpy as np
import bisect
from collections import defaultdict, namedtuple
from typing import List, Tuple, Dict, Iterator, Optional, Set
import argparse
import sys
import os
import re
from dataclasses import dataclass
from enum import Enum


class BWTCore:
    """Core BWT construction and FM-index operations."""
    
    def __init__(self, text: str, sa_sample_rate: int = 32):
        """
        Initialize BWT with FM-index.
        
        Args:
            text: Input text (should end with unique sentinel)
            sa_sample_rate: Sample every nth suffix array position for space efficiency
        """
        self.text = text
        self.n = len(text)
        self.sa_sample_rate = sa_sample_rate
        
        # Build suffix array and BWT
        self.suffix_array = self._build_suffix_array()
        self.bwt = self._build_bwt()
        
        # Build FM-index components
        self.alphabet = sorted(set(text))
        self.char_counts = self._build_char_counts()
        self.occurrence = self._build_occurrence_table()
        
        # Sample suffix array for efficient locating
        self.sampled_sa = self._sample_suffix_array()
    
    def _build_suffix_array(self) -> np.ndarray:
        """Build suffix array using numpy for efficiency."""
        suffixes = [(self.text[i:], i) for i in range(self.n)]
        suffixes.sort()
        return np.array([suffix[1] for suffix in suffixes], dtype=np.int32)
    
    def _build_bwt(self) -> str:
        """Build BWT from suffix array."""
        bwt_chars = []
        for i in range(self.n):
            sa_pos = self.suffix_array[i]
            if sa_pos == 0:
                bwt_chars.append(self.text[-1])  # Last character (sentinel)
            else:
                bwt_chars.append(self.text[sa_pos - 1])
        return ''.join(bwt_chars)
    
    def _build_char_counts(self) -> Dict[str, int]:
        """Count character frequencies and compute cumulative counts."""
        counts = {}
        cumulative = 0
        for char in self.alphabet:
            counts[char] = cumulative
            cumulative += self.text.count(char)
        return counts
    
    def _build_occurrence_table(self) -> Dict[str, List[int]]:
        """Build occurrence table for efficient rank queries."""
        occurrence = {char: [0] for char in self.alphabet}
        
        for i, char in enumerate(self.bwt):
            for c in self.alphabet:
                occurrence[c].append(occurrence[c][-1])
            occurrence[char][-1] += 1
        
        return occurrence
    
    def _sample_suffix_array(self) -> Dict[int, int]:
        """Sample suffix array positions for space-efficient locating."""
        sampled = {}
        for i in range(0, self.n, self.sa_sample_rate):
            sampled[i] = self.suffix_array[i]
        return sampled
    
    def rank(self, char: str, pos: int) -> int:
        """Count occurrences of char in bwt[0:pos]."""
        if char not in self.occurrence:
            return 0
        return self.occurrence[char][pos]
    
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
        
        sp = self.char_counts[char]
        ep = sp + self.text.count(char) - 1
        
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
        
        positions = []
        for i in range(sp, ep + 1):
            pos = self._get_suffix_position(i)
            positions.append(pos)
        
        return sorted(positions)
    
    def _get_suffix_position(self, sa_index: int) -> int:
        """Recover original text position from SA index using sampling."""
        if sa_index in self.sampled_sa:
            return self.sampled_sa[sa_index]
        
        # Walk using LF mapping until we hit a sampled position
        steps = 0
        current_idx = sa_index
        
        while current_idx not in self.sampled_sa:
            char = self.bwt[current_idx]
            current_idx = self.char_counts[char] + self.rank(char, current_idx)
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
    
    def __init__(self, bwt_core: BWTCore, max_motif_length: int = 10):
        self.bwt = bwt_core
        self.max_motif_length = max_motif_length
        self.min_copies = 2
    
    def find_strs(self, chromosome: str) -> List[TandemRepeat]:
        """Find short tandem repeats (1-10bp motifs)."""
        repeats = []
        
        print(f"Tier 1: Scanning motifs 1-{self.max_motif_length}bp...")
        
        for k in range(1, self.max_motif_length + 1):
            print(f"  Processing {k}bp motifs...")
            motif_count = 0
            
            for motif in MotifUtils.enumerate_motifs(k):
                motif_count += 1
                if motif_count % 100 == 0:
                    print(f"    Processed {motif_count} {k}bp motifs...")
                
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
    
    def __init__(self, bwt_core: BWTCore, min_period: int = 10, max_period: int = 1000):
        self.bwt = bwt_core
        self.min_period = min_period
        self.max_period = max_period
        self.min_copies = 2
    
    def find_long_repeats(self, chromosome: str) -> List[TandemRepeat]:
        """Find medium to long tandem repeats using LCP analysis."""
        print("Tier 2: Computing LCP array...")
        lcp_array = self._compute_lcp_array()
        
        print("Tier 2: Detecting LCP plateaus...")
        repeats = self._detect_lcp_plateaus(lcp_array, chromosome)
        
        return repeats
    
    def _compute_lcp_array(self) -> np.ndarray:
        """Compute LCP array from BWT using Phi array method."""
        n = self.bwt.n
        lcp = np.zeros(n, dtype=np.int32)
        
        # Build rank array (inverse of suffix array)
        rank = np.zeros(n, dtype=np.int32)
        for i in range(n):
            rank[self.bwt.suffix_array[i]] = i
        
        # Compute LCP using Kasai algorithm
        h = 0
        for i in range(n):
            if rank[i] > 0:
                j = self.bwt.suffix_array[rank[i] - 1]
                while (i + h < n and j + h < n and 
                       self.bwt.text[i + h] == self.bwt.text[j + h]):
                    h += 1
                lcp[rank[i]] = h
                if h > 0:
                    h -= 1
        
        return lcp
    
    def _detect_lcp_plateaus(self, lcp_array: np.ndarray, chromosome: str) -> List[TandemRepeat]:
        """Detect tandem repeats from LCP plateaus."""
        repeats = []
        n = len(lcp_array)
        
        # Scan for LCP plateaus
        for threshold in range(self.min_period, min(self.max_period, np.max(lcp_array)) + 1):
            i = 0
            while i < n:
                if lcp_array[i] >= threshold:
                    # Found start of plateau
                    j = i
                    while j < n and lcp_array[j] >= threshold:
                        j += 1
                    
                    # Analyze SA interval [i, j) for tandem structure
                    tandem_repeats = self._analyze_sa_interval_for_tandems(
                        i, j, threshold, chromosome
                    )
                    repeats.extend(tandem_repeats)
                    i = j
                else:
                    i += 1
        
        return repeats
    
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
                # Extract and validate motif
                motif_start = start_pos
                motif_end = motif_start + period
                if motif_end <= len(self.bwt.text):
                    motif = self.bwt.text[motif_start:motif_end]
                    
                    # Validate periodicity
                    total_length = copies * period
                    repeat_text = self.bwt.text[start_pos:start_pos + total_length]
                    
                    if self._validate_periodicity(repeat_text, motif, period):
                        repeat = TandemRepeat(
                            chrom=chromosome,
                            start=start_pos,
                            end=start_pos + total_length,
                            motif=motif,
                            copies=copies,
                            length=total_length,
                            tier=2,
                            confidence=0.9  # High confidence from LCP
                        )
                        repeats.append(repeat)
        
        return repeats
    
    def _validate_periodicity(self, text: str, motif: str, period: int) -> bool:
        """Validate that text has the expected periodic structure."""
        if len(text) < 2 * period:
            return False
        
        # Check if text is approximately periodic with given motif
        matches = 0
        total_positions = 0
        
        for i in range(len(text)):
            motif_pos = i % period
            if motif_pos < len(motif):
                total_positions += 1
                if text[i] == motif[motif_pos]:
                    matches += 1
        
        # Allow some mismatches for imperfect repeats
        similarity = matches / total_positions if total_positions > 0 else 0
        return similarity >= 0.8


class Tier3LongReadFinder:
    """Tier 3: Very Long Tandem Repeat Finder using long read evidence."""
    
    def __init__(self, bwt_core: BWTCore):
        self.bwt = bwt_core
        self.min_read_length = 1000  # Minimum read length for analysis
        self.min_span_length = 100   # Minimum repeat length to analyze
    
    def find_very_long_repeats(self, long_reads: List[str], chromosome: str) -> List[TandemRepeat]:
        """Find very long tandem repeats using long read evidence."""
        print("Tier 3: Analyzing long reads for very long repeats...")
        repeats = []
        
        for read_idx, read in enumerate(long_reads):
            if len(read) < self.min_read_length:
                continue
            
            if read_idx % 100 == 0:
                print(f"  Processed {read_idx} long reads...")
            
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
    
    def __init__(self, reference_file: str, sa_sample_rate: int = 32):
        """
        Initialize the tandem repeat finder.
        
        Args:
            reference_file: Path to reference genome FASTA
            sa_sample_rate: Suffix array sampling rate for space efficiency
        """
        self.reference_file = reference_file
        self.sa_sample_rate = sa_sample_rate
        self.bwt_cores = {}  # Store BWT for each chromosome
    
    def load_reference(self) -> Dict[str, str]:
        """Load reference sequences from FASTA file."""
        sequences = {}
        current_chrom = None
        current_seq = []
        
        print(f"Loading reference from {self.reference_file}...")
        
        with open(self.reference_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_chrom:
                        sequences[current_chrom] = ''.join(current_seq)
                    
                    current_chrom = line[1:].split()[0]  # Extract chromosome name
                    current_seq = []
                    print(f"  Loading {current_chrom}...")
                elif line:
                    current_seq.append(line.upper())
        
        if current_chrom:
            sequences[current_chrom] = ''.join(current_seq)
        
        return sequences
    
    def build_indices(self, sequences: Dict[str, str]):
        """Build BWT and FM-index for each chromosome."""
        print("Building BWT indices...")
        
        for chrom, seq in sequences.items():
            print(f"  Building index for {chrom} ({len(seq):,} bp)...")
            
            # Add unique sentinel for this chromosome
            seq_with_sentinel = seq + f"${chrom}$"
            
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
        
        for chrom, bwt_core in self.bwt_cores.items():
            print(f"\nAnalyzing chromosome {chrom}...")
            
            if enable_tier1:
                tier1 = Tier1STRFinder(bwt_core)
                tier1_repeats = tier1.find_strs(chrom)
                all_repeats.extend(tier1_repeats)
                print(f"  Tier 1 found {len(tier1_repeats)} short tandem repeats")
            
            if enable_tier2:
                tier2 = Tier2LCPFinder(bwt_core)
                tier2_repeats = tier2.find_long_repeats(chrom)
                all_repeats.extend(tier2_repeats)
                print(f"  Tier 2 found {len(tier2_repeats)} medium/long tandem repeats")
            
            if enable_tier3 and long_reads:
                tier3 = Tier3LongReadFinder(bwt_core)
                tier3_repeats = tier3.find_very_long_repeats(long_reads, chrom)
                all_repeats.extend(tier3_repeats)
                print(f"  Tier 3 found {len(tier3_repeats)} very long tandem repeats")
        
        return all_repeats
    
    def save_results(self, repeats: List[TandemRepeat], output_file: str, format_type: str = "bed"):
        """Save tandem repeat results to file."""
        print(f"\nSaving {len(repeats)} tandem repeats to {output_file}...")
        
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
    parser.add_argument("--tier1", action="store_true", default=True, help="Enable tier 1 (short repeats)")
    parser.add_argument("--tier2", action="store_true", default=True, help="Enable tier 2 (medium/long repeats)")
    parser.add_argument("--tier3", action="store_true", help="Enable tier 3 (very long repeats)")
    parser.add_argument("--long-reads", help="Long reads file for tier 3")
    parser.add_argument("--sa-sample", type=int, default=32, help="Suffix array sampling rate")
    
    args = parser.parse_args()
    
    # Initialize finder
    finder = TandemRepeatFinder(args.reference, args.sa_sample)
    
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
    repeats = finder.find_tandem_repeats(
        enable_tier1=args.tier1,
        enable_tier2=args.tier2, 
        enable_tier3=args.tier3,
        long_reads=long_reads if args.tier3 and long_reads else None
    )
    
    # Save results
    finder.save_results(repeats, args.output, args.format)
    
    print(f"\nCompleted! Found {len(repeats)} total tandem repeats.")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()