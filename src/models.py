from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import math

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
    is_compound: bool = False  # Whether this is part of a compound repeat
    compound_partner: Optional['TandemRepeat'] = None  # The other part of the compound repeat

    def to_bed(self) -> str:
        """Convert to BED format."""
        # BED motif column should expose the primitive unit, not an expanded consensus block.
        return f"{self.chrom}\t{self.start}\t{self.end}\t{self.motif}\t{self.copies:.1f}\t{self.tier}\t{self.mismatch_rate:.3f}\t{self.strand}"

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
        is_compound = self.is_compound and self.compound_partner is not None

        if is_compound:
            partner = self.compound_partner
            assert partner is not None  # for type checker
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

            return (f"{marker},{position},{str_motif},{genotype_struct},{genotype},"
                    f"{core_seq},{allele_coverage},{alleles_ratio},{reads_dist},"
                    f"{str_depth},{full_seq},{variation_str}")

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

        # Allele_coverage - percentage. percent_matches is a field defaulting to 0.0,
        # so it is always present; the old confidence-based branch was unreachable.
        allele_coverage = f"{self.percent_matches:.0f}%"

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

        return (f"{marker},{position},{str_motif},{genotype_struct},{genotype},"
            f"{core_seq},{allele_coverage},{alleles_ratio},{reads_dist},"
            f"{str_depth},{full_seq},{variation_str}")


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
    total_mismatches: int


@dataclass
class RefinedRepeat:
    """Alignment-refined repeat details prior to TandemRepeat construction."""
    start: int
    end: int
    consensus: str
    primitive_motif: str
    motif_len: int
    copies: float
    summary: RepeatAlignmentSummary
