#!/usr/bin/env python3
"""
Refactored test/driver for the BWT-based Tandem Repeat Finder.

Given a FASTA file, iterate over all sequences, run the repeat finder,
and write per-sequence result files. Keeps the synthetic test function.
"""

import argparse
import os
import sys
from typing import List, Optional

from bwt import TandemRepeatFinder
from bwt import progress_iter


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def process_fasta(
    fasta_path: str,
    output_dir: str = ".",
    output_format: str = "bed",
    sa_sample_rate: int = 16,
    enable_tier1: bool = True,
    enable_tier2: bool = False,
    enable_tier3: bool = False,
    long_reads: Optional[List[str]] = None,
    vcf_top: Optional[int] = 20,
    show_progress: bool = False,
):
    """Process a FASTA: run finder per sequence and write per-sequence outputs."""

    if not os.path.isfile(fasta_path):
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")

    ensure_dir(output_dir)

    print(f"Processing FASTA: {fasta_path}")
    print("=" * 60)

    # We'll load sequences first to get names and lengths, then process one-by-one
    loader = TandemRepeatFinder(fasta_path, sa_sample_rate, show_progress=show_progress)
    sequences = loader.load_reference()

    base = os.path.splitext(os.path.basename(fasta_path))[0]

    items = list(sequences.items())
    for chrom, seq in progress_iter(items, total=len(items), desc="Sequences", enable=show_progress):
        print(f"\n=== Running tandem repeat finder for {chrom} ({len(seq):,} bp) ===")

        # Build index only for this sequence to keep memory lower
        finder = TandemRepeatFinder(fasta_path, sa_sample_rate, show_progress=show_progress)
        finder.build_indices({chrom: seq})

        repeats = finder.find_tandem_repeats(
            enable_tier1=enable_tier1,
            enable_tier2=enable_tier2,
            enable_tier3=enable_tier3,
            long_reads=long_reads if enable_tier3 else None,
        )

        print(f"Found {len(repeats)} tandem repeats in {chrom}")

        # Save per-sequence outputs
        bed_path = os.path.join(output_dir, f"{base}__{chrom}.bed")
        vcf_path = os.path.join(output_dir, f"{base}__{chrom}.vcf")

        if output_format in ("bed", "both"):
            finder.save_results(repeats, bed_path, "bed")
            print(f"Saved BED to {bed_path}")

        if output_format in ("vcf", "both"):
            # Optionally limit VCF to top N by length for readability
            to_write = repeats
            if vcf_top is not None and vcf_top > 0:
                to_write = sorted(repeats, key=lambda r: r.length, reverse=True)[:vcf_top]
            finder.save_results(to_write, vcf_path, "vcf")
            print(f"Saved VCF to {vcf_path}")

def create_synthetic_test():
    """Create a synthetic test sequence with known tandem repeats."""
    
    print("Creating synthetic test sequence...")
    
    # Create a test sequence with known tandem repeats
    test_sequence = (
        "AAAAAAAAAA" +           # Homopolymer run (A)
        "ATATATATATATATAT" +     # 2bp repeat (AT) x 8
        "GGGGGGGGGG" +           # Homopolymer run (G)  
        "CACACACACACACACACA" +   # 2bp repeat (CA) x 9
        "TTTTTTTTTT" +           # Homopolymer run (T)
        "AGCAGCAGCAGCAGCAGC" +   # 3bp repeat (AGC) x 6
        "CCCCCCCCCC" +           # Homopolymer run (C)
        "AGTCAGTCAGTCAGTCAGTC" + # 4bp repeat (AGTC) x 5
        "NNNNNNNNNN" +           # Spacer with N's
        "ATGCATGCATGCATGCATGC" + # 4bp repeat (ATGC) x 5
        "$"                      # Sentinel
    )
    
    # Write to file
    with open("synthetic_test.fa", "w") as f:
        f.write(">synthetic_chromosome\n")
        f.write(test_sequence + "\n")
    
    print(f"Created synthetic test sequence ({len(test_sequence)} bp)")
    
    # Test on synthetic sequence
    finder = TandemRepeatFinder("synthetic_test.fa", sa_sample_rate=4)
    
    try:
        sequences = finder.load_reference()
        finder.build_indices(sequences)
        
        # Test all tiers on small synthetic sequence
        repeats = finder.find_tandem_repeats(
            enable_tier1=True,
            enable_tier2=True,
            enable_tier3=False  # No long reads for synthetic test
        )
        
        print(f"\nFound {len(repeats)} tandem repeats in synthetic sequence!")
        
        if repeats:
            print("\nAll tandem repeats found:")
            print("Start\tEnd\tMotif\tCopies\tLength\tTier")
            print("-" * 50)
            
            for repeat in sorted(repeats, key=lambda x: x.start):
                print(f"{repeat.start}\t{repeat.end}\t{repeat.motif}\t{repeat.copies:.1f}\t{repeat.length}\t{repeat.tier}")
        
        # Save synthetic results
        finder.save_results(repeats, "synthetic_results.bed", "bed")
        print(f"\nSynthetic results saved to synthetic_results.bed")
        
    except Exception as e:
        print(f"Error during synthetic testing: {e}")
        import traceback
        traceback.print_exc()

def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description=(
            "Run tandem repeat finding per sequence in a FASTA and write per-sequence results."
        )
    )

    subparsers = parser.add_subparsers(dest="command")

    # Main command: process a fasta
    run_parser = subparsers.add_parser("run", help="Process a FASTA file")
    run_parser.add_argument("fasta", help="Path to FASTA file")
    run_parser.add_argument("--output-dir", "-o", default="results", help="Directory for outputs")
    run_parser.add_argument(
        "--format",
        choices=["bed", "vcf", "both"],
        default="bed",
        help="Output format per sequence",
    )
    run_parser.add_argument("--sa-sample", type=int, default=16, help="Suffix array sampling rate")
    run_parser.add_argument("--tier1", action="store_true", default=True, help="Enable Tier 1 (short repeats)")
    run_parser.add_argument("--no-tier1", dest="tier1", action="store_false", help="Disable Tier 1")
    run_parser.add_argument("--tier2", action="store_true", default=False, help="Enable Tier 2 (medium/long repeats)")
    run_parser.add_argument("--tier3", action="store_true", default=False, help="Enable Tier 3 (very long repeats)")
    run_parser.add_argument("--long-reads", help="Optional long reads FASTA/FASTQ for Tier 3")
    run_parser.add_argument("--vcf-top", type=int, default=20, help="Limit VCF to top N by length (<=0 for all)")
    run_parser.add_argument("--progress", action="store_true", help="Show progress bars where applicable")

    # Synthetic example
    subparsers.add_parser("synthetic", help="Run built-in synthetic test")

    args = parser.parse_args()

    if args.command == "synthetic":
        create_synthetic_test()
        return

    if args.command != "run":
        parser.print_help()
        return

    # Optional: load long reads for Tier 3
    long_reads = None
    if args.tier3 and args.long_reads:
        long_reads = []
        seq = ""
        with open(args.long_reads, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">") or line.startswith("@"):  # FASTA/FASTQ header
                    if seq:
                        long_reads.append(seq)
                        seq = ""
                elif not line.startswith("+"):  # Skip FASTQ separator
                    seq += line.upper()
        if seq:
            long_reads.append(seq)

    process_fasta(
        fasta_path=args.fasta,
        output_dir=args.output_dir,
        output_format=args.format,
        sa_sample_rate=args.sa_sample,
        enable_tier1=args.tier1,
        enable_tier2=args.tier2,
        enable_tier3=args.tier3,
        long_reads=long_reads,
        vcf_top=(args.vcf_top if args.vcf_top > 0 else None),
        show_progress=args.progress,
    )

if __name__ == "__main__":
    main()