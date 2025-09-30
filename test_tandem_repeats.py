#!/usr/bin/env python3
"""
Test script for the BWT-based Tandem Repeat Finder.
Demonstrates usage on the Arabidopsis thaliana genome.
"""

import os
import sys
from bwt import TandemRepeatFinder

def test_on_small_chromosome(specific_chr=None):
    """Test on a small chromosome file for quick validation."""
    
    # Check if we have extracted chromosome files
    chr_files = ["Chr1.fa", "Chr2.fa", "Chr3.fa", "Chr4.fa", "Chr5.fa", "ChrC.fa", "ChrM.fa"]
    chr_dir = "arabadopsis_chrs"
    
    available_chrs = []
    for chr_file in chr_files:
        chr_path = os.path.join(chr_dir, chr_file)
        if os.path.exists(chr_path):
            available_chrs.append(chr_path)
    
    if not available_chrs:
        print("No chromosome files found in arabadopsis_chrs/")
        print("Please run chromosomes_extract.py first to extract individual chromosomes.")
        return
    
    # Show available chromosomes
    print("Available chromosomes:")
    for i, chr_path in enumerate(available_chrs):
        size = os.path.getsize(chr_path) / (1024 * 1024)  # MB
        print(f"  {i+1}. {os.path.basename(chr_path)} ({size:.1f} MB)")
    print()
    
    # Select chromosome to test
    test_file = None
    if specific_chr:
        # Try to find the specified chromosome
        for chr_path in available_chrs:
            if specific_chr.lower() in os.path.basename(chr_path).lower():
                test_file = chr_path
                break
        if not test_file:
            print(f"Chromosome '{specific_chr}' not found. Available: {[os.path.basename(p) for p in available_chrs]}")
            return
    else:
        # Default: prefer smallest chromosomes (organellar genomes)
        for chr_path in available_chrs:
            if "ChrM.fa" in chr_path or "ChrC.fa" in chr_path:  # Organellar genomes are smaller
                test_file = chr_path
                break
        
        if not test_file:
            test_file = available_chrs[0]  # Use first available
    
    print(f"Testing tandem repeat finder on {test_file}")
    print("=" * 60)
    
    # Initialize the tandem repeat finder
    finder = TandemRepeatFinder(test_file, sa_sample_rate=16)  # Smaller sampling for demo
    
    try:
        # Load reference and build indices
        sequences = finder.load_reference()
        finder.build_indices(sequences)
        
        print("\nRunning Tier 1 (Short Tandem Repeats)...")
        # Test Tier 1 only for quick demo (Tier 2 can be slow on large sequences)
        repeats = finder.find_tandem_repeats(
            enable_tier1=True,
            enable_tier2=False,  # Disable for demo speed
            enable_tier3=False
        )
        
        print(f"\nFound {len(repeats)} tandem repeats!")
        
        # Show some examples
        if repeats:
            print("\nTop 10 tandem repeats found:")
            print("Chrom\tStart\tEnd\tMotif\tCopies\tLength\tTier")
            print("-" * 60)
            
            for i, repeat in enumerate(sorted(repeats, key=lambda x: x.length, reverse=True)[:10]):
                print(f"{repeat.chrom}\t{repeat.start}\t{repeat.end}\t{repeat.motif}\t{repeat.copies:.1f}\t{repeat.length}\t{repeat.tier}")
        
        # Save results
        output_file = f"test_results_{os.path.basename(test_file)}.bed"
        finder.save_results(repeats, output_file, "bed")
        print(f"\nResults saved to {output_file}")
        
        # Also create a small VCF example
        vcf_file = f"test_results_{os.path.basename(test_file)}.vcf"
        finder.save_results(repeats[:20], vcf_file, "vcf")  # Just first 20 for VCF demo
        print(f"VCF example saved to {vcf_file}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

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
    """Main test function."""
    print("BWT-based Tandem Repeat Finder - Test Script")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "synthetic":
            create_synthetic_test()
        elif sys.argv[1] == "help" or sys.argv[1] == "-h":
            print("Usage:")
            print("  python test_tandem_repeats.py                    # Test on default small chromosome")
            print("  python test_tandem_repeats.py synthetic          # Test on synthetic sequence")
            print("  python test_tandem_repeats.py <chromosome>       # Test on specific chromosome")
            print("\nAvailable chromosomes: Chr1, Chr2, Chr3, Chr4, Chr5, ChrC, ChrM")
            print("Examples:")
            print("  python test_tandem_repeats.py ChrM              # Test on mitochondrial genome")
            print("  python test_tandem_repeats.py ChrC              # Test on chloroplast genome")
            print("  python test_tandem_repeats.py Chr1              # Test on chromosome 1")
        else:
            # Test on specific chromosome
            test_on_small_chromosome(sys.argv[1])
    else:
        test_on_small_chromosome()

if __name__ == "__main__":
    main()