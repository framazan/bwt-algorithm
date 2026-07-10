import argparse  # Standard library for command-line argument parsing
import sys  # Standard library for sys.exit and standard output
import time  # Standard library for measuring execution time
import os  # Standard library for file path handling and existence checks
import re  # Standard library for regular expressions
from typing import List, Iterator, Tuple  # Type hints from the typing module
from concurrent.futures import ProcessPoolExecutor, as_completed  # Module for multiprocessing
from .finder import TandemRepeatFinder  # Multi-tier tandem repeat finding coordinator
from .models import TandemRepeat  # Tandem repeat data class and output formatters


def apply_mask(seq: str, mask_mode: str) -> str:
    """Apply masking to a sequence based on the mask mode.

    - none: convert everything to uppercase (default behavior)
    - soft: replace lowercase bases (soft-masked) with N
    - hard: already-N regions remain N (no extra action needed beyond uppercase)
    - both: replace lowercase with N, keep existing Ns
    """
    if mask_mode == "none":
        return seq.upper()

    result = []
    for ch in seq:
        if ch in 'acgt':
            # Soft-masked base
            if mask_mode in ("soft", "both"):
                result.append('N')
            else:
                result.append(ch.upper())
        elif ch == 'N' or ch == 'n':
            result.append('N')
        else:
            result.append(ch.upper())
    return ''.join(result)


def _process_chromosome(chrom: str, seq: str, min_period: int, max_period: int,
                        enabled_tiers: set, show_progress: bool,
                        min_array_bp, max_array_bp, tier3_mode: str) -> List[TandemRepeat]:
    """Process a single chromosome — designed to be called in parallel."""
    finder = TandemRepeatFinder(
        seq,
        chromosome=chrom,
        min_period=min_period,
        max_period=max_period,
        show_progress=show_progress,
        enabled_tiers=enabled_tiers,
        min_array_bp=min_array_bp,
        max_array_bp=max_array_bp,
        tier3_mode=tier3_mode,
    )
    repeats = finder.find_all()
    finder.cleanup()
    return repeats


def _resolve_output_file(output_prefix: str, extension: str) -> str:
    """Return output path while avoiding duplicated extensions."""
    ext = extension if extension.startswith(".") else f".{extension}"  # Prepend dot if extension lacks one
    if output_prefix.lower().endswith(ext.lower()):
        return output_prefix  # Already includes the extension, return as-is
    return f"{output_prefix}{ext}"  # Append extension to prefix to form final file path

def parse_fasta(file_path: str) -> Iterator[Tuple[str, str]]:
    """Simple FASTA parser to avoid Biopython dependency."""
    name = None  # Initialize current sequence name
    seq_parts = []  # List to accumulate per-line fragments of the current sequence
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()  # Remove leading/trailing whitespace and newline characters
            if not line:
                continue  # Skip empty lines
            if line.startswith('>'):
                # Process header line starting with '>'
                if name:
                    yield name, "".join(seq_parts)  # Yield the completed previous sequence
                name = line[1:].split()[0]  # Take first word as ID
                seq_parts = []  # Initialize fragment list for the new sequence
            else:
                seq_parts.append(line)  # Append sequence line to fragment list
        if name:
            yield name, "".join(seq_parts)  # Yield the last sequence in the file

def main():
    parser = argparse.ArgumentParser(description="BWT-based Tandem Repeat Finder")  # Create CLI parser
    parser.add_argument("fasta_file", help="Input FASTA file")  # Input FASTA file path argument
    parser.add_argument("--min-period", type=int, default=1, help="Minimum period size (default: 1)")  # Minimum period length option
    parser.add_argument("--max-period", type=int, default=2000, help="Maximum period size (default: 2000)")  # Maximum period length option
    parser.add_argument("--min-array-bp", type=int, default=None,
                        help="Minimum repeat array length in bp (default: no minimum)")  # Minimum repeat array length option
    parser.add_argument("--max-array-bp", type=int, default=None,
                        help="Maximum repeat array length in bp (default: no maximum)")  # Maximum repeat array length option
    parser.add_argument("--tiers", type=str, default="tier1,tier2,tier3",
                        help="Comma-separated list of tiers to run (tier1,tier2,tier3) or 'all'")  # Tiers to run option
    parser.add_argument("--output", "-o", help="Output file prefix (default: input filename)")  # Output file prefix option
    parser.add_argument("--format", choices=["bed", "vcf", "trf", "strfinder"], default="bed", help="Output format")  # Output format selection option
    parser.add_argument("--verbose", "-v", action="store_true", help="Show progress")  # Verbose output flag
    parser.add_argument("--profile", action="store_true", help="Profile execution with cProfile and print top hotspots")  # Performance profiling flag
    parser.add_argument("--tier3-mode", choices=["fast", "balanced", "sensitive"],
                        default="balanced", help="Tier 3 speed/accuracy preset (default: balanced)")  # Tier 3 speed/accuracy preset option
    parser.add_argument("--threads", "-t", type=int, default=1,
                        help="Number of threads for parallel chromosome processing (default: 1)")  # Parallel processing thread count option
    parser.add_argument("--mask", choices=["none", "soft", "hard", "both"], default="none",
                        help="Masking mode: none=ignore masks, soft=skip lowercase regions, "
                             "hard=skip N regions, both=skip both (default: none)")  # Masking mode option

    args = parser.parse_args()  # Parse command-line arguments

    if not os.path.exists(args.fasta_file):
        # Print error message and exit if input file does not exist
        print(f"Error: File {args.fasta_file} not found")
        sys.exit(1)

    output_prefix = args.output if args.output else os.path.splitext(args.fasta_file)[0]  # Output prefix: use input filename (without extension) if not specified

    print(f"Processing {args.fasta_file}...")  # Notify processing start
    start_total = time.time()  # Record overall processing start time

    all_repeats: List[TandemRepeat] = []  # List to collect repeat results from all chromosomes

    tiers_arg = args.tiers.strip()  # Strip whitespace from tiers argument
    if tiers_arg.lower() == "all":
        enabled_tiers = {"tier1", "tier2", "tier3"}  # Enable all tiers if "all"
    else:
        # Normalize comma-separated tier names to lowercase and create a set
        enabled_tiers = {t.strip().lower() for t in tiers_arg.split(',') if t.strip()}

    # Optional profiler
    profiler = None  # Initialize profiler (default None)
    if args.profile:
        import cProfile  # Dynamic import of cProfile for performance profiling
        profiler = cProfile.Profile()  # Create profiler instance
        profiler.enable()  # Start profiling

    # Load input sequences and apply masking
    sequences = []
    for chrom, seq in parse_fasta(args.fasta_file):
        seq = apply_mask(seq, args.mask)  # Apply masking based on the selected mode
        if args.verbose:
            n_count = seq.count('N')
            masked_pct = n_count / len(seq) * 100 if len(seq) > 0 else 0
            mask_info = f", {n_count} N ({masked_pct:.1f}% masked)" if args.mask != "none" and n_count > 0 else ""
            print(f"Processing sequence: {chrom} ({len(seq)} bp{mask_info})")
        sequences.append((chrom, seq))

    n_threads = max(1, args.threads)

    if n_threads == 1 or len(sequences) == 1:
        # Single-thread mode: sequential processing
        for chrom, seq in sequences:
            repeats = _process_chromosome(
                chrom, seq, args.min_period, args.max_period,
                enabled_tiers, args.verbose,
                args.min_array_bp, args.max_array_bp, args.tier3_mode
            )
            all_repeats.extend(repeats)
    else:
        # Multi-thread mode: parallel processing per chromosome
        if args.verbose:
            print(f"Using {n_threads} parallel processes for {len(sequences)} sequences")
        with ProcessPoolExecutor(max_workers=n_threads) as executor:
            futures = {}
            for chrom, seq in sequences:
                future = executor.submit(
                    _process_chromosome,
                    chrom, seq, args.min_period, args.max_period,
                    enabled_tiers, args.verbose,
                    args.min_array_bp, args.max_array_bp, args.tier3_mode
                )
                futures[future] = chrom

            for future in as_completed(futures):
                chrom = futures[future]
                try:
                    repeats = future.result()
                    all_repeats.extend(repeats)
                    if args.verbose:
                        print(f"  [{chrom}] Completed: {len(repeats)} repeats found")
                except Exception as e:
                    print(f"  [{chrom}] ERROR: {e}", file=sys.stderr)

    # Stop profiler and report
    if profiler is not None:
        profiler.disable()  # Stop profiling

    print(f"Total repeats found: {len(all_repeats)}")  # Print total number of repeats found
    print(f"Total time: {time.time() - start_total:.2f}s")  # Print total processing time

    if profiler is not None:
        import pstats  # Dynamic import of pstats for profile statistics output
        profile_path = f"{output_prefix}.profile.prof"  # Profile result file path (covers all tiers, not just Tier 2)
        profiler.dump_stats(profile_path)  # Save profile data to file
        print(f"Profile written to {profile_path}")  # Print save path
        print("Top 20 cumulative time hotspots:")  # Heading for top 20 cumulative time hotspots
        stats = pstats.Stats(profiler)  # Create profile statistics object
        stats.strip_dirs().sort_stats("cumulative").print_stats(20)  # Strip directories, sort by cumulative time, print top 20

    # Write output
    if args.format == "bed":
        out_file = _resolve_output_file(output_prefix, "bed")  # Determine BED format output file path
        with open(out_file, "w") as f:
            for r in all_repeats:
                f.write(r.to_bed() + "\n")  # Write each repeat in BED format
    elif args.format == "vcf":
        out_file = _resolve_output_file(output_prefix, "vcf")  # Determine VCF format output file path
        with open(out_file, "w") as f:
            f.write("##fileformat=VCFv4.2\n")  # Write VCF file format header
            f.write("##INFO=<ID=END,Number=1,Type=Integer,Description=\"End position of the repeat\">\n")  # INFO field definition header
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")  # Write VCF column header
            for r in all_repeats:
                # Use a single anchor base for REF to keep symbolic-ALT records valid.
                # Use a single anchor base for REF to keep symbolic-ALT records valid
                ref = "N"  # Default REF value (when no sequence info available)
                if r.actual_sequence:
                    ref = r.actual_sequence[0]  # Use first base of actual sequence as REF
                elif r.consensus_motif:
                    ref = r.consensus_motif[0]  # Use first base of consensus motif as REF
                elif r.motif:
                    ref = r.motif[0]  # Use first base of motif as REF
                alt = "<STR>"  # Symbolic ALT value representing a tandem repeat
                info = f"END={r.end};{r.to_vcf_info()}"  # Combine END position with VCF INFO field
                f.write(f"{r.chrom}\t{r.start+1}\t.\t{ref}\t{alt}\t.\t.\t{info}\n")  # Write VCF record in 1-based coordinates
    elif args.format == "trf":
        out_file = _resolve_output_file(output_prefix, "dat")  # Determine output file path for TRF .dat format
        with open(out_file, "w") as f:
            for r in all_repeats:
                f.write(r.to_trf_dat() + "\n")  # Write each repeat to file in TRF .dat format
    elif args.format == "strfinder":
        out_file = _resolve_output_file(output_prefix, "csv")  # Determine output file path for STRfinder CSV format
        with open(out_file, "w") as f:
            # Write STRfinder CSV header
            f.write("STR_marker,STR_position,STR_motif,STR_genotype_structure,STR_genotype,STR_core_seq,Allele_coverage,Alleles_ratio,Reads_Distribution,STR_depth,Full_seq,Variations\n")
            for r in all_repeats:
                f.write(r.to_strfinder() + "\n")  # Write each repeat to file in STRfinder format

    print(f"Results written to {out_file}")  # Print the output file path

if __name__ == "__main__":
    main()  # Call main function when script is executed directly
