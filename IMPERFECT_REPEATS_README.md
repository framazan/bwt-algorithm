# Imperfect Repeat Support for BWT-based Tandem Repeat Finder

## Overview

This implementation extends the BWT-based tandem repeat finder to handle **imperfect repeats** (repeats with SNPs/mismatches) while maintaining the original exact-match capability. The enhancement follows a three-layer approach:

1. **Seeding**: Find candidate repeats without missing those with mismatches
2. **Extension**: Score and extend tandem arrays when copies are imperfect
3. **Reporting**: Report variability metrics (consensus motif, mismatch rate) for downstream tools

## Key Features

### 1. Similarity Model
- Uses **Hamming distance** for SNP detection (no indels)
- Per-copy mismatch tolerance: `r ≤ max(1, ⌈0.1 × motif_length⌉)`
- Adjustable mismatch rate (default: 10% per copy)

### 2. Seeding Strategy (Tier 1)
- **Seed-and-extend approach**: Uses exact FM-index hits as seeds
- Extends left/right from each seed allowing mismatches per copy
- Filters low-complexity sequences using Shannon entropy (default: ≥1.0 bits)
- Requires minimum copies (default: 3) and minimum array length (default: 12bp)

### 3. Consensus Motif Building
- Uses **majority vote** per position across all copies
- Incrementally updates consensus during extension
- Tracks mismatch statistics per copy

### 4. Strand-Agnostic Canonicalization
- Computes reverse complement for each motif
- Selects lexicographically smallest rotation across both strands
- Reports strand information (+/-)

### 5. Quality Filtering
- **Entropy filter**: Skips homopolymers and low-complexity sequences
- **Copy count filter**: Requires ≥3 copies by default
- **Array length filter**: Minimum 12bp for Tier 1, 20bp for Tier 2
- **Maximality check**: Ensures repeats cannot be extended further

## New Output Fields

### BED Format
```
chrom  start  end  consensus_motif  copies  tier  mismatch_rate  strand
```

### VCF Format (INFO fields)
- `MOTIF`: Original seed motif
- `CONS_MOTIF`: Consensus motif from all copies
- `COPIES`: Number of tandem copies
- `TIER`: Detection tier (1=short, 2=medium/long, 3=very long)
- `CONF`: Confidence score (0.0-1.0)
- `MM_RATE`: Overall mismatch rate across all copies
- `MAX_MM_PER_COPY`: Maximum mismatches in any single copy
- `N_COPIES_EVAL`: Number of copies used for consensus
- `STRAND`: Strand of canonical motif (+/-)

### TRF-Compatible Formats

**TRF Table Format** (`--format trf_table`): 12 columns
- Indices, Period, CopyNumber, ConsensusSize, PercentMatches, PercentIndels
- Score, A, C, G, T, Entropy

**TRF DAT Format** (`--format trf_dat`): 14+ fields
- Start, End, Period, CopyNumber, ConsensusSize, PercentMatches, PercentIndels
- Score, A%, C%, G%, T%, Entropy, ConsensusPattern, Sequence

See [TRF_COMPATIBILITY.md](TRF_COMPATIBILITY.md) for details.

### STRfinder-Compatible Format

**STRfinder CSV Format** (`--format strfinder`): 11 columns (tab-delimited)
1. **STR_marker**: Marker name or auto-generated ID (`STR_chr_pos`)
2. **STR_position**: Genomic position (`chr:start-end`)
3. **STR_motif**: Motif structure (`[MOTIF]n`)
4. **STR_genotype_structure**: Genotype with motif (`[MOTIF]copies`)
5. **STR_genotype**: Repeat numbers (haploid)
6. **STR_core_seq**: Core STR sequence
7. **Allele_coverage**: Confidence percentage
8. **Alleles_ratio**: `-` (haploid only)
9. **Reads_Distribution**: Copy number distribution (`copies:count`)
10. **STR_depth**: Total depth (equals copies for haploid)
11. **Full_seq**: 5' flanking + CORE + 3' flanking

See [STRFINDER_FORMAT.md](STRFINDER_FORMAT.md) for details and performance optimizations.

## Command-Line Options

### Basic Usage
```bash
# Find imperfect repeats (default mode)
python bwt.py reference.fa -o repeats.bed

# Output in VCF format with detailed INFO fields
python bwt.py reference.fa -o repeats.vcf --format vcf

# Exact matches only (original behavior)
python bwt.py reference.fa --no-mismatches
```

### Advanced Options
```bash
# Tier 1 only with custom motif length
python bwt.py reference.fa --tier1 --max-motif-len 10

# Tier 2 with custom period range
python bwt.py reference.fa --tier2 --min-period 20 --max-period 500

# Adjust quality filters
python bwt.py reference.fa --min-copies 5 --min-entropy 1.5

# Both tiers with custom settings
python bwt.py reference.fa --tier1 --tier2 \
    --max-motif-len 8 \
    --min-period 15 \
    --max-period 200 \
    --min-copies 4
```

### All Options
| Option | Default | Description |
|--------|---------|-------------|
| `--no-mismatches` | False | Disable mismatch tolerance (exact matches only) |
| `--max-motif-len` | 6 | Maximum motif length for Tier 1 |
| `--min-period` | 10 | Minimum period for Tier 2 |
| `--max-period` | 1000 | Maximum period for Tier 2 |
| `--min-copies` | 3 | Minimum number of copies required |
| `--min-entropy` | 1.0 | Minimum Shannon entropy (bits per base) |
| `--sa-sample` | 32 | Suffix array sampling rate |
| `--progress` | False | Show progress information |

## Implementation Details

### Tier 1: Short Tandem Repeats (1-10bp)

**Algorithm:**
1. Enumerate all canonical primitive motifs of length k
2. Filter by Shannon entropy (≥1.0 bits)
3. Use FM-index to find exact occurrences (seeds)
4. For each seed position:
   - Extend left/right using Hamming distance
   - Build consensus motif via majority vote
   - Stop when distance > threshold
5. Check maximality and apply filters

**Key Functions:**
- `_find_tandems_with_mismatches()`: Seed-and-extend strategy
- `_extend_tandem_array()`: Bidirectional extension with consensus update
- `_is_maximal_repeat_approx()`: Maximality check with mismatch tolerance

### Tier 2: Medium/Long Repeats (10-1000bp)

**Algorithm:**
1. Scan genome with candidate periods p ∈ [min_period, max_period]
2. For each position, extract motif of length p
3. Filter by entropy and character validity (no '$' or 'N')
4. Extend with mismatch tolerance:
   - Compare copies to evolving consensus
   - Accept if Hamming distance ≤ threshold
5. Normalize to primitive period
6. Build final consensus and compute statistics

**Key Functions:**
- `_find_repeats_simple()`: Main scanning loop with mismatch support
- `_extend_with_mismatches()`: Bidirectional extension for longer periods
- `_get_max_mismatches_per_copy()`: Dynamic threshold based on motif length

### Consensus Building Algorithm

```python
def build_consensus(copies, motif_len):
    consensus = [None] * motif_len
    for pos in range(motif_len):
        bases = [copy[pos] for copy in copies]
        # Majority vote
        consensus[pos] = most_common(bases)
    return consensus
```

**Mismatch Rate Calculation:**
```python
mm_rate = total_mismatches / (n_copies × motif_len)
```

## Performance Considerations

### Memory
- Sampling rates reduce memory: SA sampled every 32 positions, Occ every 128
- Per-chromosome processing with explicit memory cleanup
- Consensus building uses O(motif_len × n_copies) space

### Speed
- FM-index provides O(|pattern|) backward search
- Hamming distance computed via NumPy vectorization
- Seen-regions tracking prevents redundant processing
- Early termination when entropy/copy filters fail

### False Positives
Mitigated by:
1. Minimum entropy threshold (1.0 bits) → filters homopolymers
2. Minimum copy count (3) → reduces noise
3. Minimum array length (12-20bp) → avoids spurious short runs
4. Maximality check → prevents suboptimal calls

## Example Output

### BED Format
```
# chrom  start  end    consensus_motif  copies  tier  mismatch_rate  strand
chr1    1000   1036   ATCG             9.0     1     0.028          +
chr1    5230   5310   ATGATCATC        8.9     2     0.042          -
```

### VCF Format
```
##fileformat=VCFv4.2
##INFO=<ID=MOTIF,Number=1,Type=String,Description="Original seed motif">
##INFO=<ID=CONS_MOTIF,Number=1,Type=String,Description="Consensus motif from all copies">
...
#CHROM  POS   ID   REF  ALT   QUAL  FILTER  INFO
chr1    1001  TR0  .    <TR>  .     PASS    MOTIF=ATCG;CONS_MOTIF=ATCG;COPIES=9.0;TIER=1;CONF=0.97;MM_RATE=0.028;MAX_MM_PER_COPY=2;N_COPIES_EVAL=9;STRAND=+
```

## Testing

### Small Test Case
```bash
# Create test FASTA
echo ">test" > test.fa
echo "ATCGATCGATCGATCGATCG" >> test.fa  # Perfect repeat
echo "ATCGATCGATGGGTCGATCG" >> test.fa  # Imperfect repeat (2 mismatches in one copy)

# Run finder
python bwt.py test.fa -o test.bed --tier1 --max-motif-len 4
```

**Expected:** Finds ATCG repeat with ~5 copies and mismatch_rate ≈ 0.05

## Design Decisions

### Why Hamming Distance (not edit distance)?
- Simpler position arithmetic (no indels to track)
- Faster computation (vectorized NumPy comparison)
- SNPs are the dominant variant type in STRs for many applications

### Why Majority Vote (not alignment)?
- Linear time: O(motif_len × n_copies)
- Robust to outliers (1-2 divergent copies)
- No need for expensive MSA algorithms

### Why Entropy Filter?
- Prevents false positives from homopolymers (AAAA...)
- Threshold of 1.0 bits allows dinucleotide repeats (AT: entropy ≈1.0)
- Configurable via `--min-entropy`

### Why Require ≥3 Copies?
- 2-copy arrays are often spurious or borderline
- 3+ copies provide statistical confidence
- Reduces output size by ~30-50%

## Limitations

1. **No Indels**: Current implementation uses Hamming distance only
   - Extension to edit distance is possible but complicates position tracking

2. **Fixed Mismatch Rate**: 10% per copy is hardcoded
   - Could be made configurable with `--max-mismatch-rate` option

3. **Seed Dependency**: Requires at least one exact seed occurrence
   - Very divergent repeats (>20% mismatches) may be missed
   - Bitap/Shift-Or k-mismatch search could improve sensitivity

4. **Consensus Ambiguity**: Uses simple majority vote
   - Ties resolved arbitrarily (first in NumPy unique order)
   - Could emit IUPAC ambiguity codes for low-confidence positions

## Future Enhancements

### Near-Term
- [ ] Add `--max-mismatch-rate` CLI option for per-copy tolerance
- [ ] Emit IUPAC consensus motifs for ambiguous positions
- [ ] Implement Bitap fuzzy matching for Tier 1 (alternative to seed-and-extend)
- [ ] Support for gzipped FASTA inputs

### Long-Term
- [ ] Edit distance support (Levenshtein) for indel tolerance
- [ ] Multi-seed strategy with overlapping k-mers
- [ ] Phasing support for diploid genomes
- [ ] Integration with variant calling pipelines (BCFtools, GATK)

## References

- FM-index: Ferragina & Manzini (2000)
- Kasai LCP algorithm: Kasai et al. (2001)
- Tandem Repeat Finder: Benson (1999)
- Shannon entropy for sequence complexity: Wootton & Federhen (1996)

## Citation

If you use this tool in your research, please cite:
```
BWT-based Tandem Repeat Finder with Imperfect Repeat Support
[Your citation information here]
```

## License

[Your license information here]
