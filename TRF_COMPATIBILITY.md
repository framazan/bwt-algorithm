# TRF (Tandem Repeats Finder) Compatibility

## Overview

This BWT-based tandem repeat finder now includes **TRF-compatible output formats**, making it easy to integrate with existing TRF-based pipelines and tools.

## TRF Output Formats

### 1. TRF Table Format (`--format trf_table`)

Tab-delimited table matching TRF's summary table format.

**Columns:**
1. **Indices** - Location in format `start--end`
2. **Period** - Length of the consensus motif
3. **CopyNumber** - Number of tandem copies
4. **ConsensusSize** - Size of consensus pattern (same as Period)
5. **PercentMatches** - Percent matches between copies (0-100)
6. **PercentIndels** - Percent indels (always 0 for Hamming distance)
7. **Score** - Alignment score (TRF-style: +2 per match, -7 per mismatch)
8. **A** - Percent adenine composition
9. **C** - Percent cytosine composition
10. **G** - Percent guanine composition
11. **T** - Percent thymine composition
12. **Entropy** - Shannon entropy (0-2 bits)

**Example:**
```
# Indices    Period  CopyNumber  ConsensusSize  PercentMatches  PercentIndels  Score  A   C   G   T   Entropy
1000--1036   4       9.0         4              97              0              68     25  25  25  25  2.00
5230--5310   9       8.9         9              95              0              152    22  28  28  22  1.98
```

### 2. TRF DAT Format (`--format trf_dat`)

Space-delimited data file with no header, suitable for automated processing.

**Fields (in order):**
```
Start End Period CopyNumber ConsensusSize PercentMatches PercentIndels Score A C G T Entropy ConsensusPattern Sequence
```

**Example:**
```
1000 1036 4 9.0 4 97 0 68 25 25 25 25 2.00 ATCG ATCGATCGATCGATCGATCGATCGATCGATCGATCG
5230 5310 9 8.9 9 95 0 152 22 28 28 22 1.98 ATGATCATC ATGATCATCATGATCATCATGATCATCATGATCATC...
```

## Usage Examples

### Generate TRF Table Format
```bash
python bwt.py reference.fa -o repeats.trf --format trf_table
```

### Generate TRF DAT Format
```bash
python bwt.py reference.fa -o repeats.dat --format trf_dat
```

### Compare with TRF
```bash
# Run TRF (original)
trf reference.fa 2 7 7 80 10 50 500 -d -h

# Run BWT-based finder with TRF output
python bwt.py reference.fa -o bwt_repeats.dat --format trf_dat

# Compare results
diff <(sort reference.fa.2.7.7.80.10.50.500.dat) <(sort bwt_repeats.dat)
```

## Field Mappings

### TRF vs. BWT Implementation

| TRF Field | BWT Equivalent | Notes |
|-----------|----------------|-------|
| Start | start | 0-based in BWT, 1-based in TRF (adjusted in output) |
| End | end | Exclusive in BWT, inclusive in TRF |
| Period | len(consensus_motif) | Consensus motif length |
| Copy Number | copies | Calculated from array length / period |
| Consensus Size | len(consensus_motif) | Same as Period |
| Percent Matches | 100 - (mismatch_rate × 100) | Derived from Hamming distance |
| Percent Indels | 0 | Hamming distance doesn't count indels |
| Score | TRF-style score | +2 per match, -7 per mismatch |
| A, C, G, T | composition | Nucleotide percentages of consensus |
| Entropy | entropy | Shannon entropy (0-2 bits) |
| Consensus Pattern | consensus_motif | Majority-vote consensus |
| Sequence | actual_sequence | Actual genomic sequence of the repeat |

## Differences from TRF

### Advantages
1. **Faster**: BWT/FM-index is asymptotically faster than dynamic programming
2. **Memory-efficient**: Sampled suffix arrays reduce memory footprint
3. **Strand-aware**: Reports canonical motif across both strands
4. **Modern output**: Also supports BED and VCF formats

### Limitations
1. **No indels**: Uses Hamming distance only (SNPs, no insertions/deletions)
   - PercentIndels is always 0
   - May miss some complex variants
2. **Score approximation**: TRF uses Smith-Waterman alignment; we approximate with simple match/mismatch scoring
3. **Different sensitivity**: May find different repeats due to algorithmic differences

## Scoring System

### TRF-Style Scoring
We approximate TRF's alignment scoring:

```python
matches = total_bases × (1 - mismatch_rate)
mismatches = total_bases × mismatch_rate

score = (matches × 2) - (mismatches × 7)
score = max(0, score)  # No negative scores
```

**Example:**
- 40 bp repeat, 5% mismatch rate
- Matches: 40 × 0.95 = 38
- Mismatches: 40 × 0.05 = 2
- Score: (38 × 2) - (2 × 7) = 76 - 14 = 62

### Score Interpretation
- **High scores (>100)**: Perfect or near-perfect repeats
- **Medium scores (50-100)**: Good quality repeats with few mismatches
- **Low scores (<50)**: More divergent repeats
- **Zero score**: Very high mismatch rate

## Entropy Calculation

Shannon entropy measures sequence complexity:

```python
entropy = -∑ p(base) × log₂(p(base))
```

**Range:** 0-2 bits per base
- **0.0**: Homopolymer (AAAA)
- **1.0**: Dinucleotide repeat (ATAT)
- **1.5**: Trinucleotide with bias
- **2.0**: Equal distribution (ACGT)

## Composition Calculation

Percentage of each nucleotide in the consensus motif:

```python
A% = (count_A / motif_length) × 100
C% = (count_C / motif_length) × 100
G% = (count_G / motif_length) × 100
T% = (count_T / motif_length) × 100
```

**Properties:**
- A + C + G + T = 100%
- Integer percentages in output (rounded)
- Based on consensus motif, not actual copies

## Integration with TRF Pipelines

### Parsing TRF DAT Files

**Python example:**
```python
def parse_trf_dat(filename):
    repeats = []
    with open(filename) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            repeat = {
                'start': int(parts[0]),
                'end': int(parts[1]),
                'period': int(parts[2]),
                'copies': float(parts[3]),
                'consensus_size': int(parts[4]),
                'percent_matches': float(parts[5]),
                'percent_indels': float(parts[6]),
                'score': int(parts[7]),
                'A': float(parts[8]),
                'C': float(parts[9]),
                'G': float(parts[10]),
                'T': float(parts[11]),
                'entropy': float(parts[12]),
                'consensus': parts[13],
                'sequence': parts[14] if len(parts) > 14 else ''
            }
            repeats.append(repeat)
    return repeats
```

### Converting Between Formats

**TRF DAT to BED:**
```python
def trf_to_bed(trf_file, bed_file, chrom='chr1'):
    repeats = parse_trf_dat(trf_file)
    with open(bed_file, 'w') as out:
        for r in repeats:
            out.write(f"{chrom}\t{r['start']}\t{r['end']}\t{r['consensus']}\t"
                     f"{r['copies']}\t{r['score']}\n")
```

### Filtering TRF Output

**High-quality repeats only:**
```bash
# Filter by score >= 50 and copies >= 3
awk '$8 >= 50 && $4 >= 3' repeats.dat > high_quality.dat

# Filter by entropy >= 1.5 (avoid low-complexity)
awk '$13 >= 1.5' repeats.dat > high_entropy.dat

# Filter by percent matches >= 90%
awk '$6 >= 90' repeats.dat > high_match.dat
```

## Validation

### Comparing with TRF

To validate our TRF-compatible output:

1. **Run both tools:**
```bash
# TRF
trf test.fa 2 7 7 80 10 50 500 -d

# BWT-based (our tool)
python bwt.py test.fa --format trf_dat -o bwt.dat
```

2. **Compare statistics:**
```bash
# Count repeats
wc -l test.fa.2.7.7.80.10.50.500.dat bwt.dat

# Compare score distributions
awk '{print $8}' test.fa.2.7.7.80.10.50.500.dat | sort -n > trf_scores.txt
awk '{print $8}' bwt.dat | sort -n > bwt_scores.txt
```

3. **Analyze differences:**
```python
import pandas as pd

trf_repeats = parse_trf_dat('test.fa.2.7.7.80.10.50.500.dat')
bwt_repeats = parse_trf_dat('bwt.dat')

print(f"TRF found: {len(trf_repeats)} repeats")
print(f"BWT found: {len(bwt_repeats)} repeats")

# Compare score distributions
trf_scores = [r['score'] for r in trf_repeats]
bwt_scores = [r['score'] for r in bwt_repeats]

print(f"TRF mean score: {np.mean(trf_scores):.1f}")
print(f"BWT mean score: {np.mean(bwt_scores):.1f}")
```

## Best Practices

### When to Use TRF Formats

✅ **Use TRF formats when:**
- Integrating with existing TRF-based pipelines
- Need compatibility with TRF parsers
- Want simple tab/space-delimited output
- Comparing with original TRF results

❌ **Use BED/VCF formats when:**
- Need genome browser visualization (BED)
- Integrating with variant calling pipelines (VCF)
- Want detailed mismatch statistics (VCF INFO fields)
- Need chromosome-aware formats

### Quality Filtering

Recommended filters for high-quality repeats:

```bash
# TRF table format
awk 'NR==1 || ($5 >= 90 && $8 >= 50 && $4 >= 3 && $12 >= 1.0)' repeats.trf

# Explanation:
# - PercentMatches >= 90 (column 5)
# - Score >= 50 (column 8)
# - CopyNumber >= 3 (column 4)
# - Entropy >= 1.0 (column 12)
```

### Performance Optimization

For large genomes with TRF output:

```bash
# Process one chromosome at a time
for chr in chr{1..22} chrX chrY; do
    samtools faidx genome.fa $chr | \
    python bwt.py /dev/stdin --format trf_dat -o ${chr}.dat
done

# Concatenate results
cat chr*.dat > genome_all.dat
```

## Example Workflows

### Workflow 1: Find STRs and Export to TRF Format

```bash
# Find short tandem repeats (1-6bp motifs)
python bwt.py genome.fa --tier1 --max-motif-len 6 \
    --format trf_table -o strs.trf

# Filter for high-quality STRs
awk 'NR==1 || ($5 >= 95 && $4 >= 5)' strs.trf > strs_high_qual.trf

# Count by period size
tail -n +2 strs_high_qual.trf | awk '{print $2}' | sort | uniq -c
```

### Workflow 2: Compare with TRF for Validation

```bash
# Run both tools
trf test.fa 2 7 7 80 10 50 500 -d -h
python bwt.py test.fa --format trf_dat -o bwt.dat --min-copies 2

# Extract common repeats (by position)
awk '{print $1"-"$2}' test.fa.2.7.7.80.10.50.500.dat | sort > trf_pos.txt
awk '{print $1"-"$2}' bwt.dat | sort > bwt_pos.txt

# Find overlaps
comm -12 trf_pos.txt bwt_pos.txt | wc -l
```

### Workflow 3: Convert to Other Formats

```bash
# TRF DAT → BED
python3 << EOF
with open('repeats.dat') as f:
    for line in f:
        parts = line.strip().split()
        print(f"chr1\t{parts[0]}\t{parts[1]}\t{parts[13]}\t{parts[7]}\t+")
EOF > repeats.bed

# TRF TABLE → CSV
sed 's/\t/,/g' repeats.trf > repeats.csv
```

## Troubleshooting

### Issue: Scores don't match TRF exactly

**Cause:** Different alignment algorithms (Smith-Waterman vs. Hamming distance)

**Solution:** Our scores are approximations. Use as relative quality metrics, not absolute comparisons.

### Issue: Different number of repeats found

**Cause:**
- We use minimum 3 copies by default (TRF often uses 2)
- We filter by entropy (TRF doesn't)
- Different sensitivity to imperfect repeats

**Solution:** Adjust parameters:
```bash
python bwt.py genome.fa --min-copies 2 --min-entropy 0.0 --format trf_dat
```

### Issue: PercentIndels always 0

**Cause:** We use Hamming distance (SNPs only), not edit distance

**Solution:** This is by design. Use mismatch rate instead for quality assessment.

## FAQ

**Q: Can I use this as a drop-in replacement for TRF?**
A: Mostly yes for SNP-tolerant repeats. No for indel-rich repeats.

**Q: Which format is most similar to TRF's HTML tables?**
A: Use `--format trf_table`. It matches TRF's summary table format.

**Q: How do I get the actual repeat sequences like TRF's alignment files?**
A: Use `--format trf_dat`. The last field contains the actual sequence.

**Q: Why are entropy values different from TRF?**
A: Both use Shannon entropy, but we calculate on the consensus motif only, while TRF may use the entire array.

**Q: Can I process the output with TRF's downstream tools?**
A: Yes, for tools that parse the DAT format. HTML-specific tools won't work.

## References

- [TRF Homepage](https://tandem.bu.edu/trf/trf.html)
- [TRF GitHub](https://github.com/Benson-Genomics-Lab/TRF)
- [TRF Paper](https://doi.org/10.1093/nar/27.2.573) - Benson, G. (1999). Nucleic Acids Research, 27(2), 573-580.

## Version Compatibility

- **TRF Version:** Compatible with TRF 4.09 output format
- **Our Version:** Implements TRF DAT and table formats as of 2025-01-XX

---

**Last updated:** 2025-01-XX
