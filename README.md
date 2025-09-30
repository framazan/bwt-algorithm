# Advanced BWT-based Tandem Repeat Finder

A sophisticated implementation of three-tier tandem repeat detection for genomics using Burrows-Wheeler Transform (BWT) and FM-index.

## Overview

This tool implements three complementary approaches for finding tandem repeats in genomic sequences:

### Tier 1: Short Tandem Repeats (1-10bp)
- Uses FM-index for fast motif enumeration and counting
- Performs backward search to locate motif occurrences
- Identifies back-to-back positioning for tandem structure
- Optimal for short sequence repeats (STRs)

### Tier 2: Medium/Long Tandem Repeats (10bp-1000bp)
- Computes LCP (Longest Common Prefix) arrays from BWT
- Detects LCP plateaus indicating repetitive structure
- Validates periodicity and extends to maximal repeats
- Finds unknown motifs and imperfect repeats

### Tier 3: Very Long Tandem Repeats (kb+)
- Analyzes long read sequences (ONT/PacBio)
- Maps reads to detect spanning sequences
- Estimates copy numbers from alignment evidence
- Handles very long and highly variable repeat arrays

## Features

- **Efficient BWT Construction**: Space-efficient suffix array sampling
- **Canonical Motif Handling**: Reduces redundant motif analysis
- **Multiple Output Formats**: BED and VCF format support
- **Configurable Detection**: Enable/disable individual tiers
- **Genomic Coordinate System**: Proper chromosome handling
- **Quality Metrics**: Confidence scoring for repeat calls

## Installation

No external dependencies required - uses only Python standard library and numpy.

```bash
# Ensure you have Python 3.7+ and numpy
pip install numpy

# Clone or download the repository
# All code is in bwt.py
```

## Usage

### Basic Usage

```bash
# Run on a single chromosome
python bwt.py Chr1.fa -o chr1_repeats.bed

# Run with all tiers enabled
python bwt.py reference.fa --tier1 --tier2 --tier3 --long-reads reads.fasta

# Output in VCF format
python bwt.py reference.fa --format vcf -o repeats.vcf
```

### Command Line Options

```
positional arguments:
  reference             Reference genome FASTA file

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output file (default: tandem_repeats.bed)
  --format {bed,vcf}    Output format (default: bed)
  --tier1               Enable tier 1 (short repeats) [default: True]
  --tier2               Enable tier 2 (medium/long repeats) [default: True]  
  --tier3               Enable tier 3 (very long repeats)
  --long-reads LONG_READS
                        Long reads file for tier 3
  --sa-sample SA_SAMPLE
                        Suffix array sampling rate (default: 32)
```

## Test Examples

### Test with Synthetic Data

```bash
# Create and test on synthetic sequence with known repeats
python test_tandem_repeats.py synthetic
```

### Test with Real Data

```bash
# First extract chromosomes from multi-FASTA
python chromosomes_extract.py

# Test on a small chromosome (mitochondrial genome)
python test_tandem_repeats.py
```

## Algorithm Details

### Tier 1 Implementation
1. **Motif Enumeration**: Generate canonical primitive motifs 1-10bp
2. **FM-Index Search**: Use backward search for O(k) counting per motif
3. **Position Analysis**: Locate occurrences and check back-to-back spacing
4. **Maximality Check**: Ensure repeats cannot be extended

### Tier 2 Implementation  
1. **LCP Construction**: Build LCP array from BWT using Kasai algorithm
2. **Plateau Detection**: Scan for intervals with high LCP values
3. **Periodicity Validation**: Check arithmetic progressions in suffix array
4. **Motif Extraction**: Extract and validate repeat motifs

### Tier 3 Implementation
1. **Read Analysis**: Process long reads for repetitive structure
2. **Autocorrelation**: Detect periodic patterns in read sequences  
3. **Reference Mapping**: Map repetitive regions to reference coordinates
4. **Copy Number Estimation**: Estimate repeat copy numbers from read evidence

## Output Formats

### BED Format
```
# chrom  start   end     motif   copies  tier
Chr1    1000    1020    AT      10.0    1
Chr1    5000    5150    AGTC    37.5    2
```

### VCF Format
```
##fileformat=VCFv4.2
##INFO=<ID=MOTIF,Number=1,Type=String,Description="Repeat motif">
##INFO=<ID=COPIES,Number=1,Type=Float,Description="Number of copies">
##INFO=<ID=TIER,Number=1,Type=Integer,Description="Detection tier">
##INFO=<ID=CONF,Number=1,Type=Float,Description="Confidence score">
#CHROM  POS     ID      REF     ALT     QUAL    FILTER  INFO
Chr1    1001    TR0     .       <TR>    .       PASS    MOTIF=AT;COPIES=10.0;TIER=1;CONF=1.00
```

## Performance Considerations

- **Memory Usage**: ~8-12x reference size for BWT and indices
- **Suffix Array Sampling**: Higher values reduce memory but slow locating
- **Tier Selection**: Enable only needed tiers for optimal performance
- **Chromosome Processing**: Processes each chromosome independently

## Example Workflows

### STR Analysis (Tier 1 Only)
```bash
# Fast STR detection for population studies
python bwt.py genome.fa --tier1 --sa-sample 16 -o strs.bed
```

### Comprehensive Analysis (Tiers 1+2)
```bash  
# Complete short and medium repeat analysis
python bwt.py genome.fa --tier1 --tier2 --sa-sample 32 -o all_repeats.bed
```

### Long Read Integration (All Tiers)
```bash
# Full analysis with long read evidence
python bwt.py genome.fa --tier1 --tier2 --tier3 --long-reads pacbio.fasta -o comprehensive.vcf --format vcf
```

## Implementation Notes

- Uses numpy for efficient array operations
- Implements space-efficient BWT construction
- Includes canonical motif reduction to avoid redundancy
- Provides configurable confidence thresholds
- Supports both perfect and imperfect repeat detection

## Future Enhancements

- GPU acceleration for large genomes
- Distributed processing across multiple chromosomes
- Advanced long read alignment integration
- Real-time streaming analysis for very large datasets
- Machine learning-based repeat classification

## Citation

This implementation is based on the three-tier approach described in the research prompt, combining:
- FM-index based short repeat detection
- LCP array analysis for medium repeats  
- Long read evidence for large repeat arrays