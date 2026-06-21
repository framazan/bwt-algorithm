# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BWT-based tandem repeat finder for genomic sequences. Detects tandem repeats (STRs and longer repeats) in FASTA files using a Burrows-Wheeler Transform / FM-index approach with a 3-tier detection pipeline. Written in Python with optional Cython/Numba acceleration.

## Running the Tool

```bash
# Basic usage
python3 -m src.main <fasta_file> [options]

# Common options
python3 -m src.main input.fa --min-period 1 --max-period 2000 --tiers tier1,tier2,tier3 --format bed -o output_prefix -v
python3 -m src.main input.fa --tiers tier1 --format trf --profile  # profile tier execution
python3 -m src.main input.fa -t 4 --mask soft -v  # 4 threads, skip soft-masked regions

# Output formats: bed (default), vcf, trf (.dat), strfinder (.csv)
```

## Building Cython Extensions

The Cython extension `_accelerators.pyx` provides critical performance functions. Without it, pure-Python fallbacks in `accelerators.py` are used (some return empty results, effectively disabling those code paths).

```bash
# Compile Cython extensions (requires numpy, Cython)
python3 -c "
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
ext_modules = [Extension('src._accelerators', ['src/_accelerators.pyx'], include_dirs=[np.get_include()])]
setup(script_args=['build_ext', '--inplace'], ext_modules=cythonize(ext_modules, compiler_directives={'language_level': '3'}))
"
```

Rebuild scope: only edits to `_accelerators.pyx` require recompiling the `.so`.
Edits to the pure-Python tier files (`tier1.py`, `tier2.py`, `tier3.py`,
`finder.py`, `bwt_seed.py`, etc.) take effect immediately with no rebuild.

## Tuning detection sensitivity (env vars)

The tier detection thresholds are hardcoded constants exposed as environment-
variable overrides so the sensitivity/precision trade-off can be swept without
editing code. **All defaults reproduce the original behaviour exactly** — unset
means baseline. Set them on the command line, e.g.
`TIER1_MIN_ARRAY_LEN=20 TIER1_MIN_SCORE=20 python3 -m src.main ...`.

- **Tier 1** (`tier1.py`, short STRs): `TIER1_MIN_COPIES`, `TIER1_MIN_ARRAY_LEN`
  (min reported span, default 26), `TIER1_MIN_SCORE` (length×purity gate,
  default 30), `TIER1_COPYBASE`/`TIER1_COPYADD` (the `dynamic_min_copies =
  max(min_copies, COPYBASE//motif_len + COPYADD)` formula), `TIER1_EXT_COPIES`
  (perfect seed copies required before mismatch extension), `TIER1_MISMATCH`,
  `TIER1_ENTROPY_GATE`/`TIER1_MIN_ENTROPY` (opt-in low-complexity filter, default
  off — empirically it lowers recall without raising precision on short STRs).
- **Tier 2** (`tier2.py`, period 10-20 simple scan): `TIER2_MIN_COPIES`,
  `TIER2_MIN_ARRAY_LEN`, `TIER2_SHORT_REQ_COPIES` (copies required for periods
  <20 bp), `TIER2_MISMATCH`.

The dominant short-STR recall levers are `TIER1_MIN_ARRAY_LEN` + `TIER1_MIN_SCORE`
(copy-count knobs have little effect); lowering them raises recall but drops
precision. See `docs/2026-06-20-exp1-human-sensitivity.md` for the measured
recall/precision frontier and the chosen operating point.

## Dependencies

- **Required**: numpy, pydivsufsort (fast suffix array construction; falls back to NumPy prefix-doubling if unavailable)
- **Optional performance**: numba (JIT for rank queries and LCP), Cython (compiled `_accelerators`)
- **Container**: Singularity definition file at repo root builds a complete environment

## Architecture

### 3-Tier Detection Pipeline (`finder.py` — `TandemRepeatFinder`)

The coordinator builds a `BWTCore` FM-index once per chromosome, then runs enabled tiers sequentially. Each tier receives regions already found by previous tiers to avoid redundant work.

- **Tier 1** (`tier1.py` — `Tier1STRFinder`): Short perfect repeats, motifs 1–9 bp. For sequences <10 Mbp uses FM-index backward search to enumerate all canonical motifs and locate tandem runs. For larger sequences falls back to a sliding-window scanner with adaptive step size. Requires ≥3 copies.

- **Tier 2** (`tier2.py` — `Tier2LCPFinder`): Medium/imperfect repeats, motifs ≥10 bp. Two sub-phases:
  - **Long-unit strict**: Uses LCP array (Kasai's algorithm) to find adjacent suffix pairs with period ≥20 bp, then extends with mismatch tolerance.
  - **General scanning**: BWT k-mer seeding (`bwt_seed.py`) for periods 10–50 bp, detecting periodic runs in FM-index occurrence positions and extending candidates.

- **Tier 3** (`tier3.py` — `Tier3LongReadFinder`): Long repeats, periods 100 bp–100 kbp. Uses BWT k-mer seeding with large k-mers (20 bp) and sparse sampling (stride=100). Ultra-long arrays (>100 copies or >10 kb) use anchor-based boundary verification instead of full DP refinement.

### Core Modules

- **`bwt_core.py` — `BWTCore`**: FM-index construction (suffix array via pydivsufsort, BWT, checkpointed occurrence arrays, sampled SA). Provides `backward_search()`, `count_occurrences()`, `locate_positions()`, and 8-mer hash for O(1) short k-mer lookup.

- **`bwt_seed.py`**: Shared BWT k-mer seeding for Tier 2 and Tier 3. Samples k-mers at configurable stride, finds all occurrences via FM-index, detects arithmetic progressions (periodic runs) in positions, extends with mismatch tolerance.

- **`motif_utils.py` — `MotifUtils`**: Canonical motif rotation (strand-aware), primitive period detection (exact and approximate), DP alignment of repeat copies (`align_repeat_region` with banded Smith-Waterman), consensus building, TRF-compatible statistics, and the `refine_repeat()` entry point used by all tiers.

- **`accelerators.py` / `_accelerators.pyx`**: Cython-accelerated hot paths (hamming distance, mismatch extension, unit repeat scanning, LCP candidate detection, periodic run finding, DP alignment). The Python module transparently falls back to pure-Python stubs (some no-ops) when the `.so` is absent.

- **`models.py`**: Data classes — `TandemRepeat` (output record with BED/VCF/TRF/STRfinder formatters), `AlignmentResult`, `RepeatAlignmentSummary`, `RefinedRepeat`.

### Post-processing (`finder.py`)

After all tiers run: merge adjacent repeats with same canonical motif (gap ≤ max(10, motif_len)), filter overlapping repeats keeping the higher-scoring one (score = length × (1 − mismatch_rate)), and apply user-specified array length bounds.

### Key Design Decisions

- Coordinates are 0-based internally; output converts to 1-based for VCF and STRfinder formats.
- Motif canonicalization considers both strands (forward + reverse complement rotations); the lexicographically smallest rotation is canonical.
- `refine_repeat()` always reduces to the primitive period (e.g., ATAT → AT) using both exact and approximate (≤2% error) periodicity tests.
- The sentinel `$` is appended to sequences for BWT construction and excluded from repeat detection.

## Test Data

`arabadopsis_chrs/` contains Arabidopsis chromosome FASTAs and small test sequences (`test_seq1.fa` through `test_seq5.fa`) for development.

## Utility Scripts

- `scripts/mutate_fasta.py`: Introduce random point mutations into a FASTA file (for testing mismatch tolerance). Creates `.bak` backup.
