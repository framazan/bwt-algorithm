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

The Cython extension `_accelerators.pyx` provides the performance-critical hot paths. Without it, the pure-Python fallbacks in `accelerators.py` take over: they are **faithful** — the same repeats are detected, just far more slowly — and `tests/test_accel_parity.py` pins the two paths to byte-identical BED/TRF output. Build the extension for any real run. Set `BWT_DISABLE_NATIVE=1` to force the pure-Python path (used by the parity test; also handy for isolating a suspected accelerator bug).

The one exception is `TIER1_EXT_ROLLING=1`, an experimental rolling-consensus extender that exists only in the `.pyx`; requesting it without the compiled extension raises rather than silently running the default algorithm.

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

Rebuild scope: only edits to `_accelerators.pyx` require running the command
above by hand. The four `src/c_extensions/*.c` libraries rebuild themselves on
import when their source is newer than their `.so` (`build.py` checks mtimes), so
editing `align_accel.c` and friends needs no manual step. Edits to the
pure-Python tier files (`tier1.py`, `tier2.py`, `tier3.py`, `finder.py`,
`bwt_seed.py`, etc.) take effect immediately.

## Testing

`pytest`-based suite under `tests/`. Tests import `from src...`, so **run from
the repo root** (no `conftest.py`/`pytest.ini` — discovery is root-relative).

```bash
python3 -m pytest tests/ -q                                              # whole suite
python3 -m pytest tests/test_ground_truth.py -v                          # one file
python3 -m pytest tests/test_ground_truth.py::TestTier1GroundTruth::test_sensitivity -v  # one test
```

- `tests/test_ground_truth.py` runs the full `TandemRepeatFinder` on synthetic
  sequences in `tests/fixtures/` and asserts per-tier sensitivity/precision
  floors. All tiers run in both builds (the Tier-2/3 `NEEDS_CYTHON` skips are
  gone, since the fallbacks are no longer degenerate).
- `tests/test_accel_parity.py` is the guard for that: it runs the pipeline once
  per accelerator path and requires byte-identical output in all four formats. It
  skips its comparisons when the `.so` is absent (nothing to compare against), so
  run it **with** the extension built.
- `tests/test_align_parity.py` does the same for the *other* pair of duplicate
  implementations: `MotifUtils.align_repeat_region` runs the `libalign_accel` C
  loop when it loads and its own Python loop otherwise. They must return the same
  summary on random repeat regions. (They disagreed on 31% of them until
  2026-07-09; see `docs/2026-07-09-nondeterminism-uninitialised-ptr-table.md`.)
- Baseline: **84 passed** with the `.so`, ~50 passed + native-only skips without it.
- The dev environment with numpy + pydivsufsort + the compiled `.so` (and where
  the benchmark harness runs) is the conda env
  `/data/gpfs/assoc/pgl/bin/conda/conda_envs/bwtandem/bin/python`. `pytest` may
  need installing into it (`python3 -m pip install pytest`).
- `tests/test_satellite_gapfill.py` is the regression guard for the
  divergent-alpha-satellite interior gap-fill (see the env-var section below);
  it self-generates its sequences (no fixture files).

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
  - **Period-stratified gate** (recovers short-period STR recall):
    `TIER1_SHORT_PERIOD_MAX` (default 0 = disabled → baseline) plus
    `TIER1_SHORT_MIN_ARRAY_LEN` / `TIER1_SHORT_MIN_SCORE` (default = the global
    `MIN_ARRAY_LEN` / `MIN_SCORE`). When `motif_len <= SHORT_PERIOD_MAX` the
    length/score acceptance gate uses these relaxed floors instead of the global
    ones, so short perfect cores (e.g. a 7-copy dinucleotide) that sit inside a
    larger region are no longer rejected, while longer motifs stay strict.
    **Recommended (Exp1, on top of comboA):**
    `TIER1_SHORT_PERIOD_MAX=4 TIER1_SHORT_MIN_ARRAY_LEN=18 TIER1_SHORT_MIN_SCORE=18`
    — measured chr21 region recall 52.33%→56.69% at 59.17% precision and
    chr22 55.69%→59.58% at 65.71% precision (per-period p1-9 recall +5pp on
    both), all above tantan's 57.66% precision floor. Lowering the floors
    further or raising `SHORT_PERIOD_MAX` keeps buying recall but collapses
    precision (e.g. period≤9 / floor 14 ≈ 74% recall at 34% precision).
  - **Stitch-seeding** (`TIER1_STITCH_GAP`, default 0 = disabled): merges
    phase-aligned adjacent perfect sub-runs of the same period separated by
    `<= STITCH_GAP * motif_len` bp before extension. Tested but ≈neutral on
    chr21 short-period recall (+0.2pp); kept opt-in for longer-period cores.
- **Tier 2** (`tier2.py`, period 10-20 simple scan): `TIER2_MIN_COPIES`,
  `TIER2_MIN_ARRAY_LEN`, `TIER2_SHORT_REQ_COPIES` (copies required for periods
  <20 bp), `TIER2_MISMATCH`.
  - **Period-10-20 autocorrelation seeder** (`_autocorr_seed`, opt-in
    `TIER2_APPROX_SEED`, default 0 = off → baseline unchanged): a Phase C in the
    simple scan that detects local periodicity directly (vectorized identity
    between offset-p windows) so *diverged* arrays — a substitution in every copy,
    which the exact LCP/k-mer seeds never catch — still get seeded. Knobs:
    `TIER2_APPROX_MIN_IDENTITY` (default 0.78), `TIER2_APPROX_MAX_P` (default 20),
    `TIER2_APPROX_MIN_COPIES` (default 2), `TIER2_APPROX_MAX_SEEDS` (default
    200000). On chr21 it recovers 1688 arrays in ~30 s. **Left OFF for adotto
    (Exp1):** the recovered arrays sit in already-touched GT regions so region
    recall is flat, and it over-calls low-complexity DNA so bp precision drops
    (31→13.5%). May help centromere/satellite at identity ≥0.88 (untested).
    Test: `tests/test_loop_p1020.py`.
- **Satellite interior gap-fill** (`finder.py` `_fill_satellite_gaps`, the
  post-tier backstop for divergent alpha-satellite arrays): `SAT_FILL_MIN_IDENTITY`
  (default 0.45 — min gap autocorrelation identity to accept as satellite; ~2.5x
  above the ~0.18 floor of non-repetitive DNA), `SAT_FILL_MIN_PERIOD`/
  `SAT_FILL_MAX_PERIOD` (default 100/360 — autocorrelation period window, covers
  the 171bp monomer and the dimeric 342bp HOR period), `SAT_ANCHOR_MIN_MOTIF`/
  `SAT_ANCHOR_MAX_MOTIF` (default 100/0 — motif-length window for a repeat to
  anchor gap scanning; MAX=0 = no upper bound so large HOR-unit calls also
  anchor). These recover the interior gaps the 3-tier pipeline leaves uncovered
  *between* large-HOR-motif calls, where divergence is too high for perfect-seed
  extension. Measured on 6 CHM13 centromeric HOR arrays: mean bp-recall
  **92.5%→99.8%** with mean bp-precision held/improved (94.2%→94.6%); per-array
  bp-recall now 99.5-99.99% (was 90.6-95.8%). Defaults reproduce this operating
  point; raising the identity floor back toward 0.55 re-opens the gaps. Regression
  coverage: `tests/test_satellite_gapfill.py`.

The dominant short-STR recall levers are `TIER1_MIN_ARRAY_LEN` + `TIER1_MIN_SCORE`
(copy-count knobs have little effect); lowering them raises recall but drops
precision. See `docs/2026-06-20-exp1-human-sensitivity.md` for the measured
recall/precision frontier and the chosen operating point.

**The recall/precision figures below are from the pre-`706fb76` build** (which read
uninitialised heap in its alignment traceback and was not reproducible). All three
species were re-measured on the fixed build `d52a4ff` (2026-07-10, in
`docs/2026-06-25-catch-all-benchmark-for-filip.md`): the fix trades ~0.1 pp region
recall for ~0.4 pp region precision and leaves every satellite/centromere number
unchanged, so the op-points and their ordering hold. Refreshed genome figures:
human catchF **80.54 % / 50.52 %** (adj 79.5 %), catchH **82.23 % / 48.35 %**.
See `docs/2026-07-09-nondeterminism-uninitialised-ptr-table.md`.

**Exp1 recall op-point (v2.1, precision-leader):** on top of the comboChi base,
`TIER2_MISMATCH=0.30` + `TIER1_FMSCAN_MIN_DENSITY=0.45` + `TIER1_FMSCAN_MIN_LLR=6.0`
lifts full-genome adotto region recall **57.62%→59.04%, bp recall 35.11%→39.42%**,
holding region precision at **58.98%** (above tantan's 57.66% floor — the de-novo
precision leader). Within the ≥57.66% precision floor this is near the ceiling
(env-lever knobs top out ~65% proxy recall; gate lowering collapses precision).

**Exp1 catch-all recall regime (v2.2, 2026-06-25 — `CATCHALL_SCAN`):** to push recall
to ULTRA's level the precision floor must be relaxed to ~ULTRA's 53.65%. The
**catch-all periodicity pass** (`finder.py _catchall_periodicity_fill`) detects
local periodicity directly in uncovered DNA — recovering entirely-missed diverged
STRs the seed-based tiers can't — trading precision for recall via
`CATCHALL_MIN_IDENTITY`. On the v2.2 gate base (v2.1 env + `TIER1_SHORT_PERIOD_MAX=9`
+ `TIER1_SHORT_MIN_ARRAY_LEN=17` + `TIER1_SHORT_MIN_SCORE=17` + `CATCHALL_SCAN=1`),
measured full-genome adotto frontier: **identity 0.76 (+`CATCHALL_MAX_P=50`) → 72.88%
recall / 52.72% precision** (≈ULTRA precision, +15pp recall over v2); **identity 0.72
→ 82.35% recall / 47.99% precision (beats ULTRA's 81.62% recall)**; identity 0.68 →
84.42% / 42.61%. Knobs: `CATCHALL_MIN_IDENTITY` (default 0.72), `CATCHALL_MIN_P`/
`CATCHALL_MAX_P` (1/20), `CATCHALL_MIN_LEN` (20), `CATCHALL_MAX_SEEDS` (200000). bp
precision (~32%) is the cost (over-call in low-complexity). Default OFF (baseline
unchanged).

- **Precision-recovery gates** (the next precision lever, 2026-06-25): two intrinsic
  filters inside the catch-all drop the unsupported over-calls without re-running —
  `CATCHALL_MIN_COPIES` (default 2 = no-op, since `window=2*period` already forces
  copies≥2; raising to 3 drops the shortest 2-copy arrays) and `CATCHALL_MIN_ENTROPY`
  (default 0 = off; per-base Shannon entropy, drops homopolymer-ish low-complexity
  calls). Both default to baseline. **Chosen op-point: `CATCHALL_MIN_COPIES=3`** — a
  chr21+chr22 proxy sweep (`f_*` variants) measured it lifting pool precision
  **50.00%→51.57% (+1.57pp)** while holding pool recall **84.05%→82.73%** (still ≥82%);
  the entropy gate gave ~0 precision gain (the low-complexity calls it removes don't
  move region precision) so it is left off. The copies≥3 op-point is the
  precision-recovered 82%-recall regime (catchH base id0.72 + `CATCHALL_MIN_COPIES=3`).
Loop design/plan: `docs/superpowers/specs|plans/2026-06-23-exp1-recall-loop*.md`;
harness + ledger under `exp1_human/loop/` (`resume.md` = state).

## Dependencies

- **Required**: numpy, pydivsufsort (fast suffix array construction; falls back to NumPy prefix-doubling if unavailable)
- **Optional performance**: numba (JIT for rank queries and LCP), Cython (compiled `_accelerators`)
- **Container**: Singularity definition file at repo root builds a complete environment
- **Environment**: `environment.yml` defines the `bwt` conda env (python, numpy/numba/cython, pytest,
  plus the competitor tools used for benchmarking: trf, tantan, tidehunter, ncrf, mreps). `pydivsufsort`
  is pip-only and needs `--no-build-isolation` (see README "Installation"). Note the env in `environment.yml`
  is not always the interpreter the benchmark scripts call — those hardcode a separate prebuilt env (see
  the path in "Testing"); when adding deps, update whichever env you actually run.

## Architecture

### 3-Tier Detection Pipeline (`finder.py` — `TandemRepeatFinder`)

The coordinator builds a `BWTCore` FM-index once per chromosome, then runs enabled tiers sequentially. Each tier receives regions already found by previous tiers to avoid redundant work.

- **Tier 1** (`tier1.py` — `Tier1STRFinder`): Short perfect repeats, motifs 1–9 bp. For sequences <10 Mbp uses FM-index backward search to enumerate all canonical motifs and locate tandem runs. For larger sequences falls back to a sliding-window scanner with adaptive step size. Requires ≥3 copies.

- **Tier 2** (`tier2.py` — `Tier2LCPFinder`): Medium/imperfect repeats, motifs ≥10 bp. Two sub-phases:
  - **Long-unit strict**: Uses LCP array (Kasai's algorithm) to find adjacent suffix pairs with period ≥20 bp, then extends with mismatch tolerance.
  - **General scanning**: BWT k-mer seeding (`bwt_seed.py`) for periods 10–50 bp, detecting periodic runs in FM-index occurrence positions and extending candidates.

- **Tier 3** (`tier3.py` — `Tier3LongReadFinder`): Long repeats, periods 100 bp–100 kbp. Uses BWT k-mer seeding with large k-mers (20 bp) and sparse sampling (stride=100). Ultra-long arrays (>100 copies or >10 kb) use anchor-based boundary verification instead of full DP refinement.

### Core Modules

- **`bwt_core.py` — `BWTCore`**: FM-index construction (suffix array via pydivsufsort, BWT, checkpointed occurrence arrays). Provides `backward_search()`, `count_occurrences()`, `locate_positions()`. Also the module-level `SENTINEL` and `effective_length()` (sequence length minus a trailing `$`). Suffix-array sampling and the 8-mer hash were removed — both were unreachable and cost tens of GB per chromosome; the `sa_sample_rate` constructor arg is retained but inert.

- **`bwt_seed.py`**: Shared BWT k-mer seeding for Tier 2 and Tier 3. Samples k-mers at configurable stride, finds all occurrences via FM-index, detects arithmetic progressions (periodic runs) in positions, extends with mismatch tolerance.

- **`motif_utils.py` — `MotifUtils`**: Canonical motif rotation (strand-aware), primitive period detection (exact and approximate), DP alignment of repeat copies (`align_repeat_region` with banded Smith-Waterman), consensus building, TRF-compatible statistics, and the `refine_repeat()` entry point used by all tiers.

- **`autocorr.py`**: The "how self-similar is this sequence at offset `p`" math, in one place — `autocorr_identity` (scalar), `windowed_match_counts` (O(n) cumsum), `contiguous_true_runs`. Used by the satellite gap-fill, the catch-all pass, and the Tier-2 seeder. `MotifUtils._str_autocorr_identity` is the string twin.

- **`coverage.py`**: `intervals_to_mask` / `mask_from_repeats` — the boolean coverage mask that Tier 2, Tier 3, the satellite gap-fill, and the catch-all all build over previously-claimed regions.

- **`accelerators.py` / `_accelerators.pyx`**: Cython-accelerated hot paths. Five symbols are consumed by the tiers — `extend_with_mismatches`, `find_periodic_runs`, `lcp_tandem_candidates`, `align_unit_to_window`, `anchor_scan_boundaries` — each aliased to the extension when present and to a faithful pure-Python implementation otherwise. A fallback that cannot reproduce the C result must raise, never return a degenerate value.

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
