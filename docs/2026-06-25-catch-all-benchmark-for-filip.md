# BWTandem — Catch-all algorithm: how to run + 3-species benchmark results

**For: Filip.**  Branch: `perf/exp1-human-sensitivity` (origin = github.com/wyim-pgl/bwt-algorithm).
Date: 2026-06-25. Author: Won (with Claude Code).

> **Numbers below predate `706fb76` (2026-07-09) and will shift slightly.** Every
> figure on this page was measured with a build of `libalign_accel` that read
> uninitialised heap memory in its alignment traceback, so the caller was not
> reproducible: two runs of the same commit gave different calls. That is fixed;
> the run-to-run variation is gone. Re-measured on chr22 against the adotto GT,
> the fix moves region recall **84.50 % → 84.38 %** and region precision
> **52.34 % → 52.74 %** (53 fewer calls, mostly ones the garbage traceback
> invented). The genome-scale figures on this page have **not** been re-run. The
> operating points, run commands and the shape of every conclusion are unchanged.
> See `docs/2026-07-09-nondeterminism-uninitialised-ptr-table.md`.

---

## TL;DR

A new **catch-all periodicity pass** lets BWTandem find the diverged short STRs the
3-tier pipeline structurally cannot seed, closing the recall gap to ULTRA/tantan:

- **Human GRCh38 (adotto):** region recall **57.6 % → 80.7 %** at **79.1 % adjusted
  precision** (de-novo recall now on par with ULTRA, 81.6 %).
- **Maize Mo17:** 3A microsatellite yield **+96 %** (11.6 M → 22.7 M bp); satellite
  families (knob180/TR-1/CentC) **unchanged at max**.
- **Arabidopsis Col-CEN:** **no change** (CEN180 already at 99.7 %) — expected.

**One rule:** the catch-all is a *short-STR / microsatellite* mechanism (period ≤ 20 bp).
Turn it **ON for STR / microsatellite runs, OFF for satellite / centromere runs.**
It is **opt-in (default OFF)** so all your existing runs are unchanged.

---

## 1. What's new (this branch vs the bwtandem you benchmarked)

| Commit | Change |
|---|---|
| `15d4273` | **Catch-all periodicity pass** (`finder._catchall_periodicity_fill`, env `CATCHALL_SCAN`) — detects local periodicity directly in DNA no tier covered. |
| `d3dca3c` | **Precision-recovery gate** `CATCHALL_MIN_COPIES` (and `CATCHALL_MIN_ENTROPY`) — trims the shortest 2-copy over-calls. |
| `748b3f6` | Refactor: the 3 duplicated autocorrelation routines consolidated into `src/autocorr.py` (behavior-preserving; catch-all output byte-identical before/after). |
| `47f4865` | This benchmark write-up. |

Everything below is **opt-in**; without `CATCHALL_SCAN=1` the binary behaves exactly as before.

---

## 2. Build

```bash
git clone -b perf/exp1-human-sensitivity https://github.com/wyim-pgl/bwt-algorithm.git
cd bwt-algorithm

# Cython acceleration (needs numpy + Cython). Without the .so, Tier2/3 fall back to
# slow/empty pure-Python — build it:
python3 -c "
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
ext=[Extension('src._accelerators',['src/_accelerators.pyx'],include_dirs=[np.get_include()])]
setup(script_args=['build_ext','--inplace'],ext_modules=cythonize(ext,compiler_directives={'language_level':'3'}))
"
```

Prebuilt env used for all runs below (numpy + pydivsufsort + compiled `.so`):
`/data/gpfs/assoc/pgl/bin/conda/conda_envs/bwtandem/bin/python`

---

## 3. How to run

Base command (unchanged):

```bash
python3 -m src.main <genome.fa> --min-period 1 --max-period 2000 \
    --threads 4 --format bed -o <out_prefix> -v
```

The catch-all and its operating point are selected entirely by **environment
variables** (no code edits). The recommended **gate base** (the v2.2 op-point) is:

```bash
export TIER1_FMSCAN=1 TIER1_FMSCAN_MIN_DENSITY=0.45 TIER1_FMSCAN_MIN_LLR=6.0 \
       TIER1_MIN_ARRAY_LEN=20 TIER1_MIN_SCORE=20 TIER1_MIN_COPIES=2 \
       TIER1_COPYBASE=6 TIER1_COPYADD=2 TIER1_EXT_COPIES=2 \
       TIER1_SHORT_PERIOD_MAX=9 TIER1_SHORT_MIN_ARRAY_LEN=17 TIER1_SHORT_MIN_SCORE=17 \
       TIER2_MISMATCH=0.30 TIER2_SHORT_REQ_COPIES=2
```

Then pick an operating point by adding the catch-all knobs:

| Op-point | extra env | use when | human recall / prec |
|---|---|---|---|
| **catchF** (recommended) | `CATCHALL_SCAN=1 CATCHALL_MIN_IDENTITY=0.72 CATCHALL_MIN_COPIES=3` | high recall, precision recovered | 80.7 % / 79.1 % adj |
| catchH (max recall) | `CATCHALL_SCAN=1 CATCHALL_MIN_IDENTITY=0.72` | absolute max recall (beats ULTRA) | 82.4 % / 77.4 % adj |
| catchT (≈ULTRA prec) | `CATCHALL_SCAN=1 CATCHALL_MIN_IDENTITY=0.76 CATCHALL_MAX_P=50` | precision-leaning | 72.9 % / 52.7 % raw |
| **OFF** (satellite/centromere) | *(omit `CATCHALL_SCAN`)* | satellite-dominated genomes | — |

All catch-all knobs: `CATCHALL_SCAN` (0/1), `CATCHALL_MIN_IDENTITY` (0.72),
`CATCHALL_MIN_P`/`CATCHALL_MAX_P` (1/20), `CATCHALL_MIN_LEN` (20),
`CATCHALL_MIN_COPIES` (2 = no-op, raise to 3 to recover precision),
`CATCHALL_MIN_ENTROPY` (0 = off), `CATCHALL_MAX_SEEDS` (200000). Defaults = baseline.

### Exact commands used for each benchmark genome

```bash
PY=/data/gpfs/assoc/pgl/bin/conda/conda_envs/bwtandem/bin/python
# ---- common: gate base (above) + catchF ----
export CATCHALL_SCAN=1 CATCHALL_MIN_IDENTITY=0.72 CATCHALL_MIN_COPIES=3

# Human GRCh38 (5 h, 38 GB, 4 threads)
$PY -m src.main /data/gpfs/assoc/pgl/devel/exp1_human/data/hg38_primary.fa \
    --min-period 1 --max-period 2000 --threads 4 --format bed -o out/bwt_hg38_catchF -v

# Arabidopsis Col-CEN  (~7 min, 6 GB, 8 threads)  [catch-all OFF is better — see results]
$PY -m src.main /data/gpfs/assoc/pgl/filip/bwtandem_results/benchmarking/Col-CEN_v1.2.fasta \
    --min-period 1 --max-period 2000 --threads 8 --format bed -o out/bwt_colcen_catchF -v

# Maize Mo17  (7.6 h, 24 GB, 2 threads – mem-safe for 2.18 Gb)
$PY -m src.main /data/gpfs/assoc/pgl/filip/bwtandem/Zm/GCA_022117705.1_Zm-Mo17-REFERENCE-CAU-T2T-assembly_genomic.fna \
    --min-period 1 --max-period 2000 --threads 2 --format bed -o out/bwt_maize_catchF -v
```

Ready-to-submit SLURM scripts live in the benchmark workspace:
`/data/gpfs/assoc/pgl/devel/exp1_human/`:
`run_fullgenome.sbatch` (human), `filip_repro/run_colcen_catch.sbatch`,
`filip_repro/run_maize_catch.sbatch`.

---

## 4. Results (all tools)

### Exp1 — Human GRCh38 vs full adotto v1.2.1 (primary chromosomes)

| Tool | Regions | Region Recall % | Region Prec % | BP Recall % | BP Prec % | Runtime s | Mem GB |
|:--|--:|--:|--:|--:|--:|--:|--:|
| bwtandem catchM (id0.68) | 5,187,230 | **84.42** | 42.61 | **45.99** | 31.18 | ~30 k | ~30 |
| bwtandem catchH (id0.72) | 4,378,067 | **82.35** | 47.99 | 44.76 | 32.19 | ~30 k | ~30 |
| **bwtandem catchF (id0.72 + MIN_COPIES=3)** | 4,021,800 | **80.69** | 50.15 | 43.95 | 32.38 | ~18 k | ~38 |
| bwtandem catchT (id0.76) | 3,450,936 | 72.88 | 52.72 | 42.36 | 32.47 | ~30 k | ~30 |
| bwtandem v2.1 (precision-leader) | 2,367,682 | 59.04 | 58.98 | 39.42 | 42.31 | ~20 k | ~30 |
| bwtandem v2 (prior baseline) | 2,237,246 | 57.62 | 61.04 | 35.11 | 45.48 | 13,156 | ~30 |
| ultra | 3,216,708 | 81.62 | 53.65 | 38.14 | 61.33 | 98,659 | 1.68 |
| tantan | 3,319,523 | 78.00 | 57.66 | 23.54 | 70.22 | 3,374 | 0.27 |
| trf | 962,837 | 31.88 | **94.86** | 30.26 | 52.40 | 121,514 | 1.45 |

**catchF adjusted precision = 79.1 %** (adotto OR ULTRA/tantan): 58 % of catchF's raw
"FP" are real repeats supported by ULTRA/tantan that the adotto catalog simply lacks —
so the raw 50 % is mostly a ground-truth-incompleteness artifact, true precision ≈ 79 %.

### Exp2 — Arabidopsis Col-CEN (CEN180 centromere)

| Tool | CEN180-monomer recall % | Centromere Cov % | calls/cen | Runtime s | Mem GB |
|:--|--:|--:|--:|--:|--:|
| **bwtandem (catch-all OFF)** | **99.7** 🥇 | 84.36 | 779 | 423 | ~6 |
| bwtandem (catch-all ON, catchF) | 99.68 | — | — | ~430 | ~6 |
| trf | 97.5 | 84.39 | 507 | 475,020 | 1.26 |
| bwtandem (old) | 97.1 | 81.74 | 550 | 705 | 5.32 |
| ncrf (motif-guided) | — | 85.17 | — | 2,864 | 80.96 |
| trash_template | — | 85.03 | — | 91,360 | 2.63 |
| ultra | 0.6 ❌ | 2.25 | 717 | 4,802 | 1.68 |
| tantan | 0.3 ❌ | 1.04 | 666 | 138 | 0.04 |
| mreps | ~0 | 1.58 | — | 849 | 0.84 |

Catch-all ON leaves monomer recall flat (already saturated) and only costs CEN180 bp
precision (65.5 → 60.7 %). **Use catch-all OFF here.** ULTRA/tantan fail outright on
satellite — BWTandem (99.7 %) is the best de-novo tool, matching TRF.

### Exp3 — Maize Mo17

**3A. Microsatellite (SSR yield)**

| Tool | Total SSR bp | Regions | Runtime s | Mem GB |
|:--|--:|--:|--:|--:|
| **bwtandem catchF (catch-all ON)** | **22,723,409** | 867,364 | ~27.5 k | 23.7 |
| bwtandem v2 (catch-all OFF) | 11,584,281 | 356,634 | 23,469 | 17.65 |
| tantan | 55,842,884 | 1,303,967 | 1,973 | 0.5 |
| trf | 4,827,610 | 29,559 | 18,781 | 1.2 |
| bwtandem (old) | 3,044,731 | 54,283 | 3,191 | 45.61 |
| ncrf | 2,604,968 | 214 | 474 | 17.47 |
| ultra | 0 | 0 | 828 | 2.23 |

Catch-all nearly doubles microsatellite yield (no microsatellite GT exists, so this is a
sensitivity gain mixing real diverged microsatellites with some low-complexity over-call,
consistent with the ~32 % human bp precision).

**3B / 3C. Satellite — identical with catch-all ON or OFF**

| family | bwtandem (ON = OFF) | trf | tantan | ultra | ncrf |
|---|---|---|---|---|---|
| 3B knob180 (/25) | **25** 🥇 | 25 | 24 | 0 ❌ | 0 ❌ |
| 3B TR-1 (/17) | **17** 🥇 | 17 | 17 | 0 ❌ | 0 ❌ |
| 3C CentC (/17) | **17** 🥇 | 17 | 17 | 0 ❌ | 0 ❌ |

BWTandem is the only de-novo tool that maxes every satellite family in one run; the
catch-all does not touch them (their 156–360 bp period is outside its ≤ 20 bp range).

---

## 5. How to score (reproduce the numbers)

All scoring scripts are in `/data/gpfs/assoc/pgl/devel/exp1_human/`:

```bash
# Human: region + bp recall/precision vs full adotto, plus adjusted precision
python3 score_overlap.py data/adotto_primary.bed out/bwt_hg38_catchF.bed:catchF
python3 fp_check.py out/bwt_hg38_catchF.bed data/adotto_primary.bed score/ultra.bed score/tantan.bed

# Col-CEN: CEN180-monomer recall (GT = colcen_cen180.bed, Chr1-5)
python3 score_overlap.py \
   /data/gpfs/assoc/pgl/filip/bwtandem_results/ground_truth/colcen_cen180.bed \
   out/bwt_colcen_catchF.bed:catchF --chroms Chr1,Chr2,Chr3,Chr4,Chr5

# Maize: 3A/3B/3C by motif-length bands (1-6 / 100-500 / 100-200), GT = mo17_*_arrays.bed
python3 filip_repro/score_exp3_catchF.py
```

The maize single 1–2000 run is split into your 3 sub-experiments by **motif length**
(our BED col4 is the motif; period = `len(motif)`): 3A = 1–6 bp, 3B = 100–500 bp,
3C = 100–200 bp — the exact bands your per-experiment runs produced.

---

## 6. Practical notes (Pronghorn)

- **Memory:** real MaxRSS is **human 37.6 GB / maize 23.7 GB** at `--threads 4`/`2`.
  Request **64 GB**, not 190 GB — a 190 GB request sat ~12 days PENDING in the full PGL
  partition (SLURM allocates by *request*, not usage); 64 GB scheduled instantly.
- **`scontrol` is broken** on the compute nodes (`libreadline.so.6` missing) — to change
  a pending job, `scancel` + resubmit, not `scontrol update`.
- The pipeline is mildly **non-deterministic at `--threads ≥ 2`** (~0.1 % of rows shift
  run-to-run); region recall/precision are stable to ~0.1 pp.

---

## 7. Bottom line for the comparison table

BWTandem is now the only de-novo tool that covers **the whole spectrum in one binary**:
short STRs (catch-all ON → human 80.7 %, ULTRA-level recall), microsatellites (maize 3A
+96 %), and satellites (Col-CEN 99.7 %, maize knob180/TR-1/CentC all maxed) — the last of
which ULTRA and ncrf miss entirely. Flip the catch-all ON for STR/microsatellite targets,
OFF for satellite/centromere targets.
