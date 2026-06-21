# Experiment 1 (Human GRCh38, general TR) — Sensitivity Improvement Plan

Issue: #3. Branch: `perf/exp1-human-sensitivity` (off `docs/linux-setup`).
Date: 2026-06-20.

## Problem

BWTandem's adotto **region recall is 36.87%**, far below ultra (80.85%) and
tantan (75.17%), below trf (41.33%). Peak memory is **41.98 GB** on GRCh38.

### Verified diagnosis (evidence-based, this session)

- Scoring (from filip `scripts/compute_metrics.py`): *Adotto Recall* =
  `bedtools intersect -a GT -b tool -wa -u` / total GT regions — i.e. the
  fraction of ground-truth TR regions touched by **≥1** tool call (any overlap).
- The gap is **real sensitivity, not a naming/coordinate artifact**: all tools
  use `chr1..chrX` naming on the main chromosomes (bwtandem's leading `NT_*`
  rows are just unplaced scaffolds sorting first). bwtandem emits ~695 K calls
  vs ultra 3.35 M / tantan 3.47 M (≈1/5), and per chromosome bwtandem has
  ~45 K calls on chr1 vs ultra 258 K / tantan 264 K. It misses ~63% of adotto
  regions outright.
- adotto is dominated by **short tandem repeats (period 1–6)**. Structural
  causes in code that suppress short/imperfect STR detection:
  - `src/bwt_seed.py`: `effective_kmer = min(6, min_period)` and exact k-mer
    multi-copy seeding → imperfect short STRs never seed.
  - `src/tier2.py`: `self.min_period = max(10, min_period)` → periods < 10
    are not handled by tier2.
  - tolerance / `min_copies` settings reject short imperfect repeats.
- **Precision ~0.1% for every tool** is a separate metric artifact (adotto is a
  curated subset, so genome-wide calls fall outside it). Out of scope for the
  algorithm work; noted for the eventual metric fix (#3 to-do).

## Goal & success criteria

- Genuinely raise bwtandem adotto **region recall** from 36.87% toward the
  ultra/tantan band (75–81%) via honest sensitivity improvements.
- Constraints: keep peak memory ≤ current 41.98 GB (ideally reduce); keep
  precision and runtime reasonable; report a **before/after** with **all tools
  re-scored on the identical GT**.
- Honest stance: beating ultra (80.85%) is a stretch target. Primary goal is to
  **close most of the gap**; the real recall is reported even if still below
  ultra. No GT overfitting, no hardcoding, no selective reporting.

## Decisions (agreed)

- **Strategy:** parameter sweeps first; surgical code changes only where
  parameters hit a wall.
- **Ground truth:** download the **public adotto catalog** + a chr-named GRCh38
  FASTA; re-score every readable tool BED on the same GT (filip's 65 KB
  `adotto_tr_regions.bed` / `HG002_..._v1.0.bed` are mode 700 = unreadable, so
  absolute numbers will differ from filip's curated subset; relative comparison
  and before/after deltas remain valid).
- **Env / run:** clone `/data/gpfs/assoc/pgl/devel/bwt-algorithm`, conda env
  `bwtandem`, full-genome runs via SLURM `sbatch`. Internet confirmed available.

## Plan

1. **Harness (foundation).** Download public adotto TR catalog + chr-named
   GRCh38 FASTA. Re-score all tool BEDs (bwtandem/trf/ultra/tantan/mreps) with
   the `compute_metrics.py` region-overlap + bp recall/precision logic. Confirm
   re-scored bwtandem is in the ~37% ballpark (sanity check vs filip).
2. **Diagnosis (chr21 + chr22).** Extract the two small chromosomes. Bin the
   *missed* adotto regions (no bwtandem overlap) by period / length / purity →
   confirm short, imperfect STR concentration; quantify the recoverable share.
3. **Parameter sweep (chr21/22).** Sweep `effective_kmer` floor, tier2
   `min_period` floor, `tolerance_ratio` / `min_copies`, tier enablement,
   `--min-period`. Record recall / precision / runtime / peak-mem per config.
4. **Surgical code changes (only if needed).** Candidates: relax exact-kmer
   seeding for short periods (allow imperfect copies); lower tier2 floor; add a
   short-period path. Validate each on chr21/22 before/after.
5. **Memory.** Investigate `sa_sample_rate` / per-chromosome chunked indexing to
   reduce 41.98 GB (also required to make the full-genome rerun feasible).
6. **Full-genome validation.** Run the best config on full GRCh38 via `sbatch`;
   re-score all tools; commit an honest before/after table.

## Non-goals (this iteration)

- Fixing the precision methodology (#3 separate to-do).
- Benchmarks #4 (maize) and #5 (arabidopsis) — preserved on
  `wip/maize-satellite-sensitivity`, addressed later.
- CHM13 — out of scope until memory is reduced.

---

## Results (chr21 + chr22, scored vs full public adotto v1.2.1 catalog)

Harness validated: a fresh baseline run on the instrumented code (no env vars)
reproduces the published baseline exactly (chr21 region recall 29.50%,
precision 79.50%), and the tool ranking matches filip's report
(ultra/tantan high recall, bwtandem/trf low). Absolute numbers differ from
filip's because we score against the full 1.78M-region adotto catalog, not the
locked 65 KB curated subset.

### Diagnosis (verified)

The recall gap is concentrated in **short periods**: dominant period ≤20 bp
covers 85.6% of adotto regions, and that is exactly where baseline bwtandem is
weak (period 1-6: 21%, 7-9: 16%, 10-20: 30% recall) while it matches ultra at
period 21-100. 53.6% of all regions are baseline-missed but ultra-hittable
(= recoverable). The miss is driven by Tier1's strict short-STR thresholds
(`min_array_length=26`, high `dynamic_min_copies`, `min_score>=30`), not region
length (adotto regions are mostly >=54 bp).

### Chosen config (`comboA`)

Tier1: `MIN_ARRAY_LEN=20 MIN_SCORE=20 MIN_COPIES=2 COPYBASE=6 COPYADD=2
EXT_COPIES=2`; Tier2: `SHORT_REQ_COPIES=2 MISMATCH=0.25`.

| metric | chr21 base | chr21 comboA | chr22 base | chr22 comboA |
|---|---|---|---|---|
| region recall | 29.50% | **52.33%** | 29.79% | **55.69%** |
| raw precision | 79.50% | 65.77% | 83.02% | 71.81% |
| adjusted precision* | ~99.9% | **83.1%** | — | **86.3%** |
| bp recall | 34.98% | 42.83% | 29.46% | 38.78% |

\* adjusted precision = fraction of calls overlapping adotto **OR** confirmed by
ultra/tantan. The large raw-precision drop is mostly catalog incompleteness:
~half of the "new" calls outside adotto are confirmed by an independent de novo
tool. Genuine over-calling is only ~10-15% of calls. After tuning, bwtandem's
**raw** precision (66-72%) is still higher than ultra (54-59%) and tantan
(58-62%) — it remains the most precise general-TR tool while roughly doubling
recall (~+24 pp, generalises across both chromosomes).

### Negative result: entropy gate does not work

An opt-in low-complexity entropy gate (`TIER1_ENTROPY_GATE`) was tested to
suppress over-calling at aggressive thresholds. It **fails**: it removes real
low-entropy short STRs (AT/AAT-type) so recall drops sharply while adjusted
precision does not improve. Spurious and genuine short STRs are intrinsically
indistinguishable, so there is no clean quality gate for this regime. Kept in
the code as a documented, default-off option.

### Tier2 contribution

Lowering the period-10-20 simple-scan thresholds
(`TIER2_SHORT_REQ_COPIES 3->2`, `MISMATCH 0.2->0.25`) lifts period-10-20 recall
~45% -> 54% on chr21 for a small overall gain (+1.6 pp) at minor precision cost.
A more aggressive Tier2 (`comboB`: also `MIN_COPIES=2 MISMATCH=0.3`) reaches
54.9% overall recall but at 80% adjusted precision — diminishing returns, so
`comboA` is the chosen operating point.

### Honest ceiling

Raising recall to ultra's raw level (~82%) is not achievable without trading
away precision (at `len16_sc16`, recall hits 61% but ~half the new calls are
unsupported garbage). The honest, defensible outcome for #3: bwtandem becomes
**much more competitive on recall (~2x) while remaining the most precise tool**,
and the `adjusted precision` reframing simultaneously addresses the #3
precision-methodology to-do.

### Pending

Full-genome (hg38 primary chromosomes) validation of `comboA` running on SLURM
(`cpu-s2-core-0`, job submitted) to produce the headline before/after on the
whole genome, scored against the full adotto catalog alongside all reference
tools.
