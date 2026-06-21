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
