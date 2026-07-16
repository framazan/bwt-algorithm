# Exp1 Region-Recall Maximization Loop (Approach A: staged config→code)

Issue: #3 (continuation). Branch: `perf/exp1-human-sensitivity`.
Date: 2026-06-23. Status: design — pending user review.
Parent design: `docs/2026-06-21-exp1-max-recall-design.md` (diagnosis still valid).

## Goal / objective function

Maximize **full-genome adotto v1.2.1 region recall** subject to a **hard floor of
region precision ≥ 57.66%** (tantan's precision — keep bwtandem the de-novo
precision leader). Aspiration: **82%** (ULTRA's recall). The loop terminates
successfully if it reaches recall ≥82% at precision ≥57.66% on the full genome;
otherwise it reports the **best frontier point** it reached and stops.

**Honest feasibility note.** Current v2 (commit `de87396`) sits at **recall
57.62% / precision 61.04%** genome-wide — only **3.38 pp of precision headroom**
above the floor. Lowering thresholds buys recall but spends that headroom almost
immediately (the parent design measured the chr21 knob ceiling at ~55% recall /
62% precision, collapsing to 49% precision when pushed). So **recall must come
from recovering true positives the tiers currently cannot find — i.e. code — not
from relaxing acceptance gates.** 82% at ≥57.66% precision would beat ULTRA
(82%@53.65%) *and* tantan (78%@57.66%) on the frontier; it is a stretch, and the
loop is built to report wherever it honestly lands.

## Where we start (measured)

Genome-wide, full adotto v1.2.1 (1,784,804 regions), primary chr:

| tool | region recall | region precision | bp recall |
|---|---|---|---|
| **bwtandem v2 (current)** | 57.62% | 61.04% | 35.11% |
| trf | 31.88% | 94.86% | 30.26% |
| ultra | 81.62% | 53.65% | 38.14% |
| tantan | 78.00% | **57.66%** ← floor | 23.54% |

Per-period gap vs ULTRA (where the missing recall lives):

| period | bwtandem | ultra | regions | status |
|---|---|---|---|---|
| 1-6 | 53% | 78% | 1.05M | FM-scan detector active (commit `4dbafe3`) |
| 7-9 | 53% | 82% | 177k | **FM-scan not extended here** → lever |
| 10-20 | 59% | 89% | 358k | **Tier2 p10-20 weak** (logs: Phase B +0) → biggest lever |
| 21+ | 80-93% | 84-90% | 204k | already competitive — not the problem |

**Already landed** (from parent plan, ad-hoc not checklist order): C1 FM-scan
seeding p1-6 (`4dbafe3`), C2 rolling-consensus extender (`adc694e`,`3d1cfe4`),
period-stratified gate (`ed1e9d5`). **Remaining code levers:** period 10-20 path
(parent's C4), FM-scan extension to p7-9, merge/fragmentation relaxation (C5).

## Constraints / environment

- **Login node has only 4 GB RAM → ALL compute runs via `sbatch`.** Never run a
  detection or full-genome scoring job on the login node.
- Conda python: `/data/gpfs/assoc/pgl/bin/conda/conda_envs/bwtandem/bin/python`.
- Pure-Python tier edits (`tier1.py`/`tier2.py`/`bwt_seed.py`/`finder.py`) take
  effect immediately — no rebuild. Only `_accelerators.pyx` edits need the `.so`
  rebuild (defer; prototype in Python first).
- Scorer: `exp1_human/score_overlap.py` (bedtools at
  `/data/gpfs/assoc/pgl/bin/bedtools2/bin/bedtools`), reproduces Filip's metric
  definitions exactly.
- Pipeline is ~0.1% nondeterministic run-to-run → **validate by metric, not
  byte-identity.**

## Approach A — staged config → code

- **Phase 1 — config sweep (establish the knob ceiling).** Coordinate-descent
  over the documented env levers around the current op-point on the chr21+chr22
  proxy; keep any change that raises pooled recall while holding precision ≥
  floor. Cheap (~8 min/iter), zero code risk. Output: the maximum recall the
  knobs can reach at the floor, and which levers still have headroom.
- **Phase 2 — gap-targeted code (recover true positives).** Attack the largest
  TP-recoverable buckets, in expected-payoff order:
  1. **Period 10-20 path (C4)** — 358k regions, +30 pp vs ultra. Extend Tier2
     strict-LCP down to ~12 bp and add an approximate/spaced k-mer seed mode in
     `bwt_seed.py`; `max_occurrences` cap as the precision guard.
  2. **FM-scan → p7-9** — 177k regions. Extend the FM-index canonical-motif
     enumeration (currently p1-6) to motif length 7-9.
  3. **Merge/fragmentation relaxation (C5)** — rotation-aware fuzzy short-period
     merge in `finder.py` (currently fuzzy only for motifs ≥50 bp).
  Each code change is gated: `pytest tests/test_ground_truth.py` green **and**
  pooled proxy precision ≥ floor **and** pooled proxy recall strictly up →
  accept and commit; else revert.

Rejected alternatives: B (code-first) — higher ceiling but slower, harder
attribution; C (config-only) — cannot pass ~60% at the floor (user chose code).

## Components (new, under `exp1_human/loop/`)

- **`loop/run_proxy.sbatch`** — args: a run tag + an env-override block. Runs
  chr21.fa and chr22.fa (`src.main`, period 1-2000, threads mem-safe), scores
  the pooled output against the pooled GT, appends one row to `loop/ledger.tsv`.
  `mem`/`time` sized for the proxy (≈8–12 min, ~8 GB).
- **`loop/gt_2122_all.bed`** — built once = `cat gt_chr21_all.bed gt_chr22_all.bed`.
  Pooled scoring (`recall = pooled GT hit / pooled GT`) mirrors how the
  full-genome metric aggregates, unlike per-chr averaging.
- **`loop/ledger.tsv`** — append-only results memory. One row per evaluated
  candidate: `iso8601, tag, phase, kind(config|code), source(env-string|commit),
  chr21_rec, chr21_prec, chr22_rec, chr22_prec, pool_rec, pool_prec, accepted,
  note`. This is the loop's cross-wakeup state.
- **`loop/best.json`** — current best accepted point (proxy metrics + the
  full-genome confirmation if validated) and the calibrated proxy→full-genome
  precision offset.
- **`loop/validate_full.sbatch`** — runs the best candidate full-genome
  (`hg38_primary.fa`) and scores vs `adotto_primary.bed`; the only place the
  reported recall/precision and the floor are authoritative.
- **`loop/README.md`** — schema + how to resume the loop from the ledger.

## Loop control (`/loop` self-paced agent)

The agent is the controller; `loop/ledger.tsv` is its memory. Each wakeup:

1. Check the in-flight sbatch job (`squeue`/`sacct`).
2. **Running** → `ScheduleWakeup ~270 s` (proxy job ≈8–12 min ⇒ 2–3 polls; stays
   within the prompt-cache TTL).
3. **Done** → read the new ledger row. If `pool_prec ≥ proxy_floor` and
   `pool_rec >` best: accept, update `best.json`; if it is a new best with
   margin, submit `validate_full.sbatch`.
4. Choose the next candidate — Phase 1: next lever in the coordinate-descent
   grid; Phase 2: next code patch (implement → pytest → proxy). Submit it,
   `ScheduleWakeup`.

**Proxy precision floor with margin.** The proxy reads precision higher than the
full genome (measured: chr21/22 ~6–7 pp above genome-wide). Iteration 0 re-runs
the *current* op-point on the proxy; since its genome-wide precision is known
(61.04%), the offset is calibrated and `proxy_floor = 57.66% + offset` (≈64%
initially), refined after each full-genome validation.

**Stop conditions.** (a) Success: full-genome recall ≥82% at precision ≥57.66%.
(b) Frontier: Phase 1 ceiling reached **and** all Phase 2 code levers tried,
with ≥3 consecutive iterations yielding no accepted improvement → report the best
point, stop. (c) User halts at any time (ledger + best.json make it resumable).

**Safety.** All code edits to pure-Python tiers only (immediate effect). Every
accepted code change keeps `tests/test_ground_truth.py` green and is committed
individually on `perf/exp1-human-sensitivity` with its proxy metrics in the
message, so any step is revertible. No `_accelerators.pyx` / `.so` rebuild unless
a validated Python prototype justifies the port.

## Success criteria

- Primary: full-genome adotto region recall maximized at precision ≥ 57.66%;
  82% ends the loop successfully.
- Tracked every iteration: pooled proxy recall/precision; on validation also
  genome-wide region recall, region precision, bp recall, bp precision, peak RAM,
  runtime — appended to the ledger, never selectively dropped.
- Deliverable at stop: updated `benchmarking_results_updated.md` Exp1 row + an
  honest before/after, and the `loop/` harness preserved for reproduction.

## Risks & mitigations

- **82% may be unreachable at the floor.** The loop reports the honest frontier;
  "best precision among high-recall callers" remains a valid narrative.
- **Autonomous code changes drift/break.** Guarded by the test suite + precision
  floor + per-step commits (revert any step).
- **Proxy ↔ full-genome divergence.** Calibrated offset + full-genome validation
  of every claimed best before it is reported.
- **sbatch queue contention.** Self-pacing tolerates it; the ledger makes long
  waits harmless.

## Out of scope

- `_accelerators.pyx` rebuild-requiring changes (unless a Python prototype wins
  first), Tier3 / long-period work, the CHM13 centromere Phase 2 of the parent
  design, and any recall pursued below the 57.66% precision floor.
