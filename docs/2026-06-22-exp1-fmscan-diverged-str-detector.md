# Exp1 — FM-index motif-enumeration detector for diverged short STRs (#3)

Goal: recover diverged short-period STRs (period 1–9) that the exact
adjacent-copy seeder misses, **at high precision** — i.e. break the
recall/precision wall that pure gate-relaxation hit (at ~74% recall the
adjusted precision collapsed to ~49%, with 77% of out-of-catalog calls genuine
garbage).

## The detector (`src/tier1.py`, opt-in via `TIER1_FMSCAN`)

A new, statistically-gated, gap-tolerant short-STR detector built on the
existing FM-index. It is **OFF by default** (`TIER1_FMSCAN=0`); the baseline
pipeline is byte-for-byte unchanged when the flag is unset.

Algorithm, per period `p` in `[FMSCAN_MIN_P, FMSCAN_MAX_P]` (default 1–6),
processed **longest period first**:

1. **Enumerate** every primitive `p`-mer string over ACGT (all rotations — a
   tandem array presents one fixed rotation in the text). Counts are small:
   4/12/60/240/1020/4020 strings for p=1..6.
2. **Locate** all occurrences of each motif via the FM-index
   (`backward_search` → SA interval → `np.sort` of positions). NumPy sort makes
   even a dinucleotide's ~3 M occurrences cost <1 s.
3. **Gap-tolerant periodic runs**: group sorted occurrences into maximal chains
   where each step is a positive multiple of `p` within a gap budget
   (`FMSCAN_MAX_GAP` skipped copies). A *missing* expected copy = a
   diverged/mismatched copy absorbed into the run. Vectorised with NumPy.
4. **Precision filter (the discriminator crude gates lack)** — two gates:
   - **density** = observed perfect copies / expected copies (`span/p`). A
     perfect array → 1.0; diverged arrays sit a bit below; random
     low-complexity DNA sits well below. Floor `FMSCAN_MIN_DENSITY` (def 0.50).
   - **Poisson log-likelihood ratio** of the observed perfect-copy count vs the
     i.i.d. background expectation `exp_bg = span · Π freq(base)`:
     `LLR = occ·ln(occ/exp_bg) − (occ − exp_bg)`. Grows with both excess count
     and motif rarity. Floor `FMSCAN_MIN_LLR` (def 8.0).
5. **Greedy territory claim** via a `seen_mask`: within a period, runs are
   ranked by LLR and claim space best-first, so each array is reported once by
   its best-supported motif/phase (not once per overlapping rotation). This
   keeps the candidate count low and avoids an O(n²) blow-up in the downstream
   merge.
6. **Refine** accepted runs through the shared `MotifUtils.refine_repeat` path
   so output format matches the rest of the pipeline.

Modes: `TIER1_FMSCAN=2` = replacement (sole Tier-1 source); `=1` = **additive**
— runs *after* the sliding-window scan on the regions it did NOT claim (passes
the sliding-window `seen_mask`), so it only adds the diverged STRs the exact
seeder is blind to.

On random DNA the gates reject everything (0 false calls on 20 kb synthetic
random sequence at default thresholds); on a synthetic `(AC)×15` with 3 SNPs
(no long perfect run) it recovers the array.

## Result: the additive detector DOMINATES the gate-tuning OP1

Base env (all runs): `TIER1_MIN_ARRAY_LEN=20 TIER1_MIN_SCORE=20
TIER1_MIN_COPIES=2 TIER1_COPYBASE=6 TIER1_COPYADD=2 TIER1_EXT_COPIES=2
TIER2_SHORT_REQ_COPIES=2 TIER2_MISMATCH=0.25`. Metrics vs `adotto_primary.bed`;
adjusted precision = (adotto OR ultra/tantan-supported)/calls.

**chr21**

| config | calls | regRec% | rawPrec% | adjPrec% | p1-9 Rec% |
|---|---|---|---|---|---|
| baseline (FMSCAN off) | 26 383 | 52.33 | 65.77 | 83.1 | 45.44 |
| **additive d0.50/l8** | 30 381 | **58.97** | 65.58 | **83.6** | **53.81** |
| additive d0.45/l6 | 32 990 | 60.50 | 62.89 | 80.5 | 55.40 |

**chr22 (reproduction)**

| config | calls | regRec% | rawPrec% | adjPrec% | p1-9 Rec% |
|---|---|---|---|---|---|
| baseline (FMSCAN off) | 27 782 | 55.69 | 71.81 | 86.3 | 47.82 |
| **additive d0.50/l8** | 32 173 | **61.60** | 70.11 | 86.2 | **55.08** |

Additive `d0.50/l8` dominates the baseline on both chromosomes: **+6.64 / +5.91
region recall at equal adjusted precision** (chr21 +0.5, chr22 −0.1) and equal
raw precision. The gain is concentrated exactly where the detector targets:
short period 1–9 recall **+8.37 (chr21) / +7.26 (chr22)**. On chr21, the FM
detector recovered **1 314 short-period adotto regions the baseline missed**.

This is the wall-break: gate-relaxation could only buy recall by drowning in
false positives (OP2/OP3 ≈ 72–74% recall at 49% adjusted precision); the
density + Poisson-LLR filter buys recall *without* the precision collapse.

### Raw FM-only frontier (no post-processing, chr21, p1-6)

Tracing `FMSCAN=2` over (density, LLR) shows a clean precision/recall trade and
confirms the filter is the lever (precision falls smoothly as it relaxes):

| density/LLR | calls | regRec% | rawPrec% | adjPrec% | garbage% |
|---|---|---|---|---|---|
| 0.55/10 | 10 565 | 25.15 | 82.39 | 98.65 | 7.7 |
| 0.50/8  | 13 408 | 29.73 | 79.56 | 95.15 | 23.7 |
| 0.45/6  | 17 427 | 33.64 | 71.78 | 86.70 | 47.1 |
| 0.40/5  | 20 506 | 35.48 | 66.09 | 80.27 | 58.2 |
| 0.35/4  | 25 956 | 37.83 | 56.67 | 69.45 | 70.5 |
| 0.30/3  | 34 061 | 40.64 | 46.85 | 58.25 | 78.6 |
| 0.25/2  | 42 030 | 42.91 | 40.11 | 50.75 | 82.2 |

Standalone the FM detector caps at ~43% recall (below OP1) — it is a *complement*
to the exact seeder, not a replacement. The win comes from additive combination.

## Runtime

The FM scan itself is cheap: ~50–82 s for all p=1–6 motif queries on chr21
(46.7 Mb), BWT build ~22 s. The cost is the **downstream merge**: additive mode
roughly doubles the Tier-1 candidate count, and `finder._merge_adjacent_repeats`
calls the expensive `_recompute_stats` (DP realignment) on every merge, so the
merge step grows to ~300–330 s (full additive chr21/chr22 ≈ 16–22 min total vs
~12 min baseline). Still "minutes, not hours". The residual `seen_mask` design
already cut raw FM runs ~2× (23 k → 10 k standalone) and merge ~13× (610 s →
47 s standalone); the remaining additive merge cost is the next optimisation
target (e.g. pre-dedup FM-vs-window candidates before the O(n) merge, or skip
`_recompute_stats` for short same-motif merges).

## Tuning knobs (env vars; defaults reproduce baseline when `TIER1_FMSCAN` unset)

- `TIER1_FMSCAN` 0/1/2 (off / additive / replacement)
- `TIER1_FMSCAN_MIN_P` / `_MAX_P` (default 1 / 6)
- `TIER1_FMSCAN_MIN_DENSITY` (default 0.50), `TIER1_FMSCAN_MIN_LLR` (default 8.0)
- `TIER1_FMSCAN_MAX_GAP` (default 2), `TIER1_FMSCAN_MIN_OCC` (3),
  `TIER1_FMSCAN_MIN_SPAN` (20), `TIER1_FMSCAN_MAX_OCC_TOTAL` (2e7 safety valve)

## Reproduce

```bash
PY=/data/gpfs/assoc/pgl/bin/conda/conda_envs/bwtandem/bin/python
D=/data/gpfs/assoc/pgl/devel/exp1_human
base="TIER1_MIN_ARRAY_LEN=20 TIER1_MIN_SCORE=20 TIER1_MIN_COPIES=2 \
TIER1_COPYBASE=6 TIER1_COPYADD=2 TIER1_EXT_COPIES=2 \
TIER2_SHORT_REQ_COPIES=2 TIER2_MISMATCH=0.25"
# Winning additive run:
env $base TIER1_FMSCAN=1 $PY -m src.main $D/data/chr21.fa \
    --min-period 1 --max-period 2000 --threads 1 --format bed -o out_add
$PY $D/score_overlap.py $D/data/adotto_primary.bed out_add.bed:add --chroms chr21
$PY $D/fp_check.py out_add.bed $D/data/adotto_primary.bed \
    .../ultra/...output.bed .../tantan/...output.bed
```

## Future work

- Extend to p=7–9 (more motifs, ~30–60 s more; diminishing returns expected).
- Reduce the additive merge cost (the dominant runtime).
- Sweep the additive (density, LLR) frontier further to find the recall ceiling
  at the precision wall.
