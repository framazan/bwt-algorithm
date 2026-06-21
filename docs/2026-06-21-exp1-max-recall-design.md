# Exp1 follow-up — Maximal-recall push for publication (design)

Issue: #3 (continuation). Branch: `perf/exp1-human-sensitivity`.
Date: 2026-06-21. Status: design approved, pre-implementation.

## Goal

Make bwtandem **publication-competitive on recall** against the current de novo
tandem-repeat callers (ultra, tantan, trf) on the adotto GRCh38 benchmark, and
add a **centromere/satellite strength** on CHM13/T2T.

Decision (user): pursue the **maximal-recall** strategy — push region recall
toward / past tantan's 78% (ultra's 82% is the stretch) — accepting precision
risk and uncertain success, and **report the honest operating point** wherever
it lands. Execute **genome core first, centromere second** (staged, each step
validated before the next).

## Where we start (measured, 2026-06-21)

Genome-wide, full adotto v1.2.1 catalog (1,784,804 regions), primary chr:

| tool | region recall | raw region precision | bp recall |
|---|---|---|---|
| bwtandem (comboA, current) | 44.36% | 66.41% | 30.60% |
| trf | 31.88% | 94.86% | 30.26% |
| ultra | 81.62% | 53.65% | 38.14% |
| tantan | 78.00% | 57.66% | 23.54% |

Peak RAM already reduced 149 GB → 29.63 GB (commit `40d4f2d`), so full-genome
and CHM13 runs are feasible.

## Diagnosis (grounds every change below)

Genome-wide miss analysis (993,045 missed regions; 813,008 = 81.9% recoverable,
i.e. also found by ultra OR tantan):

- **The gap is short-period STRs.** Recoverable by period: 1-3 bp = 38.7%
  (314,610 regions, within-class recall only **26.6%**); 4-6 = 28.0%; 10-20 =
  19.3%; period ≥21 already 80-94% within-class recall (not the problem).
- **The discriminator is copy number, not purity.** Period-1-3 recoverable vs
  hit STRs have identical purity (~83% in the 85-97 band) but differ in copies:
  56% of recoverable have <8 copies (median 7.0) vs 26% of hits (median 13.2).
  Spans are normal (median 65 bp; only 0.1% below the 26 bp array-length gate).
- **"Must-fix" set** (regions hit by ultra AND tantan AND trf, missed by
  bwtandem) = 142,669 regions = 14.4% of misses; recovering it lifts recall
  ~+8 pp at maximal confidence. Dominated by period-2 dinucleotides (~50k) and
  period 10-20 (~54k).
- **Fragmentation is as large a problem as outright misses.** Of the 346,298
  all-3-confirmed regions bwtandem *does* touch, median coverage is only 47.6%
  of the GT span (96.9% covered <90%). bp precision is 27.9% (vs ultra 64%) —
  many calls overlap a region but are mis-bounded/under-extended.
- **Tuning alone cannot close the gap.** The full chr21 env-var sweep tops out
  at comboB 54.89% recall / 62.61% precision; pushing further (len16sc16)
  collapses precision to 49.65%. ~27 pp to ultra is **unreachable by tuning** —
  the ceiling is what the tiers can *find*, not what thresholds admit.

### Structural root causes (code)

- Tier1 seeds **only on exact adjacent-copy matches** (`tier1.py:99`
  `match_arr = text_arr[:n-k]==text_arr[k:n]`, C `find_period_runs`
  `tier1.py:91`). A short STR with a SNP in (nearly) every copy never forms a
  2-copy perfect run, so it never enters the candidate list.
- mono/di/tri STRs require **5 perfect seed copies** before any mismatch
  extension (`ext_copies_short=5`, `tier1.py:139`).
- `dynamic_min_copies = max(min_copies, copy_base//motif_len + copy_add)` =
  15/9/7 copies for motif_len 1/2/3 (`tier1.py:77`; copy_base=12, copy_add=3).
- The mismatch extender uses a **single fixed first copy as consensus** and a
  **cumulative-mismatch break** (`_accelerators.pyx:218-221,241,255`), so
  drifting/diverged arrays truncate early; period==1 gets zero tolerance
  (`_accelerators.pyx:131-147`).
- Period **10-20 is a notch**: Tier1 caps motifs at 9 bp (`tier1.py:64`); Tier2
  floors `min_period` at 10 (`tier2.py:64`) and its strict-LCP path only starts
  at 20 (`tier2.py:111`), leaving 10-19 to the weaker exact-k-mer seed scan
  (`bwt_seed.py:99-129`).
- Post-merge requires **identical canonical motif** and fuzzy merge only for
  motifs ≥50 bp (`finder.py:296-299`); short diverged fragments stay split.

## Success criteria

- **Primary:** maximize adotto region recall toward/past tantan (78%); report
  the achieved recall and its precision honestly (no GT overfitting, no
  selective reporting).
- **Guardrail metrics tracked at every step:** region recall, raw region
  precision, adjusted/consensus precision (adotto OR ultra/tantan), bp recall,
  bp precision, peak RAM, runtime.
- **Operating point** chosen from a re-swept frontier after the structural
  changes land — the recall-maximal point whose precision is still defensible.
- **Centromere (Phase 2):** equal-or-better satellite bp recall at competitive
  precision on CHM13 centromeres, with FM-index scaling as the efficiency
  argument.

## Plan — Phase 1: genome core (the recall engine)

Each component validated chr21 → chr21+chr22 → chr1+chr16+chr21 (via
`exp1_human/score_overlap.py`) before the next; each independently revertible
(env flag where feasible). Pure-Python prototypes first; Cython hot-path port +
`.so` rebuild only once logic is validated.

- **C1 — Mismatch-tolerant short-STR seeding in Tier1** (the lever; targets
  period 1-9 imperfect arrays, ~621k recoverable). Replace the exact
  adjacent-copy seed (`tier1.py:91-115`, `:99`) with a "stitch perfect sub-runs
  across ≤1 mismatch/copy" detector so 92-97% purity arrays with a SNP every
  3-4 copies form one seed instead of zero. Gate output with existing
  MIN_SCORE/entropy. Expected +8 to +12 pp.
- **C2 — Rolling-consensus, windowed-break extender** (targets fragmentation;
  raises bp precision "for free"). In `extend_with_mismatches`
  (`_accelerators.pyx:201-280`): running per-position majority consensus instead
  of the fixed first copy (`:218-221`); break on K consecutive bad *copies*
  rather than cumulative mismatch (`:241,255`); remove period==1 zero-tolerance
  (`:131-147`). Expected +2 to +4 pp recall and a large bp-precision lift that
  buys precision budget for C1/C3. Requires `.so` rebuild.
- **C3 — Copy-gate relaxation + high-purity 2-copy allowance** (pure tuning).
  copy_base 12→6, copy_add 3→2 (→ 8/5/4 copies for len 1/2/3, `tier1.py:45,77`);
  ext_copies_short 5→3 (`tier1.py:48,139`); allow 2 copies when measured purity
  ≥95% (`tier1.py:81/154`, `tier2.py:375`). Keep MIN_ARRAY_LEN=26 (only 0.1% of
  targets fall below it). Expected +3 to +5 pp.
- **C4 — Period 10-20 path** (fills the notch; ~54k must-fix). Extend Tier2
  strict-LCP down to min_unit ~12 (`tier2.py:111,154`); lower required_copies for
  period<20 and SHORT_REQ_COPIES (`tier2.py:375`); add a 1-mismatch / spaced
  k-mer seed mode in `bwt_seed.py:99-129` with the `max_occurrences` cap
  (`bwt_seed.py:135`) as the precision guard. Expected +3 to +4 pp.
- **C5 — Merge/fragmentation relaxation** (recall + bp recall). Rotation-aware
  fuzzy canonical-motif merge for short periods (currently ≥50 only,
  `finder.py:297`) and wider short-period gap (`finder.py:293`). Expected +1 to
  +2 pp. Low risk.
- **C6 — Re-sweep + genome validation.** After C1-C5 land, re-run the env-var
  threshold sweep to find the new frontier; pick the recall-maximal point with
  defensible precision; run full GRCh38; re-score all tools on the identical GT;
  commit an honest before/after.

Cumulative expectation: region recall in the **mid-to-high 60s%** at ≥~60%
precision, with an attempt to push past tantan's 78% if C1/C4 deliver upper
estimates. The honest result is reported wherever it lands.

## Plan — Phase 2: centromere / CHM13 (differentiator)

- **Data:** download CHM13v2.0 (T2T) assembly + CenSat centromeric-satellite
  annotation and alpha-satellite HOR/SF tracks (ground truth for the
  repeat-rich regions GRCh38 lacks/masks).
- **Algorithm:** multi-anchor, mismatch-tolerant Tier3 seeding — sample several
  shorter anchor k-mers per period and accept periodic runs from any anchor
  (replacing the single exact-20-mer seed, `bwt_seed.py:129`,
  `tier3.py:145-158`); de-gate `_fill_satellite_gaps` from
  proximity-to-existing-call (`finder.py:443-454`) so unseeded high-divergence
  HSat arrays still recover via autocorrelation.
- **Scoring:** run with Tier3 enabled (period up to 100 kb); score region + bp
  recall/precision vs CenSat/HOR BEDs, stratified by satellite family
  (alpha-HOR / HSat / monomeric); report HOR periodicity correctness (≈171 bp
  alpha monomer + higher-order multimer) and runtime/memory vs trf/ultra on a
  full centromere.
- **Win condition:** equal-or-better satellite bp recall at competitive
  precision, with FM-index scaling as the efficiency argument.

## Validation & testing discipline

- **TDD** for the two new algorithmic units (C1 seeder, C2 extender): unit tests
  on synthetic STR arrays with known period, copy number, and divergence
  (per-copy SNPs, indels, interruptions) asserting the seed/extension boundaries.
- **Regression by metric, not byte-identity.** The pipeline is ~0.1%
  nondeterministic run-to-run (old vs old BEDs differ); validate by region/bp
  recall and precision stability, not exact diffs.
- **Staged scoring** chr21 (fast) → chr1+chr16+chr21 → full genome, tracking all
  guardrail metrics; temp artifacts under `exp1_human/diag/`.
- **Memory/runtime** re-checked at genome scale (must stay feasible; ideally near
  the current 29.63 GB).

## Risks & mitigations

- **May not reach 78%.** Report the honest operating point; the Pareto-optimal /
  best-precision framing remains a valid fallback narrative.
- **Precision erosion from aggressive recall.** Mitigated by C2 (bp precision
  up), the opt-in entropy gate, `max_occurrences` caps, and frontier selection
  in C6; track adjusted/consensus precision too (adotto is incomplete — ~half of
  "new" calls are ultra/tantan-confirmed).
- **Cython rebuilds (C2, possibly C1).** Prototype in pure Python first; only
  rebuild `_accelerators.pyx` once logic is validated.
- **Centromere runtime/precision on megabase arrays.** Gate with
  `anchor_match_pct` and autocorrelation identity; lean on anchor-based boundary
  verification rather than full DP.

## Reproduce / harness

Scoring + sweeps live outside the repo at `/data/gpfs/assoc/pgl/devel/exp1_human`
(`score_overlap.py`, `final_score.sh`, `run_fullgenome.sbatch`, `data/`,
reference BEDs at `/data/gpfs/assoc/pgl/filip/bwtandem_results/beds/`). comboA
env defaults are the current operating point; see
`docs/2026-06-20-exp1-human-sensitivity.md` for the tuning knobs and prior
results.
