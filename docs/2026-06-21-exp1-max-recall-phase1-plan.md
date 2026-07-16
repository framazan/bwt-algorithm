# bwtandem Maximal-Recall Phase 1 (genome core) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise bwtandem's adotto region recall from 44.36% toward/past tantan (78%) by fixing the two structural recall blockers — exact-only Tier1 seeding and the under-extending mismatch extender — plus the period-10-20 notch and fragmentation, while tracking precision honestly.

**Architecture:** Six staged, independently-revertible changes to the existing 3-tier pipeline. Pure-Python prototypes validated on real chromosomes via the existing `exp1_human/score_overlap.py` harness before any Cython port/`.so` rebuild. Each change is gated on a recall gain at non-collapsing precision.

**Tech Stack:** Python 3.11 (conda env `bwtandem`), NumPy, optional Cython `_accelerators.pyx`, bedtools for scoring. Tests are standalone scripts using `tests/fixtures/generate_synthetic.py` (no pytest in env).

**Baseline (measured 2026-06-21, comboA = current default):** region recall 44.36% / raw precision 66.41% / bp recall 30.60% genome-wide; period-1-3 within-class recall 26.6%. comboA ALREADY sets COPYBASE=6 COPYADD=2 EXT_COPIES=2, so copy-gate relaxation is already in the baseline — the remaining levers are seeding (C1), extension (C2), period 10-20 (C4), and merge (C5).

**Conventions:**
- Python: `/data/gpfs/assoc/pgl/bin/conda/conda_envs/bwtandem/bin/python` (call as `$PY`).
- comboA env block (export before every run/score):
  `TIER1_MIN_ARRAY_LEN=20 TIER1_MIN_SCORE=20 TIER1_MIN_COPIES=2 TIER1_COPYBASE=6 TIER1_COPYADD=2 TIER1_EXT_COPIES=2 TIER2_SHORT_REQ_COPIES=2 TIER2_MISMATCH=0.25`
- Validation harness dir: `/data/gpfs/assoc/pgl/devel/exp1_human` (call as `$D`). Scorer: `$PY $D/score_overlap.py $D/data/adotto_primary.bed TOOL.bed:name --chroms chr21` (add chr22, then chr1+chr16 for wider gates).
- chr21 local competitor points (re-scored vs adotto): ultra 82.15%R/54.17%P, tantan 77.31%R/58.75%P, trf 34.84%R/96.34%P. chr21 comboA baseline: 52.33%R/65.77%P.
- Only `_accelerators.pyx` edits need a rebuild (see CLAUDE.md build block); pure-Python tier edits take effect immediately.
- **Validation by metric, not byte-identity** — the pipeline is ~0.1% nondeterministic run-to-run.

---

## File Structure

- `src/tier1.py` — C1 (seed stitching), C3b (purity-gated 2-copy). Modify the candidate-building block (`tier1.py:97-124`) and the extend/accept block (`tier1.py:125-205`).
- `src/_accelerators.pyx` — C2 (rolling-consensus extender, `:201-280`); optional C1 hot-path port. Needs `.so` rebuild.
- `src/tier2.py` — C4 (strict-LCP floor `:111,154`; required_copies `:375`).
- `src/bwt_seed.py` — C4 (approximate k-mer seed mode `:98-129`).
- `src/finder.py` — C5 (merge predicate/gap `:288-300`).
- `tests/test_maxrecall_seeding.py` — NEW standalone behavioral tests (C1).
- `tests/test_maxrecall_extender.py` — NEW standalone tests (C2).
- `exp1_human/diag/` — scratch scoring outputs (gitignored, not committed).

---

## Task 0: Validation harness + baseline snapshot + env flag scaffold

**Files:**
- Create: `tests/test_maxrecall_seeding.py` (skeleton runner)
- Use: `tests/fixtures/generate_synthetic.py`, `tests/test_ground_truth.py` helpers

- [ ] **Step 1: Snapshot the chr21 baseline (control) for all later comparisons**

```bash
PY=/data/gpfs/assoc/pgl/bin/conda/conda_envs/bwtandem/bin/python
D=/data/gpfs/assoc/pgl/devel/exp1_human; mkdir -p $D/diag/maxrecall
cd /data/gpfs/assoc/pgl/devel/bwt-algorithm
export TIER1_MIN_ARRAY_LEN=20 TIER1_MIN_SCORE=20 TIER1_MIN_COPIES=2 TIER1_COPYBASE=6 TIER1_COPYADD=2 TIER1_EXT_COPIES=2 TIER2_SHORT_REQ_COPIES=2 TIER2_MISMATCH=0.25
$PY -m src.main $D/data/chr21.fa --min-period 1 --max-period 2000 --threads 1 --format bed -o $D/diag/maxrecall/c0_chr21 -v > $D/diag/maxrecall/c0_chr21.log 2>&1
$PY $D/score_overlap.py $D/data/adotto_primary.bed $D/diag/maxrecall/c0_chr21.bed:C0_baseline --chroms chr21 | tee $D/diag/maxrecall/c0_score.txt
```
Expected: region recall ≈ 52.3%, precision ≈ 65.8% (matches comboA baseline).

- [ ] **Step 2: Write the synthetic-STR test skeleton (diverged short STRs the baseline misses)**

```python
# tests/test_maxrecall_seeding.py
"""Behavioral tests: diverged short STRs must be detected by Tier1.
Run: python tests/test_maxrecall_seeding.py  (exits nonzero on failure)
Mirrors the 'recoverable' class: period 1-3, ~85-97% purity, 8-20 copies.
"""
import os, sys, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fixtures.generate_synthetic import random_dna, make_repeat, write_fasta
from test_ground_truth import run_finder, overlap_ratio, periods_compatible

CASES = [
    # (label, motif, copies, mismatch_rate)
    ("dinuc_AC_div10", "AC", 22, 0.10),
    ("mono_A_div05",   "A",  40, 0.05),
    ("trinuc_CAG_div08","CAG",16, 0.08),
    ("dinuc_AT_div12", "AT", 20, 0.12),
]

def build_case(motif, copies, mm):
    left  = random_dna(300, gc=0.45)
    array = make_repeat(motif, copies, mismatch_rate=mm)
    right = random_dna(300, gc=0.45)
    seq = left + array + right
    return seq, len(left), len(left) + len(array)

def detected(seq, a_start, a_end, motif):
    tmp = tempfile.mkdtemp()
    try:
        fa = os.path.join(tmp, "case.fa"); write_fasta(fa, "case", seq)
        preds = run_finder(fa, enabled_tiers={"tier1"}, min_period=1, max_period=9)
        for p in preds:
            if overlap_ratio(a_start, a_end, p["start"], p["end"]) >= 0.5 \
               and periods_compatible(len(motif), p.get("period", len(motif))):
                return True
        return False
    finally:
        shutil.rmtree(tmp)

def main():
    fails = []
    for label, motif, copies, mm in CASES:
        seq, s, e = build_case(motif, copies, mm)
        ok = detected(seq, s, e, motif)
        print(f"{'PASS' if ok else 'FAIL'}  {label}")
        if not ok: fails.append(label)
    if fails:
        print(f"\n{len(fails)} FAILED: {fails}"); sys.exit(1)
    print("\nALL PASS"); sys.exit(0)

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the test against the BASELINE to confirm it fails (these are the misses)**

Run: `cd /data/gpfs/assoc/pgl/devel/bwt-algorithm && $PY tests/test_maxrecall_seeding.py`
Expected: one or more FAIL (baseline misses diverged short STRs). Record which fail — this is the C1 target. (Check `run_finder`/`overlap_ratio`/`periods_compatible` signatures in `tests/test_ground_truth.py`; adapt the call if `run_finder` needs different kwargs — fix the harness, not the assertions.)

- [ ] **Step 4: Commit the harness**

```bash
git add tests/test_maxrecall_seeding.py
git commit -m "test(#3): synthetic diverged short-STR cases (baseline misses, C1 target)"
```

---

## Task 1: C1 — Mismatch-tolerant short-STR seeding (pure-Python prototype)

**Rationale:** `find_period_runs` / the Python fallback (`tier1.py:97-124`) return only maximal *perfect* adjacency runs. A diverged STR fragments into short perfect sub-runs separated by single mismatches, so no seed reaches threshold. Fix: after collecting perfect-run candidates per `motif_len`, **stitch adjacent same-period candidates separated by a small gap** into one longer seed, then let the existing extend/refine path handle it.

**Files:**
- Modify: `src/tier1.py` candidate block (`tier1.py:97-124`) — add a stitch pass after `candidates` is built (covers BOTH the C-extension and Python-fallback paths, since both fill `candidates`).
- Test: `tests/test_maxrecall_seeding.py`

- [ ] **Step 1: Add an env-gated stitch helper and apply it to `candidates`**

In `src/tier1.py`, immediately after the `if _c_lib ... else ...` block that fills `candidates` (right before `for array_start, array_end, seed_copies in candidates:` at ~`tier1.py:124`), insert:

```python
            # C1: stitch adjacent perfect sub-runs of this period across small
            # interruptions so diverged short STRs (a SNP every few copies) form
            # one seed instead of many sub-threshold fragments. Env-gated.
            stitch_gap = int(os.environ.get("TIER1_STITCH_GAP", "0"))  # in copies
            if stitch_gap > 0 and len(candidates) > 1:
                candidates.sort(key=lambda c: c[0])
                max_gap_bp = stitch_gap * motif_len
                merged = [list(candidates[0])]
                for s, e, c in candidates[1:]:
                    ps, pe, pc = merged[-1]
                    # phase-aligned & close enough -> one diverged array
                    if 0 <= s - pe <= max_gap_bp and ((s - ps) % motif_len == 0):
                        merged[-1][1] = e
                        merged[-1][2] = (e - ps) // motif_len
                    else:
                        merged.append([s, e, c])
                candidates = [tuple(m) for m in merged]
```

Confirm `import os` is present at the top of `tier1.py` (it is — `tier1.py:1`).

- [ ] **Step 2: Run the synthetic test WITHOUT the flag (still baseline) to confirm unchanged**

Run: `$PY tests/test_maxrecall_seeding.py`
Expected: same failures as Task 0 Step 3 (flag defaults off → no behavior change). This proves the change is inert when disabled.

- [ ] **Step 3: Run the synthetic test WITH the flag**

Run: `TIER1_STITCH_GAP=3 $PY tests/test_maxrecall_seeding.py`
Expected: previously-failing diverged cases now PASS (stitching forms a seed that extend/refine accepts). If some still fail, raise `TIER1_STITCH_GAP` to 4-5 and/or check that `extend_with_mismatches` + `refine_repeat` accept the stitched span; if the merged seed is rejected only by `required_threshold`, that is expected to be recovered by C2 — note it and continue.

- [ ] **Step 4: Real-data gate on chr21 (recall must rise, precision must not collapse)**

```bash
export TIER1_MIN_ARRAY_LEN=20 TIER1_MIN_SCORE=20 TIER1_MIN_COPIES=2 TIER1_COPYBASE=6 TIER1_COPYADD=2 TIER1_EXT_COPIES=2 TIER2_SHORT_REQ_COPIES=2 TIER2_MISMATCH=0.25 TIER1_STITCH_GAP=3
$PY -m src.main $D/data/chr21.fa --min-period 1 --max-period 2000 --threads 1 --format bed -o $D/diag/maxrecall/c1_chr21 -v > $D/diag/maxrecall/c1_chr21.log 2>&1
$PY $D/score_overlap.py $D/data/adotto_primary.bed $D/diag/maxrecall/c0_chr21.bed:C0 $D/diag/maxrecall/c1_chr21.bed:C1 --chroms chr21 | tee $D/diag/maxrecall/c1_score.txt
```
Expected: C1 region recall > C0 (target +5pp or more), C1 precision ≥ ~60% (do not accept a drop below ~58%, tantan's floor). Sweep `TIER1_STITCH_GAP` ∈ {2,3,4,5} and record the recall/precision of each in `c1_score.txt`; keep the best recall point with precision ≥ ~60%.

- [ ] **Step 5: Wider gate (chr21+chr22, then chr1+chr16+chr21) to confirm generalization**

```bash
for CHR in chr22 chr1 chr16; do $PY -m src.main $D/data/$CHR.fa ... ; done   # reuse the Task-1 command per chromosome where FASTA exists; chr1/chr16 may need extraction from hg38_primary.fa (samtools faidx) — extract once into $D/data/.
$PY $D/score_overlap.py $D/data/adotto_primary.bed $D/diag/maxrecall/c1_multi.bed:C1 --chroms chr1,chr16,chr21,chr22
```
Expected: recall gain holds across chromosomes; precision ≥ ~60%. (If chr1/chr16 FASTAs are absent, extract with `$PY -c` or samtools; do this once.)

- [ ] **Step 6: Commit C1**

```bash
git add src/tier1.py
git commit -m "feat(tier1,#3): C1 stitch perfect sub-runs (TIER1_STITCH_GAP) to seed diverged short STRs"
```

---

## Task 2: C2 — Rolling-consensus, windowed-break mismatch extender

**Rationale:** `extend_with_mismatches` (`_accelerators.pyx:201-280`) compares every copy to a FIXED first copy (`:218-221`) and breaks on CUMULATIVE mismatch (`:241,255`), so diverged arrays truncate early → median 47.6% span coverage, bp precision 27.9%. Use a rolling per-position consensus and break only on K consecutive bad copies. This raises bp precision (buys precision budget) and lets C1's stitched seeds extend to true boundaries.

**Files:**
- Modify: `src/_accelerators.pyx:201-280` (extender) and `:131-147` (`_max_mismatch_threshold`, remove the period==1 → 0 special case).
- Test: `tests/test_maxrecall_extender.py`
- Rebuild: `.so` (CLAUDE.md build block).

- [ ] **Step 1: Write the extender unit test (diverged array must extend to full span)**

```python
# tests/test_maxrecall_extender.py
"""Tier1 must cover >=90% of a diverged array's true span (extender test).
Run: python tests/test_maxrecall_extender.py
"""
import os, sys, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fixtures.generate_synthetic import random_dna, make_repeat, write_fasta
from test_ground_truth import run_finder, overlap_ratio

def coverage(motif, copies, mm):
    left = random_dna(300); array = make_repeat(motif, copies, mismatch_rate=mm); right = random_dna(300)
    seq = left + array + right; a_s, a_e = len(left), len(left)+len(array)
    tmp = tempfile.mkdtemp()
    try:
        fa = os.path.join(tmp,"c.fa"); write_fasta(fa,"c",seq)
        preds = run_finder(fa, enabled_tiers={"tier1"}, min_period=1, max_period=9)
        best = 0.0
        for p in preds:
            inter = max(0, min(a_e,p["end"]) - max(a_s,p["start"]))
            best = max(best, inter/(a_e-a_s))
        return best
    finally:
        shutil.rmtree(tmp)

def main():
    fails=[]
    for motif,copies,mm in [("AC",30,0.10),("CAG",25,0.08),("AT",28,0.12)]:
        cov = coverage(motif,copies,mm)
        ok = cov >= 0.90
        print(f"{'PASS' if ok else 'FAIL'}  {motif}x{copies}@{mm}  coverage={cov:.2f}")
        if not ok: fails.append((motif,cov))
    if fails: print(f"\nFAILED: {fails}"); sys.exit(1)
    print("\nALL PASS"); sys.exit(0)

if __name__=="__main__": main()
```

- [ ] **Step 2: Run against current extender to confirm it fails (under-coverage)**

Run: `TIER1_STITCH_GAP=3 $PY tests/test_maxrecall_extender.py`
Expected: FAIL with coverage well below 0.90 (current fixed-consensus extender truncates).

- [ ] **Step 3: Implement rolling consensus + windowed break in `_accelerators.pyx`**

In `extend_with_mismatches` (`_accelerators.pyx:201-280`):
1. Build the initial consensus from the seed copy as today, but after each accepted copy, update a per-position majority count and recompute the consensus base for each of the `period` positions (rolling).
2. Replace the cumulative-mismatch break with a sliding window: maintain `consecutive_bad` = number of consecutive copies whose per-copy mismatch fraction exceeds `allowed_rate`; break only when `consecutive_bad >= K` (env `TIER1_EXT_BAD_RUN`, default 2). Extend both directions from the seed.
3. In `_max_mismatch_threshold` (`:131-147`), remove the `if period == 1: return 0` special case so homopolymers tolerate the normal rate.

(Implement with C-typed arrays of length `period` for the per-position base counts; keep the function signature unchanged so callers in `tier1.py:140` and `bwt_seed.py` are untouched.)

- [ ] **Step 4: Rebuild the `.so`**

```bash
cd /data/gpfs/assoc/pgl/devel/bwt-algorithm
$PY -c "
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
ext=[Extension('src._accelerators',['src/_accelerators.pyx'],include_dirs=[np.get_include()])]
setup(script_args=['build_ext','--inplace'],ext_modules=cythonize(ext,compiler_directives={'language_level':'3'}))
"
```
Expected: compiles, produces `src/_accelerators*.so`.

- [ ] **Step 5: Run the extender test to verify it passes**

Run: `TIER1_STITCH_GAP=3 $PY tests/test_maxrecall_extender.py`
Expected: ALL PASS (coverage ≥ 0.90).

- [ ] **Step 6: Real-data gate — region recall AND bp precision must both improve**

```bash
export ...comboA... TIER1_STITCH_GAP=3
$PY -m src.main $D/data/chr21.fa ... -o $D/diag/maxrecall/c2_chr21 ...
$PY $D/score_overlap.py $D/data/adotto_primary.bed $D/diag/maxrecall/c1_chr21.bed:C1 $D/diag/maxrecall/c2_chr21.bed:C2 --chroms chr21 | tee $D/diag/maxrecall/c2_score.txt
```
Expected: C2 region recall ≥ C1 AND C2 **bpPrec% materially higher** than C1 (the key C2 win). Tune `TIER1_EXT_BAD_RUN` ∈ {2,3}.

- [ ] **Step 7: Commit C2**

```bash
git add src/_accelerators.pyx tests/test_maxrecall_extender.py
git commit -m "feat(accel,#3): C2 rolling-consensus windowed-break extender (bp-precision + full-span)"
```

---

## Task 3: C4 — Period 10-20 path

**Files:**
- Modify: `src/tier2.py:111` (strict-LCP `min_unit` floor → env `TIER2_MIN_UNIT`, default 20, allow 12), `src/tier2.py:375` (`required_copies` for period<20), `src/finder.py:143` (`scan_lower`).
- Modify: `src/bwt_seed.py:98-100,129` (approximate 1-mismatch seed mode, env `BWT_SEED_MISMATCH`, default 0=exact).

- [ ] **Step 1: Lower the strict-LCP unit floor behind an env flag**

In `src/tier2.py`, replace the hardcoded `min_unit = max(20, ...)` region (`tier2.py:111`/`finder.py:143` per the diagnosis) with `min_unit = int(os.environ.get("TIER2_MIN_UNIT", "20"))` (default preserves behavior). Ensure `import os`.

- [ ] **Step 2: Make required_copies for short periods env-tunable**

In `src/tier2.py:375`, `required_copies = 2 if period >= 20 else int(os.environ.get("TIER2_SHORT_REQ_COPIES", "3"))` (already partly env-driven; confirm the period<20 branch reads the env).

- [ ] **Step 3: chr21 gate**

```bash
export ...comboA... TIER1_STITCH_GAP=3 TIER2_MIN_UNIT=12
$PY -m src.main $D/data/chr21.fa ... -o $D/diag/maxrecall/c4_chr21 ...
$PY $D/score_overlap.py $D/data/adotto_primary.bed $D/diag/maxrecall/c2_chr21.bed:C2 $D/diag/maxrecall/c4_chr21.bed:C4 --chroms chr21 | tee $D/diag/maxrecall/c4_score.txt
```
Expected: recall up on period-10-20 (overall +2-4pp), precision ≥ ~58%. If precision drops too far, add `max_occurrences` cap tightening in `bwt_seed.py:135` and re-test.

- [ ] **Step 4: (If needed) approximate k-mer seed mode in bwt_seed**

Only if Step 3 leaves the 10-20 band weak: in `src/bwt_seed.py:98-129`, add a `BWT_SEED_MISMATCH=1` mode that, in addition to the exact backward_search, also searches the k-mer's 1-substitution neighbors and unions the occurrence positions before the periodic-run detection. Gate behind the env flag (default exact). Re-run Step 3's gate.

- [ ] **Step 5: Commit C4**

```bash
git add src/tier2.py src/bwt_seed.py src/finder.py
git commit -m "feat(tier2,#3): C4 period 10-20 path (TIER2_MIN_UNIT, approx seeds)"
```

---

## Task 4: C5 — Merge/fragmentation relaxation + purity-gated 2-copy

**Files:**
- Modify: `src/finder.py:288-300` (merge predicate/gap).
- Modify: `src/tier1.py` accept block + `src/tier2.py:375` (allow 2 copies when purity ≥ env `MIN_PURITY_2COPY`, default 0.95).

- [ ] **Step 1: Relax short-period merge**

In `src/finder.py:293`, raise short-period `max_gap` to `int(os.environ.get("MERGE_GAP_COPIES","2")) * period_len` (default reproduces current `max(10, period_len)` — choose the larger). In `finder.py:297`, allow rotation-aware fuzzy canonical merge for short periods too (extend the `len(canon)>=50` branch to short motifs using the existing hamming check at a fixed ≤10% threshold).

- [ ] **Step 2: Purity-gated 2-copy allowance**

In the Tier1 accept block (`tier1.py:154`/`:189-192`) and `tier2.py:375`, when the refined repeat's purity (`1 - mismatch_rate`) ≥ `MIN_PURITY_2COPY` (default 0.95), allow `copies >= 2` instead of `>= 3`/`dynamic_min_copies`. Gate behind the env so default behavior is unchanged.

- [ ] **Step 3: chr21 gate (recall + bp recall up, precision held)**

```bash
export ...comboA... TIER1_STITCH_GAP=3 TIER2_MIN_UNIT=12 MERGE_GAP_COPIES=2 MIN_PURITY_2COPY=0.95
$PY -m src.main $D/data/chr21.fa ... -o $D/diag/maxrecall/c5_chr21 ...
$PY $D/score_overlap.py $D/data/adotto_primary.bed $D/diag/maxrecall/c4_chr21.bed:C4 $D/diag/maxrecall/c5_chr21.bed:C5 --chroms chr21 | tee $D/diag/maxrecall/c5_score.txt
```
Expected: bpRecall up, region recall ≥ C4, precision held ≥ ~58%.

- [ ] **Step 4: Run both synthetic test scripts (regression)**

Run: `TIER1_STITCH_GAP=3 $PY tests/test_maxrecall_seeding.py && TIER1_STITCH_GAP=3 $PY tests/test_maxrecall_extender.py`
Expected: ALL PASS.

- [ ] **Step 5: Commit C5**

```bash
git add src/finder.py src/tier1.py src/tier2.py
git commit -m "feat(#3): C5 short-period fuzzy merge + purity-gated 2-copy allowance"
```

---

## Task 5: C6 — Frontier re-sweep + full-genome validation + honest write-up

**Files:**
- Use: `$D/run_fullgenome.sbatch`, `$D/final_score.sh`, `$D/score_overlap.py`.
- Modify/Create: `docs/2026-06-21-exp1-max-recall-results.md` (new results doc).

- [ ] **Step 1: Re-sweep the new frontier on chr21+chr22**

Sweep the new flags (`TIER1_STITCH_GAP` ∈ {2,3,4}, `TIER2_MIN_UNIT` ∈ {12,15,20}, `MERGE_GAP_COPIES` ∈ {1,2}, plus the existing MIN_ARRAY_LEN/MIN_SCORE) and tabulate (recall, raw precision, bp recall, bp precision) for each on chr21+chr22. Save to `$D/diag/maxrecall/frontier.tsv`.

- [ ] **Step 2: Pick the operating point**

Choose the recall-maximal config whose raw precision is still defensible (≥ ~58%, tantan's floor) AND whose adjusted precision (run `$D/fp_check.py`) stays ≥ ~80%. Record the chosen env block as `comboC` in the results doc.

- [ ] **Step 3: Full-genome run with comboC (SLURM)**

```bash
cd $D
sbatch --mem=96G --export=ALL,OUT=$D/out/bwt_hg38_maxrecall,<comboC env vars> run_fullgenome.sbatch
```
(Bake the comboC env into the `--export` list or edit a copy of the sbatch.) Watch the job to completion; confirm EXIT=0, capture sacct MaxRSS.

- [ ] **Step 4: Genome-wide re-score (all tools, identical GT)**

```bash
$PY $D/score_overlap.py $D/data/adotto_primary.bed \
  $D/out/bwt_hg38_lowmem.bed:bwt_comboA $D/out/bwt_hg38_maxrecall.bed:bwt_comboC \
  $D/score/trf.bed:trf $D/score/ultra.bed:ultra $D/score/tantan.bed:tantan | tee $D/diag/maxrecall/genome_final.txt
$PY $D/fp_check.py $D/out/bwt_hg38_maxrecall.bed $D/data/adotto_primary.bed $D/score/ultra.bed $D/score/tantan.bed
```
Expected: comboC region recall materially > 44.36%; record exact recall/precision vs tantan/ultra and state plainly whether/where Pareto-dominance or recall-superiority is achieved.

- [ ] **Step 5: Write the honest results doc + update CLAUDE.md tuning knobs**

Create `docs/2026-06-21-exp1-max-recall-results.md` with the before/after table (comboA vs comboC vs trf/ultra/tantan), the achieved recall and its precision, peak RAM, runtime, and a frank statement of how close to tantan/ultra recall we got. Add the new env vars (`TIER1_STITCH_GAP`, `TIER1_EXT_BAD_RUN`, `TIER2_MIN_UNIT`, `MERGE_GAP_COPIES`, `MIN_PURITY_2COPY`, `BWT_SEED_MISMATCH`) to the CLAUDE.md tuning section.

- [ ] **Step 6: Commit**

```bash
git add docs/2026-06-21-exp1-max-recall-results.md CLAUDE.md
git commit -m "docs(#3): C6 genome-wide max-recall results (comboC before/after) + env knobs"
```

---

## Self-review notes (gaps acknowledged)

- **C3 copy-gate** is already in the comboA baseline (COPYBASE=6/COPYADD=2/EXT_COPIES=2); only the purity-gated 2-copy *new* behavior is added (Task 4 Step 2). No separate C3 task.
- **Cython port of C1** is deferred: the pure-Python stitch (Task 1) runs on the candidate list (small), so a `.so` port is only needed if profiling shows it dominates runtime at genome scale — add it in C6 if the full-genome runtime regresses notably; otherwise YAGNI.
- **Phase 2 (centromere/CHM13)** is a SEPARATE plan, written after Phase 1 lands (it depends on CHM13 download + Phase 1 outcomes). Not in this plan.
- **Threshold values** (stitch_gap, bad_run, min_unit, gaps) are starting points; the empirical gates in each task tune them. This is expected for an algorithmic task — the gates, not the literals, define success.
- **`run_finder` kwargs**: verify the exact signature in `tests/test_ground_truth.py:203` and adapt the test harness calls if needed (fix harness, never weaken assertions).
