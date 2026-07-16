# Exp1 Region-Recall Maximization Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. NOTE: Task 2 onward is driven by a `/loop` self-paced controller (the agent), using `loop/ledger.tsv` as cross-wakeup memory — see the design at `docs/superpowers/specs/2026-06-23-exp1-recall-loop-design.md`.

**Goal:** Push full-genome adotto region recall toward 82% while holding region precision ≥ 57.66%, by sweeping env levers then recovering true positives in the period-10-20 notch and via merge relaxation.

**Architecture:** A `/loop` self-paced agent submits chr21+chr22 proxy runs via `sbatch`, scores them with the existing `score_overlap.py`, records every result in an append-only ledger, and adaptively picks the next candidate (Phase 1 = env-lever coordinate descent; Phase 2 = test-first code levers). Winners are confirmed full-genome before being reported.

**Tech Stack:** Python 3.11 (conda env `bwtandem`), bedtools, SLURM (`sbatch`). Pure-Python tier edits take effect immediately; no `.so` rebuild in this plan.

## Global Constraints

- **Login node has 4 GB RAM — ALL detection/scoring runs go through `sbatch`.** Never run `src.main` or full-genome scoring on the login node.
- Conda python: `/data/gpfs/assoc/pgl/bin/conda/conda_envs/bwtandem/bin/python` (call as `$PY`).
- Harness dir: `/data/gpfs/assoc/pgl/devel/exp1_human` (call as `$D`); repo: `/data/gpfs/assoc/pgl/devel/bwt-algorithm`. Branch: `perf/exp1-human-sensitivity`.
- Scorer: `$D/score_overlap.py GT.bed TOOL.bed:name --chroms chr21[,chr22]`. Output data line fields: `name regions regRecall% regPrec% bpRecall% bpPrec%`.
- **Hard precision floor: full-genome region precision ≥ 57.66%.** Proxy reads precision ~6–7 pp high → use a calibrated `proxy_floor` (Task 1).
- Current op-point env block (comboChi baseline): `TIER1_MIN_ARRAY_LEN=20 TIER1_MIN_SCORE=20 TIER1_MIN_COPIES=2 TIER1_COPYBASE=6 TIER1_COPYADD=2 TIER1_EXT_COPIES=2 TIER1_FMSCAN=1 TIER2_SHORT_REQ_COPIES=2 TIER2_MISMATCH=0.25` plus the period-gate (`TIER1_SHORT_PERIOD_MAX=4 TIER1_SHORT_MIN_ARRAY_LEN=18 TIER1_SHORT_MIN_SCORE=18`).
- **Validate by metric, not byte-identity** (pipeline is ~0.1% nondeterministic).
- Baseline measured: full-genome recall 57.62% / precision 61.04%.

---

### Task 1: Loop harness scaffold + proxy→genome precision calibration

**Files:**
- Create: `$D/loop/run_proxy.sbatch`, `$D/loop/gt_2122_all.bed`, `$D/loop/ledger.tsv`, `$D/loop/README.md`, `$D/loop/best.json`

**Interfaces:**
- Produces: `run_proxy.sbatch` submitted as `sbatch --export=ALL,TAG=<tag>,KIND=<config|code>,PHASE=<n>,<ENV=val,...> $D/loop/run_proxy.sbatch`; appends exactly one row to `loop/ledger.tsv` with columns `ts tag phase kind source chr21_rec chr21_prec chr22_rec chr22_prec pool_rec pool_prec accepted note`.

- [ ] **Step 1: Build the pooled proxy ground truth**

```bash
D=/data/gpfs/assoc/pgl/devel/exp1_human
mkdir -p $D/loop
cat $D/data/gt_chr21_all.bed $D/data/gt_chr22_all.bed > $D/loop/gt_2122_all.bed
wc -l $D/loop/gt_2122_all.bed   # expect ≈ (gt_chr21_all + gt_chr22_all line counts)
```

- [ ] **Step 2: Write the ledger header**

```bash
printf 'ts\ttag\tphase\tkind\tsource\tchr21_rec\tchr21_prec\tchr22_rec\tchr22_prec\tpool_rec\tpool_prec\taccepted\tnote\n' > $D/loop/ledger.tsv
```

- [ ] **Step 3: Write `$D/loop/run_proxy.sbatch`** (full content)

```bash
#!/bin/bash
#SBATCH --job-name=bwt_proxy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=/data/gpfs/assoc/pgl/devel/exp1_human/loop/logs/proxy_%j.log
set -uo pipefail
REPO=/data/gpfs/assoc/pgl/devel/bwt-algorithm
PY=/data/gpfs/assoc/pgl/bin/conda/conda_envs/bwtandem/bin/python
D=/data/gpfs/assoc/pgl/devel/exp1_human
mkdir -p "$D/loop/logs" "$D/loop/beds"
TAG="${TAG:?set TAG}"; PHASE="${PHASE:-?}"; KIND="${KIND:-config}"; NOTE="${NOTE:-}"
OUT21="$D/loop/beds/${TAG}_chr21"; OUT22="$D/loop/beds/${TAG}_chr22"
cd "$REPO"
echo "START $(date) TAG=$TAG"
env | grep -E '^TIER[12]_' | sort   # record the exact knobs used
"$PY" -m src.main "$D/data/chr21.fa" --min-period 1 --max-period 2000 --threads 2 --format bed -o "$OUT21" -v
"$PY" -m src.main "$D/data/chr22.fa" --min-period 1 --max-period 2000 --threads 2 --format bed -o "$OUT22" -v
cat "${OUT21}.bed" "${OUT22}.bed" > "$D/loop/beds/${TAG}_pool.bed"
S21=$("$PY" "$D/score_overlap.py" "$D/data/gt_chr21_all.bed" "${OUT21}.bed:c21" --chroms chr21 | awk '$1=="c21"{print $3,$4}')
S22=$("$PY" "$D/score_overlap.py" "$D/data/gt_chr22_all.bed" "${OUT22}.bed:c22" --chroms chr22 | awk '$1=="c22"{print $3,$4}')
SP=$("$PY" "$D/score_overlap.py" "$D/loop/gt_2122_all.bed" "$D/loop/beds/${TAG}_pool.bed:pool" --chroms chr21,chr22 | awk '$1=="pool"{print $3,$4}')
TS=$(date -u +%FT%TZ)
SRC=$(env | grep -E '^TIER[12]_' | sort | tr '\n' ';')
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$TS" "$TAG" "$PHASE" "$KIND" "$SRC" $S21 $S22 $SP "pending" >> "$D/loop/ledger.tsv"
echo "LEDGER $TS $TAG c21=[$S21] c22=[$S22] pool=[$SP]"
echo "EXIT=$? END $(date)"
echo "PROXY_DONE"
```

- [ ] **Step 4: Submit the calibration run (current op-point) and wait**

```bash
cd /data/gpfs/assoc/pgl/devel/exp1_human
JID=$(sbatch --parsable \
  --export=ALL,TAG=cal0,PHASE=0,KIND=config,TIER1_MIN_ARRAY_LEN=20,TIER1_MIN_SCORE=20,TIER1_MIN_COPIES=2,TIER1_COPYBASE=6,TIER1_COPYADD=2,TIER1_EXT_COPIES=2,TIER1_FMSCAN=1,TIER1_SHORT_PERIOD_MAX=4,TIER1_SHORT_MIN_ARRAY_LEN=18,TIER1_SHORT_MIN_SCORE=18,TIER2_SHORT_REQ_COPIES=2,TIER2_MISMATCH=0.25 \
  loop/run_proxy.sbatch)
echo "cal0 JOBID=$JID"
```
Expected when done: a `cal0` ledger row; `pool_rec` ≈ 53–56 (chr21/22 are slightly below genome-wide 57.6), `pool_prec` ≈ 64–68.

- [ ] **Step 5: Calibrate the proxy floor and write `best.json`**

```bash
D=/data/gpfs/assoc/pgl/devel/exp1_human
$PY - "$D/loop/ledger.tsv" <<'PY'
import sys, json
rows=[l.rstrip('\n').split('\t') for l in open(sys.argv[1])][1:]
cal=[r for r in rows if r[1]=='cal0'][-1]
pool_prec=float(cal[10])
# genome-wide precision of this op-point is known = 61.04
offset = pool_prec - 61.04
proxy_floor = round(57.66 + offset, 2)
json.dump({"best_tag":"cal0","pool_rec":float(cal[9]),"pool_prec":pool_prec,
           "proxy_prec_offset":round(offset,2),"proxy_floor":proxy_floor,
           "full_validated":{"recall":57.62,"precision":61.04}},
          open("/data/gpfs/assoc/pgl/devel/exp1_human/loop/best.json","w"), indent=2)
print("proxy_floor =", proxy_floor, "(offset", round(offset,2), "pp)")
PY
```

- [ ] **Step 6: Write `$D/loop/README.md`** (ledger schema, how to resume, `proxy_floor` meaning, submit command template) and commit the harness

```bash
cd /data/gpfs/assoc/pgl/devel/bwt-algorithm
git add -A ../exp1_human/loop 2>/dev/null || true   # exp1_human is outside the repo; if so, the harness lives there un-versioned — note its path in the commit
git commit -qm "feat(exp1-loop): proxy harness + ledger + precision calibration" || echo "harness is outside repo tree; recorded in plan instead"
```
(If `$D` is outside the repo working tree, the harness is not git-tracked — that is expected; the ledger itself is the durable record.)

---

### Task 2: Phase 1 — env-lever coordinate descent (driven by the `/loop` controller)

**Files:** none new (submits `run_proxy.sbatch` with different env).

**Procedure (one ledger row per candidate; the controller runs this until the grid is exhausted):**

Baseline = `cal0` env. Vary ONE lever at a time from the current best; accept a candidate iff `pool_rec > best.pool_rec` AND `pool_prec ≥ proxy_floor`; on accept, fold the lever into the baseline and continue. Levers, in order (each value is one submission):

1. `TIER1_FMSCAN_MAX_P` ∈ {7, 8, 9}  — extend FM-scan into period 7-9 (177k regions; this is the cheap p7-9 lever).
2. `TIER2_SHORT_REQ_COPIES` ∈ {2 (base), 1} — admit 2-copy medium repeats (period 10-19 path).
3. `TIER2_MISMATCH` ∈ {0.25 (base), 0.30, 0.33}.
4. `TIER1_SHORT_PERIOD_MAX` ∈ {4 (base), 6, 9} with `TIER1_SHORT_MIN_ARRAY_LEN`/`TIER1_SHORT_MIN_SCORE` ∈ {18 (base), 16}.
5. `TIER1_FMSCAN_MIN_DENSITY` ∈ {0.50 (base), 0.45} and `TIER1_FMSCAN_MIN_LLR` ∈ {8.0 (base), 6.0} — relax FM-scan admission (precision guard: must hold floor).
6. `TIER1_STITCH_GAP` ∈ {0 (base), 1, 2}.

- [ ] **Step 1: Submit the next un-tried candidate**

```bash
cd /data/gpfs/assoc/pgl/devel/exp1_human
sbatch --parsable --export=ALL,TAG=p1_fmscanP7,PHASE=1,KIND=config,<BASELINE_ENV>,TIER1_FMSCAN_MAX_P=7 loop/run_proxy.sbatch
```
(`<BASELINE_ENV>` = the current accepted best env. Substitute the real lever per the grid above.)

- [ ] **Step 2: On completion, mark accept/reject in the ledger + best.json**

```bash
$PY - <<'PY'
import json,sys
D="/data/gpfs/assoc/pgl/devel/exp1_human/loop"
rows=[l.rstrip('\n').split('\t') for l in open(f"{D}/ledger.tsv")]
hdr,rows=rows[0],rows[1:]
best=json.load(open(f"{D}/best.json")); floor=best["proxy_floor"]
last=rows[-1]; prec=float(last[10]); rec=float(last[9])
acc = (prec>=floor) and (rec>best["pool_rec"])
last[11]= "accept" if acc else "reject"
open(f"{D}/ledger.tsv","w").write("\t".join(hdr)+"\n"+"\n".join("\t".join(r) for r in rows)+"\n")
if acc:
    best.update(best_tag=last[1], pool_rec=rec, pool_prec=prec); json.dump(best,open(f"{D}/best.json","w"),indent=2)
print("accept" if acc else "reject", "rec",rec,"prec",prec,"floor",floor)
PY
```

- [ ] **Step 3: Self-pace** — if a job is still running, `ScheduleWakeup` ~270 s; else go to Step 1 with the next candidate. When the grid is exhausted (no untried lever improves at the floor), record the Phase-1 ceiling in `best.json` and proceed to Task 3.

---

### Task 3: Phase 2 — Tier2 period 10-20 approximate seeding (test-first code lever)

**Files:**
- Create: `tests/test_loop_p1020.py`
- Modify: `src/bwt_seed.py` (approximate/spaced k-mer seed mode), `src/tier2.py:394` (required_copies for period 10-19), gated behind a new env flag `TIER2_APPROX_SEED` (default off → baseline unchanged).

**Interfaces:**
- Consumes: `bwt_kmer_seed_scan(text_arr, min_period, max_period, ..., allowed_mismatch_rate, max_occurrences)` (`bwt_seed.py:33`).
- Produces: same signature; behavior change only when `TIER2_APPROX_SEED=1`.

- [ ] **Step 1: Write the failing behavioral test**

```python
# tests/test_loop_p1020.py
"""A period-13 array with a mismatch in most copies must be detected (period 10-20 notch)."""
import os, subprocess, sys, tempfile, random
PY = sys.executable
def synth():
    random.seed(7); motif="ACGTACGTACGTA"  # 13 bp
    copies=[]
    for i in range(40):
        m=list(motif)
        if i%2==0: m[random.randrange(13)]=random.choice("ACGT")  # ~1 SNP/2 copies
        copies.append("".join(m))
    return "".join(copies)
def test_period13_detected():
    seq=">t\n"+synth()+"\n"
    with tempfile.NamedTemporaryFile("w",suffix=".fa",delete=False) as f:
        f.write(seq); fa=f.name
    env=dict(os.environ, TIER2_APPROX_SEED="1", TIER2_SHORT_REQ_COPIES="2")
    out=fa+".bed"
    subprocess.run([PY,"-m","src.main",fa,"--min-period","10","--max-period","20",
                    "--tiers","tier2","--format","bed","-o",fa[:-3]],env=env,check=True)
    lines=[l for l in open(out)] if os.path.exists(out) else []
    assert any(abs((int(l.split(chr(9))[2])-int(l.split(chr(9))[1]))-520)<120 for l in lines), \
        f"period-13 array not detected: {len(lines)} calls"
if __name__=="__main__":
    test_period13_detected(); print("PASS")
```

- [ ] **Step 2: Run it (via sbatch — login node is 4 GB) to confirm baseline FAILS**

```bash
sbatch --wrap="cd /data/gpfs/assoc/pgl/devel/bwt-algorithm && $PY tests/test_loop_p1020.py" --mem=4G --time=00:10:00 -o /tmp/p1020_%j.log
# expect: AssertionError (period-13 not detected) in the log
```

- [ ] **Step 3: Implement the approximate seed mode** in `src/bwt_seed.py` behind `TIER2_APPROX_SEED` (read source at execution time; add a spaced-seed / 1-mismatch k-mer variant that admits periodic runs for period 10-19, keeping the `max_occurrences` cap as the precision guard) and lower `required_copies` use of `short_req_copies` at `tier2.py:394` only when the flag is set.

- [ ] **Step 4: Run the test (sbatch) to confirm PASS**, then run the regression suite

```bash
sbatch --wrap="cd /data/gpfs/assoc/pgl/devel/bwt-algorithm && $PY tests/test_loop_p1020.py && $PY -m pytest tests/test_ground_truth.py -q" --mem=8G --time=00:30:00 -o /tmp/p1020chk_%j.log
# expect: PASS + pytest green (no regressions)
```

- [ ] **Step 5: Proxy gate** — submit `run_proxy.sbatch` with the best Phase-1 env plus `TIER2_APPROX_SEED=1`; accept iff `pool_rec` up AND `pool_prec ≥ proxy_floor` (Task 2 Step 2 logic). 

- [ ] **Step 6: Commit iff accepted**

```bash
git add src/bwt_seed.py src/tier2.py tests/test_loop_p1020.py
git commit -m "feat(tier2): approximate seeding recovers period 10-20 (proxy rec X→Y @prec Z)"
```

---

### Task 4: Phase 2 — short-period merge / fragmentation relaxation (C5, test-first)

**Files:**
- Create: `tests/test_loop_merge.py`
- Modify: `src/finder.py:288-300` (rotation-aware fuzzy merge for short periods; currently fuzzy only for motifs ≥50 bp), behind env flag `MERGE_SHORT_FUZZY` (default off).

- [ ] **Step 1: Write the failing test** — two phase-aligned period-3 calls split by a 4 bp diverged gap must merge into one region.

```python
# tests/test_loop_merge.py
import os, subprocess, sys, tempfile
PY=sys.executable
def test_short_fuzzy_merge():
    # AAT x20, 4bp junk, AAT x20 -> should be ONE merged region with the flag
    seq=">t\n"+("AAT"*20)+"CGCG"+("AAT"*20)+"\n"
    with tempfile.NamedTemporaryFile("w",suffix=".fa",delete=False) as f: f.write(seq); fa=f.name
    env=dict(os.environ, MERGE_SHORT_FUZZY="1")
    subprocess.run([PY,"-m","src.main",fa,"--min-period","1","--max-period","9",
                    "--format","bed","-o",fa[:-3]],env=env,check=True)
    lines=[l for l in open(fa[:-3]+".bed")]
    assert len(lines)==1, f"expected 1 merged region, got {len(lines)}"
if __name__=="__main__": test_short_fuzzy_merge(); print("PASS")
```

- [ ] **Step 2: Run (sbatch) to confirm baseline FAILS** (expect 2 calls).

```bash
sbatch --wrap="cd /data/gpfs/assoc/pgl/devel/bwt-algorithm && $PY tests/test_loop_merge.py" --mem=4G --time=00:05:00 -o /tmp/merge_%j.log
```

- [ ] **Step 3: Implement** the short-period rotation-aware fuzzy merge in `finder.py` behind `MERGE_SHORT_FUZZY` (read source at execution).

- [ ] **Step 4: Run test (sbatch) → PASS + pytest regression green.**

- [ ] **Step 5: Proxy gate** (best env + accepted Task-3 flags + `MERGE_SHORT_FUZZY=1`); accept iff `pool_rec` up AND `pool_prec ≥ proxy_floor`.

- [ ] **Step 6: Commit iff accepted.**

```bash
git add src/finder.py tests/test_loop_merge.py
git commit -m "feat(finder): short-period fuzzy merge reduces fragmentation (proxy rec X→Y @prec Z)"
```

---

### Task 5: Frontier selection + full-genome validation + honest write-up

**Files:** Modify `$D/benchmarking_results_updated.md` (or `filip_repro/` copy), `CLAUDE.md` (tuning knobs), `loop/best.json`.

- [ ] **Step 1: Submit the best accepted config full-genome**

```bash
cd /data/gpfs/assoc/pgl/devel/exp1_human
sbatch --export=ALL,OUT=$D/out/bwt_hg38_loopbest,<BEST_ENV> run_fullgenome.sbatch
```

- [ ] **Step 2: Score genome-wide vs adotto + assert the floor**

```bash
sbatch --wrap="$PY $D/score_overlap.py $D/data/adotto_primary.bed $D/out/bwt_hg38_loopbest.bed:loopbest" --mem=16G --time=02:00:00 -o $D/loop/logs/fullscore_%j.log
# record regRecall%, regPrec%; REQUIRE regPrec% >= 57.66 to claim the point
```

- [ ] **Step 3: Stop decision** — if `regRecall ≥ 82` at `regPrec ≥ 57.66` → success. Else if Phase-1 ceiling reached AND Tasks 3–4 exhausted with ≥3 no-improvement iterations → record the frontier. Update `best.json` with the validated genome-wide numbers.

- [ ] **Step 4: Honest write-up** — add a before/after row to the benchmark doc (v2 baseline 57.62/61.04 → loop-best recall/precision), note which levers paid off, update CLAUDE.md tuning-knob section with the new defaults. Commit.

```bash
cd /data/gpfs/assoc/pgl/devel/bwt-algorithm
git add CLAUDE.md docs/
git commit -m "docs(exp1): recall-loop results — recall 57.62%→X% at Y% precision"
```

---

## Self-Review

- **Spec coverage:** objective/floor (Global Constraints + Task 5 Step 2), staged config→code (Tasks 2 vs 3–4), harness+ledger+best.json (Task 1), proxy-floor calibration (Task 1 Step 5), period 10-20 lever (Task 3), FM-scan p7-9 (Task 2 lever 1), merge C5 (Task 4), full-genome validation + honest report + stop conditions (Task 5) — all covered.
- **Placeholder scan:** code levers (Tasks 3–4) specify the test, the file/function targets, the env gate, and the acceptance rule; the implementation diff is produced via TDD after reading current source (honest — the exact patch depends on the live code, not guessable). `<BASELINE_ENV>`/`<BEST_ENV>` are explicit substitution points carried in the ledger, not vague TODOs.
- **Type consistency:** ledger column order is identical in Task 1 Step 3 (writer) and Task 2 Step 2 (reader); `proxy_floor`/`pool_rec`/`pool_prec` keys in `best.json` consistent across Tasks 1, 2, 3, 5.
