# Design — Honest accelerator fallbacks, parity testing, and the safe refactor bands

_Written 2026-07-09. Supersedes nothing; complements
`docs/2026-07-04-refactoring-opportunities.md` (the 30-item catalog), from which
Bands A/B/C and item F0 are drawn._

---

## 1. Problem

### 1.1 The defect (measured, not inferred)

When the compiled Cython extension `src/_accelerators*.so` is absent, the tool
**silently produces near-empty results and exits 0**. Measured on this repo at
`dfcdbcb`, over the five synthetic fixtures in `tests/fixtures/`:

| fixture | calls with `.so` | calls without `.so` | truth regions |
|---------|-----------------:|--------------------:|--------------:|
| synth_tier1 | 9 | 7 | 9 |
| synth_tier2 | 8 | **0** | 9 |
| synth_tier3 | 8 | **0** | 9 |
| synth_mixed | 9 | 4 | 10 |

Tier 2 and Tier 3 detect **nothing**. There is no error, no warning on stderr,
and no non-zero exit status. A user who installs without a C compiler gets a
BED file that looks plausible and is wrong.

### 1.2 Why the test suite does not catch it

`pytest` is green in the dev environment because the `.so` is present there
(V1: 44 passed). Without the `.so`, the nine Tier-2/Tier-3 ground-truth tests
are gated behind `@NEEDS_CYTHON` and **skip**, hiding the total collapse.
What is left over is not even green:

```
V2 (pytest, .so absent):  1 failed, 34 passed, 9 skipped
  FAILED tests/test_ground_truth.py::TestTier1GroundTruth::test_sensitivity
    Tier 1 sensitivity 87.5% < 95%. Missed: [('TTTC', 53252)]
```

That failure has presumably existed unobserved for as long as the skip markers
have. The `.so`-absent build is an untested configuration that the package
nevertheless advertises as supported (`accelerators.py` docstring: "Optional
accelerators"; `CLAUDE.md`: "pure-Python fallbacks in `accelerators.py` are
used").

### 1.3 Root cause, narrowed

Exactly **five** accelerator symbols are imported anywhere in `src/`:

| symbol | importers | fallback quality |
|--------|-----------|------------------|
| `lcp_tandem_candidates` | `tier2` | faithful pure-Python ✅ |
| `anchor_scan_boundaries` | `tier3` | faithful pure-Python ✅ |
| `align_unit_to_window` | `motif_utils` | returns `None`, but `motif_utils` has its own Python DP body ✅ |
| **`extend_with_mismatches`** | `tier1`, `tier2`, `bwt_seed` | **`return None` — degenerate** ❌ |
| **`find_periodic_runs`** | `bwt_seed` | **`return []` — degenerate** ❌ |

Only those two stubs cause the collapse. Every other symbol defined in
`accelerators.py` (`hamming_distance`, `pack_sequence`, `scan_unit_repeats`,
`scan_simple_repeats`, `find_periodic_patterns`, `find_tandem_runs`) has **zero
importers** — they are dead wrappers (catalog item A2).

Reading `_accelerators.pyx`, both degenerate functions are plain scans with no
C-specific semantics, so both admit an **exactly equivalent** Python/NumPy port:

- `find_periodic_runs` — an O(k) single pass over adjacent gaps in a sorted
  position array, emitting `(run_start, run_end, period)` where
  `period = int(last_diff + 0.5)`.
- `extend_with_mismatches` (default branch) — fix `consensus = s[start:start+period]`,
  extend right then left one whole period at a time while
  `_total_mismatches(span) <= _max_mismatch_threshold(period, copies, rate)`
  (only evaluated when the incoming copy is imperfect), then a partial exact
  extension on both flanks. Helpers: `_hamming_distance` (plain byte compare),
  `_max_mismatch_threshold` (`ceil(clamp(rate, 0.01, 0.5) * period * copies)`,
  floored at 1), `_total_mismatches` (sum of per-copy Hamming vs consensus,
  stopping at `n`).

### 1.4 Two smaller defects found alongside

- On import failure, `accelerators.py:24` does `print(f"DEBUG: importing _accelerators failed: {e}")`
  — to **stdout**, which corrupts any piped output.
- The same path silently attempts a `pyximport` runtime compile, which on a
  machine without `gcc` fails after doing real work (observed:
  `distutils.compilers.C.errors.CompileError: command 'gcc' failed`).

### 1.5 What this design deliberately does *not* touch

Detection sensitivity. The recall/precision frontier on the adotto benchmark is
closed: `exp1_human/loop/resume.md` records four independent negative results
(Phase 6 composition/excess gate; Phase 7 phase/period/profile coherence, all
discriminators AUC ≈ 0.5; Phase 8 period/copies/length band-gating; Phase 10
boundary refinement, ruled out at a computed +5–6pp ceiling before
implementation). Every change below is **behavior-preserving on the `.so` path**
and is verified as such.

---

## 2. Goals

1. **A build without the `.so` must produce the same answers as one with it.**
   Not "approximately" — byte-identical BED.
2. **That property must be enforced by a test**, so it cannot silently regress.
3. **The `.so` path must not change at all.** Byte-identical BED before/after,
   on fixtures and on real chromosomes.
4. Land the catalog's zero-to-low-risk cleanup (Bands A, B, C) on top, now that
   (2) provides a safety net that `pytest` alone did not.

Non-goals: performance of the fallback path (correct-and-slow beats
fast-and-empty); the rolling extender; Band D/E/F.

---

## 3. Design

### 3.1 `src/accelerators.py` — rewrite

**Native detection.** Keep the `try: from . import _accelerators` import. Remove
the `pyximport` runtime-compile attempt (surprising, slow, and it fails anyway
without a toolchain). Replace the stdout `print` with a single
`warnings.warn(...)` at import.

**Escape hatch.** Honour `BWT_DISABLE_NATIVE=1` to force the pure-Python path
even when the `.so` is importable. This is what makes the parity test possible
without renaming files, and it lets users A/B a suspected accelerator bug.

**Faithful ports.** Implement `extend_with_mismatches` and `find_periodic_runs`
in Python/NumPy, transcribing `_accelerators.pyx` statement for statement.
Points that must be preserved exactly:

- `_max_mismatch_threshold`: `period <= 0 or copies <= 0 → 0`; clamp `rate` into
  `[0.01, 0.5]`; `threshold = ceil(rate * period * copies)`; floor at `1`.
  (Note the `.pyx` comment: the old `period == 1 → 0` special case was
  **removed**; homopolymers get the normal rate. Do not reintroduce it.)
- `_total_mismatches`: iterate `copies = (end - start) // period` whole copies,
  `break` when `copy_end > n`. Vectorised as a `(copies, period)` reshape +
  compare against consensus, with `copies` clipped to those that fit under `n` —
  identical result, no Python inner loop.
- `extend_with_mismatches` returns `(array_start, array_end, copies, full_start, full_end)`
  or `None`; the mismatch budget is only consulted when the candidate copy is
  imperfect (`new_mm > 0`), so a run of perfect copies always extends.
- `find_periodic_runs`: the run flush at a gap outside `[min_period, max_period]`
  resets state, and the tolerance is `max(1.0, last_diff * tolerance_ratio)`.
  `int(last_diff + 0.5)` truncates toward zero in both C and Python, so the
  degenerate `last_diff == -1.0` flush maps to `0` in both.

**`TIER1_EXT_ROLLING`.** The `.pyx` reads this env var at import and switches
`extend_with_mismatches` to a different algorithm (`_extend_rolling`). It has
**no callers, no benchmark script, and no documentation** outside the `.pyx`
(grep-confirmed), and defaults to `0`. The fallback implements the default
branch only and raises a clear `NotImplementedError` if `TIER1_EXT_ROLLING=1`
is requested without the `.so`. Failing loudly beats silently running a
different algorithm.

**Fallback taxonomy (catalog F0).** Every fallback carries a docstring line
saying which it is: `FAITHFUL FALLBACK` (bit-identical to the C path) or
`DEGRADED` (if any survive — after this change, none do).

**Dead wrappers (catalog A2).** Delete `hamming_distance`, `pack_sequence`,
`scan_unit_repeats`, `scan_simple_repeats`, `find_periodic_patterns`,
`find_tandem_runs` from *both* branches, and drop the unused `find_tandem_runs`
import in `bwt_seed.py:17`. The `.pyx`/`.so` symbols stay; no rebuild needed.

### 3.2 Tests

**`tests/test_accel_parity.py` (new) — the load-bearing test.**
For each fixture, run the full `TandemRepeatFinder` twice in subprocesses — once
normally, once with `BWT_DISABLE_NATIVE=1` — and assert the emitted BED is
**byte-identical**. Subprocesses are required because `accelerators` binds its
symbols at import. Skip the whole module (not silently — with a reason) when the
`.so` is absent, since there is then only one path to compare.

**`tests/test_ground_truth.py`.** Remove the nine `@NEEDS_CYTHON` skips. With
faithful fallbacks the Tier-2/3 cases must pass in both configurations. This
converts V2 from `1 failed, 34 passed, 9 skipped` into a real gate.

### 3.3 Bands A, B, C (from the catalog)

Applied as written there, honouring its "do NOT do" list. Summary:

- **A** (dead code, ~-260 lines): A1 seven unreferenced `MotifUtils` statics;
  A2 (subsumed by §3.1); A3 four dead `BWTCore` methods; A4 orphaned
  `src/utils.py`; A5 `main.py` dead `out_file` + stale `tier2_profile.prof`
  name; A6 `models.py` unreachable `hasattr` branch; A7 `bwt_seed` double sort;
  A8 `_detect_satellite_period` dead `best_score`; A9 `tier1`
  `perfect_length`/`seed_length` alias; A10 tautological comments + stale
  docstrings.
- **B** (finish the `autocorr.py` consolidation): B1 two inline autocorrelations
  in `finder.py` → `autocorr_identity` / `contiguous_true_runs`; B2 the three
  string-based autocorrelation loops in `motif_utils.py` → one allocation-free
  local helper (**not** the array primitive — they carry break/threshold logic).
- **C** (shared helpers): C1 sentinel-strip ×4 (only the three sites that
  already strip — tier1/tier3 deliberately do not); C2 coverage-mask ×6 →
  `intervals_to_mask`; C3 Shannon entropy str/uint8; C4 `_has_invalid_char`
  (keep the `$`/`N` and ACGT-only strictnesses **distinct**).

A5's profiler rename (`{prefix}.tier2_profile.prof` → `{prefix}.profile.prof`)
is the one user-visible path change; it is intentional (profiling now covers all
tiers) and is called out in the commit message.

Band D (env-var parsing) needs a human decision on empty-string semantics and is
out of scope. Bands E/F are out of scope.

---

## 4. Verification

| tier | command | pre-change | required post-change |
|------|---------|-----------|----------------------|
| V1 | `pytest tests/ -q` with `.so` | 44 passed | all passed, 0 failed |
| V2 | `pytest tests/ -q` with the `.so` removed | 1 failed / 34 passed / 9 skipped | all passed, **0 ground-truth skips** |
| V3-lite | 5 fixtures × {bed, trf, vcf, strfinder} × 6 env configs | md5 baseline captured at `dfcdbcb` | **byte-identical** |
| V3 | chr22 under the catchH gate base, `PYTHONHASHSEED=0` + non-zero `MALLOC_PERTURB_` | clean-worktree run at `dfcdbcb` | **byte-identical BED** |

> **V3 was redesigned mid-flight.** The original criterion — "chr21+chr22 BED
> byte-identical" — turned out to be **unachievable**: two runs of the *same*
> commit from the *same* worktree differ (pool 131 735 vs 131 736 rows, 67
> sorted-diff lines). The caller reads uninitialised heap memory in
> `align_accel.c`, so its output depends on the allocator, and with
> `MALLOC_PERTURB_=0` even on the directory the repo sits in. See
> `docs/2026-07-09-nondeterminism-uninitialised-ptr-table.md`.
>
> A **non-zero** `MALLOC_PERTURB_` makes that garbage a constant and restores
> determinism, so V3 compares old and new under a pinned
> `PYTHONHASHSEED=0` + `MALLOC_PERTURB_∈{1,255}`, with a same-code /
> different-path control to confirm the pinning removed the path dependence.
> chr22 alone is used (≈18 min) — it carries almost all of the variance.

V3 runs `src/main.py` with the exact `x_*` ledger environment —
`TIER1_FMSCAN=1`, `TIER1_SHORT_PERIOD_MAX=9`, `TIER2_MISMATCH=0.30`,
`CATCHALL_SCAN=1 CATCHALL_MIN_IDENTITY=0.72` — chosen because it exercises the
widest set of code paths (Tier 1 FM-scan, Tier 2, Tier 3, the satellite gap-fill,
and the catch-all periodicity pass) in one run. Both sides run from detached
`git worktree`s so the live tree can be edited concurrently, and so the compared
binaries cannot drift.

Byte-identity of the BED **under pinned allocator behaviour** is the acceptance
criterion. The synthetic fixtures never reach the uninitialised cells, so
V1/V2/V3-lite are unaffected and remain exact.

### Results

| check | result |
|-------|--------|
| V1 — `pytest` with the `.so` | **81 passed** (was 44) |
| V2 — `pytest` with the `.so` removed | **47 passed**, 0 failed, only native-only tests skipped (was 1 failed / 34 passed / 9 skipped) |
| V3-lite — 5 fixtures × 4 formats × 6 env configs, both accelerator paths | **byte-identical** to the `dfcdbcb` baseline |
| V3 control — same commit, two different worktrees, `MALLOC_PERTURB_=255` | **byte-identical** (pinning works) |
| **V3 — `dfcdbcb` vs `8838507`, chr22, `MALLOC_PERTURB_=255`** | **byte-identical**, 66 907 rows |
| **V3 — same, `MALLOC_PERTURB_=1`** | **byte-identical** |

Two independent garbage constants agree, and they agree with each other
(`0x00` and `0xfe` are both non-opcodes, so the traceback stops identically).
The refactor changes no output.

Also measured, and reported separately: with `MALLOC_PERTURB_=0` the *real* heap
garbage occasionally reads as an opcode, adding **12 spurious calls on chr22**
relative to the pinned runs.

---

## 5. Risks

- **A faithful-looking port that is not.** The mitigation is the parity test
  plus V3 byte-identity, not code review. If the fallback and the `.so` disagree
  anywhere in ~100 Mb of real sequence, V3 catches it.
- **Vectorising `_total_mismatches` changes a boundary.** The `copy_end > n`
  break is replaced by clipping the copy count; these coincide because every
  caller passes `end <= n`. The reshape must therefore be over `copies_fit =
  min(copies, (n - start) // period)`.
- **Removing `pyximport` breaks someone's workflow.** It cannot currently
  succeed in the documented environments (no in-tree build step invokes it, and
  it needs a compiler that, if present, would have produced the `.so` anyway).
  Removal is noted in the commit message.
- **Band A/B/C touch `finder.py`'s satellite path**, which carries the CHM13
  92.5→99.8% bp-recall operating point. `tests/test_satellite_gapfill.py` plus
  V3 cover it.

---

## 6. Sequencing

Three commits, each independently verifiable:

1. `fix(accelerators): faithful pure-Python fallbacks + parity test` — §3.1, §3.2.
   Gate: V1, V2, V3-lite.
2. `refactor: dead-code sweep (Band A)` — §3.3 Band A. Gate: V1, V2, V3-lite.
3. `refactor: shared helpers (Bands B, C)` — §3.3 Bands B/C. Gate: V1, V2,
   V3-lite, `test_satellite_gapfill`.

V3 (the chr21+chr22 proxy) runs once against the final tree and is the merge
gate for all three.
