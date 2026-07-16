# Refactoring Opportunities — BWT Tandem Repeat Finder (`src/`)

_Compiled 2026-07-04. Read-only audit of the whole `src/` tree by four parallel
surveys (cross-cutting, core files, tier files, support modules). It follows the
per-file mechanical cleanup already landed in the working tree on
`finder.py` / `motif_utils.py` / `tier1.py` / `tier2.py` (dead locals/imports/
attributes/methods, `_min_rotation_int`, `align_repeat_region` redundant-recompute
removal, `tier2._record_repeat` consolidation, entropy short-circuit inline)._

> **Status 2026-07-09 — Bands A, B, C and F0 are LANDED.** See
> `docs/superpowers/specs/2026-07-09-accelerator-fallback-parity-design.md`.
> F0 turned out to be far more serious than the LOW/Effort-S rating below: the
> two degenerate stubs made a build without the `.so` detect **zero** Tier-2 and
> Tier-3 repeats while exiting 0. They were replaced with faithful ports rather
> than a warning, and `tests/test_accel_parity.py` now pins the two accelerator
> paths to byte-identical output — which is what made Bands A/B/C safe to land.
>
> | Band | Items | Commit |
> |------|-------|--------|
> | F0 + A2 | faithful fallbacks, dead wrappers, stdout→stderr, `BWT_DISABLE_NATIVE` | `71704fe` |
> | A | A1, A3–A10 | `f341151` |
> | B, C | B1, B2, C1–C4 (C5 skipped: marginal) | `b2bf5c4` |
>
> **Remaining:** Band D (needs the empty-string decision), Bands E and F.
> The `src/` tree is 645 lines lighter and every claim below about A/B/C is now
> historical. Bands D/E/F are still live plans.

---

## 0. Governing constraints (read first)

This is a **scientific tool with measured recall/precision operating points**. The
default env-var values, detection thresholds, and emitted output are load-bearing.
Every item below is classified by **behavior-preservation risk**, and the safe work
is overwhelmingly *dead-code removal* and *mechanical extraction*, **not** logic
change. Three hard rules:

1. **Never change a numeric threshold, default, or emitted string.** The
   BED/VCF/TRF/STRfinder formatter output (strings, coordinates, `:.1f`/`:.0f`/
   `:.3f` precision, 1-based conversions, 150/500-char truncation) is parsed by
   downstream tools and benchmark scorers — any change there is **HIGH risk**.
2. **Mind the `.so` test gap.** `pytest` with the compiled `_accelerators.so`
   present (the dev env) exercises the Tier-1/2/3 **C paths** and the ground-truth
   fixtures. It does **not** exercise the **pure-Python fallback bodies** in
   `motif_utils.py` (`align_repeat_region` Python loop, `_align_unit_to_window`
   Python body) — those run only in a **no-`.so`** build. Any refactor of a
   fallback body must be validated by a `pytest` run with the `.so` removed/renamed.
3. **The measured operating points live on real genomes, not fixtures.** Structural
   refactors of the Tier-2/3 detection paths and the satellite gap-fill must be
   validated by a **chr21/chr22 BED-identical diff** (output byte-identical
   before/after), not by `pytest` alone. Regression guards that already exist:
   `tests/test_ground_truth.py` (44 cases), `tests/test_satellite_gapfill.py`
   (self-generating), `tests/test_autocorr.py`, `tests/test_loop_p1020.py`.

**Verification tiers referenced below:**
- **V1** = `python -m pytest tests/ -q` with `.so` present (baseline: 44 passed).
- **V2** = same suite with `.so` absent (exercises the Python fallbacks).
- **V3** = chr21 (and chr22) full-run BED output diff, must be identical.

---

## 1. Executive summary

44 raw findings across the four surveys collapse to **~30 distinct opportunities**
after dedup. They fall into six bands:

| Band | What | Count | Net effect |
|------|------|-------|-----------|
| A. Dead-code removal | unreferenced methods, wrappers, modules, dead branches | 10 | ~-260 lines, zero behavior risk |
| B. Complete-the-consolidation | inline autocorr / str-autocorr that were left behind | 2 | finishes intended work |
| C. Low-risk shared helpers | sentinel-strip, coverage-mask, entropy, validity | 5 | dedup, LOW risk |
| D. Env-var parsing | ~40 sites, 2–3 idioms, **empty-string semantic fork** | 1 | needs a human decision |
| E. Structural decomposition | 8 over-long methods → mechanical extraction | 12 | readability, test-gated |
| F. Cross-tier consolidation | tier2 scan-skeleton + refine/record sharing | 3 | HIGH value, **HIGH risk — do last** |

**Recommended sequencing:** A → B → C → (decide D) → E → F. Bands A/B/C are mostly
V1-verifiable and safe to land as a single "refactor: dead code + shared helpers"
PR. Bands E/F should be separate, individually benchmarked PRs.

---

## Band A — Dead-code removal (LOW risk, do first)

Everything here was grep-confirmed to have **zero in-repo callers**. The only
residual risk vector is an out-of-tree script importing a public-looking
`MotifUtils.*` / accelerator symbol — confirm `MotifUtils` and `accelerators` are
treated as internal before deleting the public-looking ones.

- **A1 · `MotifUtils` — 7 unreferenced static methods (~95 lines).**
  `is_transition` (motif_utils.py:174-189), `count_transversions_array` (207-227),
  `is_insertion_variant` (719-727), `is_deletion_variant` (729-737),
  `normalize_variant` (850-869), `rotate_deletion_variant` (871-882), and the
  str-based `build_consensus_motif` (884-919, distinct from the live
  `build_consensus_motif_array` at 921). `is_transition` is called only by
  `count_transversions_array`, so it is transitively dead. _Risk LOW · Effort S · V1._

- **A2 · `accelerators.py` — 6 dead Python wrappers (both native-alias + fallback
  branches).** `hamming_distance` (34/129), `pack_sequence` (36/141),
  `scan_unit_repeats` (41-50/144-152), `scan_simple_repeats` (52-64/154-164),
  `find_periodic_patterns` (66-74/166-173), `find_tandem_runs` (106-112/220-247).
  Only 5 accelerator symbols are actually imported anywhere
  (`extend_with_mismatches`, `lcp_tandem_candidates`, `find_periodic_runs`,
  `align_unit_to_window`, `anchor_scan_boundaries`). Also drop the unused
  `find_tandem_runs` import at `bwt_seed.py:17`. The `.pyx`/`.so` symbols can stay;
  only the Python wrappers are dead (no rebuild needed). _Risk LOW · Effort M · V1+V3
  smoke._

- **A3 · `BWTCore` — 4 dead methods (~50 lines).** `_build_kmer_hash`
  (bwt_core.py:150-183), `get_kmer_positions` (185-205), `_sample_suffix_array`
  (387-392), `_get_suffix_position` (500-514). The constructor already hard-disables
  their backing structures (`self.kmer_hash = {}` at 122, `self.sampled_sa = {}` at
  138) with comments saying the consumers are never called. Keep the two explanatory
  `__init__` comments. _Risk LOW · Effort S · V1._

- **A4 · `src/utils.py` — entirely orphaned.** `natural_sort_key` (utils.py:1-18)
  is imported nowhere. Either delete the module, or — if chromosome output *should*
  be naturally sorted — wire it into `main.py` (output order is currently
  detection/`as_completed` order). Wiring it in **reorders output records** (MED
  risk, verify scorers are order-insensitive); deleting is LOW. Recommend delete
  unless the sort was an intended-but-unwired feature. _Risk LOW(del)/MED(wire) ·
  Effort S._

- **A5 · `main.py` dead `out_file` + stale profile name.** Line 116 computes
  `out_file` that every format branch (202/207/226/231) recomputes; worse, for
  `--format trf` line 116 yields a wrong `prefix.trf` path that is silently
  discarded (the branch writes `.dat`). Delete line 116. Separately, the profiler
  dump is hardcoded `{output_prefix}.tier2_profile.prof` (main.py:193) though
  profiling now covers all tiers — rename to `{output_prefix}.profile.prof` (**note:
  user-visible path change**, flag in PR). _Risk LOW/MED · Effort S._

- **A6 · `models.py` dead `hasattr` branch.** `to_strfinder` (models.py:187) guards
  `if hasattr(self, 'percent_matches') and self.percent_matches is not None:` —
  `percent_matches` is a dataclass field defaulting to `0.0`, so the guard is always
  True and the `else` (confidence-based) branch is unreachable. Drop the dead branch.
  Also local `is_compound` (102) recomputes the `self.is_compound` field.
  **Do not touch the emitted strings.** _Risk LOW · Effort S · V1._

- **A7 · `bwt_seed.py` — `positions` sorted twice.** `positions.sort()`
  (bwt_seed.py:141) then `np.array(sorted(positions), ...)` (147) re-sorts an
  already-sorted list. Keep exactly one sort (`np.asarray(positions, dtype=int64)`
  at 147, or drop the 141 in-place sort if the len-check at 142 is order-agnostic —
  it is). Real work saved per FM-index query. _Risk LOW · Effort S · V1._

- **A8 · `_detect_satellite_period` redundant `best_score` + stale docstring.**
  motif_utils.py:758-813. The docstring claims "highest autocorrelation peak," but
  the loop `break`s on the first period clearing `min_identity`, so `best_score`
  never advances past `0.0` and the `identity > best_score` sub-clause is always
  true — dead. Drop `best_score` and the sub-clause (keep `if identity >=
  min_identity:`), fix the docstring to "smallest period in [50, min(n//2,500))
  with identity ≥ 0.60." Output provably identical. _Risk LOW-MED · Effort S._

- **A9 · `tier1` `perfect_length`/`seed_length` alias.** tier1.py:246 assigns
  `seed_length = array_end - array_start`; 254 assigns `perfect_length =
  seed_length`; neither is reassigned, so they are one value under two names
  (used at 267/270 vs 300-301/314-315). Collapse to one name. _Risk LOW · Effort S ·
  V1._

- **A10 · Redundant/tautological comments.** Duplicate adjacent comment pairs
  (accelerators.py:32-33 & 38-40; main.py:213-214) and tautological end-of-line
  comments throughout `tier3.py` and `bwt_seed.py` (e.g. `import math  # Standard
  library for math functions`, `n = text_arr.size  # Total sequence length`) that
  restate the code, inconsistent with tier1/tier2 density. Also fix the stale
  `tier3.find_long_repeats` docstring "Sparse sampling (stride=100)" → stride is
  adaptively computed/clamped (20-300). Collapse/strip to match house style. _Risk
  LOW (comments only) · Effort S-M._

---

## Band B — Complete an already-intended consolidation (LOW risk)

`src/autocorr.py` (commit 748b3f6) was created to own the "self-similarity at offset
p" math, but two families of call sites were left behind.

- **B1 · Two inline autocorrelations in `finder.py` → the `autocorr` primitives.**
  Both already-imported (finder.py:11).
  - `_merge_adjacent_repeats` (finder.py:367-371) re-implements `autocorr_identity`:
    `matches = np.sum(gap_seq[:total] == gap_seq[mlen:mlen+total]); identity =
    matches/total` → `autocorr_identity(gap_seq, mlen)`. The `gap_len >= mlen*2`
    guard makes `total > 0` always hold ⇒ identical value.
  - `_fill_satellite_gaps` gap detection (finder.py:463-473) re-implements
    `contiguous_true_runs` on the inverted mask (`np.diff` + boundary fixups) →
    `contiguous_true_runs(~covered)` then the existing `e-s >= 300` filter.
    `.astype(np.int8)` vs the primitive's `.view(np.int8)` are identical for a bool
    array. _Risk LOW · Effort S · V1 + V3 (satellite path)._

- **B2 · Str-based autocorrelation idiom 3× in `motif_utils` → a local helper.**
  `matches = sum(1 for i in range(total) if s[i]==s[i+p]); identity = matches/total`
  is hand-written at motif_utils.py:687-689 (`refine_repeat`) and 783 & 808
  (`_detect_satellite_period`). These are the **string-based** holdouts (the
  array-based ones were consolidated in 748b3f6). Add an **allocation-free** str
  helper `MotifUtils._str_autocorr_identity(s, p)` (do NOT encode-to-uint8 per call
  — `refine_repeat` is hot). **Do not mechanically fold these into the array
  primitive:** they are wrapped in early-exit/threshold/sub-period logic (`break` on
  first good period, `>0.85 and >best`, `>=min_identity`) that a plain reduction
  would change. _Risk LOW-MED · Effort S._

---

## Band C — Low-risk shared helpers (extraction)

- **C1 · Sentinel-strip ×4 → one helper + named constant.**
  `if n > 0 and text_arr[n-1] == 36: n -= 1` (36 = `ord('$')`) recurs at
  tier2.py:170, tier2.py:320, bwt_seed.py:90, finder.py:570. Add
  `BWTCore.effective_length` (or `strip_sentinel_len(text_arr)` + a module-level
  `SENTINEL = ord('$')`). **Note the cross-tier inconsistency:** tier1 (154) and
  tier3 (115) do **not** strip — they keep `$` in `n` and rely on `min(end,n)` +
  downstream `$`-checks. Swap only the 3 stripping sites to the helper; **do not**
  unify tier1/tier3 to also strip (that changes their `n` and every downstream
  slice — MED risk, out of scope). _Risk LOW · Effort S · V1._

- **C2 · Coverage-mask construction ×6 → `intervals_to_mask` / `_coverage_mask`.**
  Building a bool coverage mask is open-coded: from `TandemRepeat`s at
  finder.py:458-460 & 585-588; from `(start,end)` sets at tier2.py:342-344 &
  tier3.py:123-126. Add `intervals_to_mask(intervals, n)` (+ a thin
  `mask_from_repeats` adapter). Adopt the guarded `if r.start < n` form — it is
  behavior-identical for valid inputs (an out-of-range slice-assign is a no-op).
  Pair with `_uncovered_blocks(covered, n, min_len)` for the `np.diff` block logic
  in `_fill_satellite_gaps` (min-length filter as a param; satellite uses 300).
  _Risk LOW-MED · Effort S-M · V1 + V3 (satellite)._

- **C3 · Two Shannon-entropy impls (str vs uint8) → shared.**
  `MotifUtils.calculate_entropy(str)` (motif_utils.py:158-172, `Counter`) and
  `finder._seq_shannon_entropy(uint8)` (finder.py:14-27, `np.bincount`) compute the
  same `-Σ p·log2 p`. Add `MotifUtils.calculate_entropy_array(arr)` (the finder
  body) and have the catch-all gate (finder.py:616) call it; keep the str fast-path
  to avoid an encode in the TRF-stats hot path. _Risk LOW · Effort S._ (Lower
  priority — different modules/input types; churn may not pay off.)

- **C4 · `_has_invalid_char` helper (2 identical sites).**
  tier1.py:233 and tier1.py:525 both do `if '$' in motif_check or 'N' in
  motif_check`. Extract `_has_invalid_char(s)`. **Do NOT** collapse these into the
  stricter ACGT-only form used at finder.py:622 / bwt_seed.py:122 — that form also
  rejects lowercase/other bytes, a real (if unlikely) behavior change on non-ACGTN
  input. Keep both strictnesses distinct. _Risk LOW · Effort S · V1._

- **C5 · (optional) C-extension load boilerplate ×5.** The
  `try: from .c_extensions.build import load_X ... except: _c = None` idiom appears
  in tier1.py:14-19, tier2.py:15-19, bwt_core.py:14-19, motif_utils.py:10-14 &
  16-20. A `utils.try_load_c(loader_name)` unifies it, but each loads a different
  sublib so the win is marginal. Listed for completeness. _Risk LOW · Effort S._

---

## Band D — Env-var parsing (needs a human decision)

**D1 · ~40 hand-inlined env reads in 2–3 idioms with a real semantic fork.**
Sites: finder.py:111-120 & 262 & 575-583; tier1.py:40-109; tier2.py:69-97 & 508.
Two incompatible forms plus a `TIER*_MISMATCH` special case:

- **default-arg form** `int(os.environ.get("X", "d"))` — for `X=""` → `int("")`
  **raises ValueError**.
- **`or`-fallback form** `int(os.environ.get("X") or "d")` — for `X=""` → uses `d`
  (safe).

They are identical for unset and any non-empty value; they diverge **only** on the
empty-string edge case (`export X=`). A `utils.env_int/env_float/env_flag` helper
removes the repetition, but **unifying the default-arg sites onto the safe form is a
behavior change** (crash → default) for that degenerate input.

**Decision required:** (a) unify on the safe form and note the empty-string
behavior change in the PR (arguably a bugfix), **or** (b) provide a parametrized
helper `env_num(name, default, cast, empty_as_default: bool)` that preserves both
semantics verbatim (`True` for the CATCHALL_*/`or`-group, `False` for the
SAT_*/default-arg group) and only unifies *syntax*. Recommend (b) for strict
preservation. **Do NOT** build a `TierConfig` "read all env once" object — each
var's default and semantics is individually documented in CLAUDE.md and
centralizing invites drift. _Risk LOW (or-sites) / MED (default-arg sites) · Effort
M · V1._

---

## Band E — Structural decomposition (mechanical, test-gated)

All of these are "move, don't rewrite" extractions to tame over-long methods. Value
is readability; risk is that a mechanical extraction silently drops a threshold or
reorders shared mutable state. **Preserve every constant and the exact dedup/masking
order verbatim.** Verify per the `.so` notes below.

- **E1 · `finder.find_all()` — 150-line method, 3× tier scaffold + 5× timing
  boilerplate.** finder.py:122-273. Extract `_run_tier1/2/3(...)` (seams = the three
  `if self.tierN:` blocks) mutating the shared `all_repeats`/`*_seen` exactly as
  today, plus a `_timed_phase(label)` context manager for the
  `show_progress`/`t0`/`print` idiom (keep the exact per-phase strings). **Preserve
  the asymmetry** where Tier 2's long-unit phase does a *manual* overlap test vs
  `tier1_seen` (168-176) while other phases pass a `seen` set — that is behavior, not
  a smell. _Risk MED · Effort M · V1 + V3._

- **E2 · `finder._merge_adjacent_repeats()` — deep nesting + magic-number ladder.**
  finder.py:309-401. Extract `_max_merge_gap(period_len)` (the 100/20/`max(10,p)`
  ladder), `_motifs_compatible(canon1, canon2)` (exact-or-fuzzy ≤0.10 hamming),
  `_merge_quality_ok(...)` (satellite-vs-short check). Name constants (0.55, 0.15,
  50000) **without changing values**; rewrite the loop with guard clauses. Core
  scientific logic, well-exercised in every env. _Risk MED · Effort M · V1 + V3._

- **E3 · `finder._fill_satellite_gaps()` — 110-line, magic-number cluster.**
  finder.py:444-554. After C2's mask/block helpers, split out
  `_satellite_anchor_spans(repeats)` (479-486) and `_scan_block_for_satellite(...)`
  (505-537), leaving an orchestration loop. Promote hardcoded magics (1000, 300,
  50000, 100000, 5000/2500, 0.80, 50) to named constants next to the `self.sat_fill_*`
  fields, values unchanged. **Carries the CHM13 bp-recall 92.5→99.8% operating
  point.** _Risk MED-HIGH · Effort M-L · V1 + `test_satellite_gapfill` + V3._

- **E4 · `motif_utils.align_repeat_region()` — variation-string tail (S-M, cleaner
  win).** motif_utils.py:516-637; extract `_format_variations(operations_by_copy)`
  (606-622). The tail runs only on the fallback-produced `operations_by_copy`, so
  it's a **no-`.so` path** — validate with **V2**. _Risk MED · Effort S-M · V2._

- **E5 · `motif_utils._align_unit_to_window()` — 178-line 4-phase DP monolith.**
  motif_utils.py:255-433 (longest method in the file). Extract `_dp_align_matrix`
  (282-329), `_best_end_column` (331-340), `_traceback` (342-364),
  `_reconstruct_operations` (366-418). **Python body is not run in the dev/benchmark
  env** (the `.so` short-circuits at 264-278); the pending-ins/del reconstruction is
  intricate and off-by-one-prone. **Only safe behind V2.** _Risk MED-HIGH · Effort L
  · V2._

- **E6 · `motif_utils.refine_repeat()` — 4-stage primitive-length cascade.**
  motif_utils.py:639-717 (esp. 668-701). Extract `_resolve_primitive_length(...)`
  holding the exact→approx→`<=20bp` autocorr→`>=200bp` satellite cascade, each stage
  keeping its threshold (0.85, ≤20, ≥200, 0.02). The `<=20` inline loop can call the
  B2 str helper. Covered by Tier-1 `test_ground_truth` even without the `.so`. _Risk
  MED · Effort M · V1._

- **E7 · `tier1.find_strs()` — ~180-line method.** tier1.py:151-332. Extract
  `_gen_period_candidates(...)` (wraps the C-ext 197-208 / pure-Python 209-235
  branches → same `candidates` shape) and `_process_candidate(...)` (the 245-315
  extend→gate→refine→fallback→score body minus the append). Tier-1 pure-Python path
  is V1-covered; the C-ext branch needs V3. _Risk MED · Effort M._

- **E8 · `tier1._fmscan_strs()` — 133-line gather-then-claim.** tier1.py:415-548.
  Extract `_gather_period_runs(p, ...)` (density+LLR gather, 460-502),
  `_claim_period_runs(...)` (greedy territory claim, 504-542), and `_poisson_llr(occ,
  exp_bg)` (485-499). `TIER1_FMSCAN` is default-OFF so the default path is untouched;
  when enabled it has the v2.1 measured OP. _Risk MED · Effort M._

- **E9 · `tier3.find_long_repeats()` — 55-line inline ultra-long build.**
  tier3.py:176-231 (the `copies>100 or span>10000` branch) dwarfs the `else` short
  branch (232-251). Extract `_build_ultralong_repeat(...)` and `_build_dp_repeat(...)`;
  loop body becomes `repeat = builder(...); if repeat: <dedup+append+mask>`.
  `.so`-only path → V3. _Risk MED · Effort M._

- **E10 · `models.py` formatter internal dedup.** Add internal-only helpers that
  provably can't change output: a `_cons` property (`self.consensus_motif or
  self.motif`, repeated at 39/78/107-108/144), a `_composition()` for the
  `{'A':25.0,...}` default (63/83), collapse the always-equal
  `consensus_size`/`period` pair (59-60/79-80). **Leave every emitted format string,
  column order, precision, and coordinate byte-for-byte.** _Risk LOW (internal) /
  HIGH (any emitted string) · Effort M · V1._

- **E11 · `main.py` output dispatch.** Drive bed/trf/strfinder from a small table
  `{format: (ext, header_or_None, method_name)}`; keep VCF as its own branch (its
  per-record REF-derivation at 215-222 doesn't fit a uniform method). **Preserve the
  extension mapping (`trf`→`.dat`, `strfinder`→`.csv`) and exact header strings
  verbatim.** _Risk MED · Effort M._

- **E12 · `bwt_core` Kasai LCP duplicated jit/non-jit.** bwt_core.py:31-47 (`@njit`)
  vs 59-88 (Python tail 73-88) write the same Kasai loop twice. Extract
  `_kasai_core(text_codes, sa)` and apply `_nb.njit` conditionally (`_kasai_lcp_uint8
  = _nb.njit(cache=True)(_kasai_core)` when numba present, else raw). Numba compiles
  the same source ⇒ bit-identical. The LCP feeds Tier-2 `lcp_tandem_candidates`, so
  smoke-test. _Risk LOW-MED · Effort M · V1 + V3._

---

## Band F — Cross-tier consolidation (HIGH value, HIGH risk — do last, one PR each)

These remove the largest blocks of parallel-but-divergent code, but they touch the
`.so`-only Tier-2/3 detection paths whose small deltas are the measured operating
points. **Land each behind a chr21/chr22 BED-identical diff (V3).** A subtle
parameter-threading bug here moves recall/precision silently and `pytest` will not
catch it.

- **F1 · `tier2` — the two scan methods duplicate the whole Phase-A(LCP) +
  Phase-B(k-mer) skeleton.** `find_long_unit_repeats_strict` (tier2.py:144-278) vs
  `_find_repeats_simple` (303-480) are near-identical, differing in 4 load-bearing
  deltas: (a) `lcp_thresh = max(8, min_p//2)` dynamic vs fixed `10`; (b)
  `tier1_mask` seeding of `covered_mask`; (c) `required_copies = 2 if period>=20
  else short_req_copies` vs the `min_copies` arg; (d) a `cov_frac > 0.5` pre-refine
  skip. Extract `_lcp_phase_a(..., *, copies_fn, cov_frac_gate)` and `_kmer_phase_b(...)`
  with the 4 deltas as params/callbacks. **Highest value, do most carefully.** _Risk
  MED-HIGH · Effort L · V3 (BED-identical)._

- **F2 · `tier2` — the reduce→refine→record triple repeated 5×.** tier2.py:231-239,
  265-273, 395-409, 444-451, 541-563. Add `_accept(chromosome, motif_bytes, start,
  end, out, covered, mask, n, *, min_copies=None) -> bool` doing
  `_reduce_to_primitive` + `_refine_and_create_repeat` + `_record_repeat`. The
  autocorr retry loop calls it per candidate and breaks on first True; the
  `cov_frac` pre-checks stay *outside* the helper. Complements the existing
  `_record_repeat`. _Risk LOW-MED · Effort S-M · V3._

- **F3 · Shared `refine_and_build` across the three tiers.** The refine→build dance
  is centralized in tier2 (`_refine_and_create_repeat`, 108-127) but re-threaded in
  tier1 (`_build_repeat` + inline `refine_repeat`, 285-308/528-536) and inlined in
  tier3 (237-248). Promote `MotifUtils.refine_and_build(sequence, start, end, motif,
  *, tier, text_arr, mismatch_fraction, indel_fraction, min_copies, strand)`. **Each
  caller passes a different `sequence` object (tier1 `sequence_str`, tier2
  `self.sequence_str`, tier3 `self.bwt.text`) and different constants (tier3
  hardcodes 0.2/0.1)** — thread every one through unchanged. Best done *after* F1/F2
  (which would consume it). _Risk MED · Effort M · V3 + tier fixtures._

- **F0 · Maintainability: accelerator fallback-stub taxonomy + one-time warning.**
  Not a code-logic change, but the highest-leverage clarity fix. The
  `accelerators.py` fallback branch (127-300) mixes **real fallbacks**
  (`lcp_tandem_candidates`, `anchor_scan_boundaries`) with **degenerate stubs that
  silently gut detection** when the `.so` is absent: `extend_with_mismatches → None`
  and `find_periodic_runs → []` cause Tier-1/2/3 to fall back to raw seed boundaries
  / produce nothing, collapsing recall with no error. Add (a) a one-time
  `warnings.warn(...)` when `_native is None` and a degenerate stub is entered, and
  (b) a classifying docstring on each stub ("no-op fallback — disables mismatch
  extension; build `_accelerators` for full recall" vs "faithful pure-Python
  fallback"). **Do not change any return value.** _Risk LOW · Effort S._

---

## Explicit "do NOT do" list (traps the surveys flagged)

- **Do not** collapse the two env-var empty-string semantics into one (D1) without a
  deliberate, separately-reviewed decision — it changes the `X=""` contract.
- **Do not** unify the motif-validity checks across the `$`/`N` (looser) and
  ACGT-only (stricter) forms (C4) — different strictness is a real behavior fork.
- **Do not** mechanically port the `motif_utils` str-autocorrelation loops into the
  array primitive (B2) — they carry break/threshold logic a plain reduction lacks.
- **Do not** force one shared "append + seen + mask" helper across all tiers — each
  marks a **different span** (tier1 marks the *seed* region, not the repeat span;
  tier3/bwt_seed key by bucketed `region_key`; tier2 by exact `(start,end)`).
  Unifying naively moves coverage. (Naming-only unification of `seen_mask` /
  `covered_mask` / `mask` is fine.)
- **Do not** change any emitted BED/VCF/TRF/STRfinder string, column, precision, or
  coordinate (E10/E11) — externally consumed.
- **Do not** build a centralized `TierConfig` object (D1) — invites default drift
  from the individually-documented CLAUDE.md knobs.

---

## Consolidated backlog table

| ID | Title | Band | Risk | Effort | Verify |
|----|-------|------|------|--------|--------|
| A1 | MotifUtils 7 dead static methods (~95L) | A | LOW | S | V1 |
| A2 | accelerators 6 dead wrappers + dead import | A | LOW | M | V1+V3 |
| A3 | BWTCore 4 dead methods (~50L) | A | LOW | S | V1 |
| A4 | utils.py orphaned module | A | LOW | S | V1 |
| A5 | main.py dead out_file + stale profile name | A | LOW/MED | S | V1 |
| A6 | models.py dead hasattr branch | A | LOW | S | V1 |
| A7 | bwt_seed double-sort | A | LOW | S | V1 |
| A8 | _detect_satellite_period redundant best_score | A | LOW-MED | S | V1 |
| A9 | tier1 perfect_length/seed_length alias | A | LOW | S | V1 |
| A10 | tautological/duplicate comments + stale docstring | A | LOW | S-M | — |
| B1 | finder inline autocorr → autocorr primitives | B | LOW | S | V1+V3 |
| B2 | motif_utils str-autocorr ×3 → local helper | B | LOW-MED | S | V1 |
| C1 | sentinel-strip ×4 → helper + SENTINEL const | C | LOW | S | V1 |
| C2 | coverage-mask ×6 → intervals_to_mask | C | LOW-MED | S-M | V1+V3 |
| C3 | Shannon entropy str/uint8 → shared | C | LOW | S | V1 |
| C4 | _has_invalid_char (2 sites) | C | LOW | S | V1 |
| C5 | C-ext load boilerplate ×5 (optional) | C | LOW | S | V1 |
| D1 | env-var parsing helper (empty-string fork) | D | LOW/MED | M | V1 |
| E1 | find_all() 150L: tier scaffold + timing | E | MED | M | V1+V3 |
| E2 | _merge_adjacent_repeats nesting + magics | E | MED | M | V1+V3 |
| E3 | _fill_satellite_gaps 110L + magics | E | MED-HIGH | M-L | V1+satgap+V3 |
| E4 | align_repeat_region variation-string tail | E | MED | S-M | V2 |
| E5 | _align_unit_to_window 178L DP monolith | E | MED-HIGH | L | V2 |
| E6 | refine_repeat 4-stage cascade | E | MED | M | V1 |
| E7 | tier1 find_strs ~180L | E | MED | M | V1+V3 |
| E8 | tier1 _fmscan_strs 133L | E | MED | M | V1(off)+V3 |
| E9 | tier3 ultra-long/dp builders | E | MED | M | V3 |
| E10 | models formatter internal dedup | E | LOW/HIGH | M | V1 |
| E11 | main.py output dispatch table | E | MED | M | V1 |
| E12 | bwt_core Kasai LCP jit/non-jit dedup | E | LOW-MED | M | V1+V3 |
| F0 | accelerators fallback-stub taxonomy + warning | F | LOW | S | V1 |
| F1 | tier2 two-scan Phase-A/B skeleton dedup | F | MED-HIGH | L | V3 |
| F2 | tier2 reduce→refine→record ×5 → _accept | F | LOW-MED | S-M | V3 |
| F3 | shared refine_and_build across tiers | F | MED | M | V3 |

---

## Suggested PR breakdown

1. **PR "refactor: dead code sweep"** — A1-A10 (+ F0 taxonomy). Pure removal +
   comments + the one-time warning. Gate: V1 (44 passed) + one chr21 run for A2.
   ~-260 lines, zero behavior risk. Land first.
2. **PR "refactor: shared helpers"** — B1, B2, C1-C4 (+ C5 optional). Gate: V1 +
   V3 for B1/C2 (satellite path).
3. **PR "refactor: env-var parsing"** — D1, after the (b)-vs-(a) decision. Gate: V1.
4. **PRs "refactor: decompose <method>"** — E1-E12, one method-family per PR. Gate
   per the table (mind V2 for E4/E5, `test_satellite_gapfill` for E3).
5. **PRs "refactor: tier consolidation"** — F1, F2, F3, each its own PR behind a
   chr21+chr22 BED-identical diff.

_Source surveys (scratchpad): `refactor-survey-crosscut.md`,
`refactor-survey-core.md`, `refactor-survey-tiers.md`, `refactor-survey-support.md`._
