# The caller is not reproducible: `align_accel.c` reads uninitialised memory

_Measured 2026-07-09, on `perf/exp1-human-sensitivity`. This is a **pre-existing**
defect, unrelated to the accelerator-fallback work landed the same day. It is
**not fixed** — fixing it changes detection results, so it needs its own
measured PR. The evidence and a candidate patch are below._

---

## 1. What was measured

Running the **same commit**, from the **same git worktree**, with the **same
environment**, over chr21 + chr22 (the `x_*` catchH gate base) produces
**different BED output on different runs**:

| run | pool rows | pool sorted-diff vs the other |
|-----|----------:|------------------------------:|
| `pre`  | 131 735 | — |
| `pre2` | 131 736 | 67 lines |

Region recall/precision move by ≲0.01 pp, so the scores look stable; the calls
underneath are not. `exp1_human/loop/resume.md` had noted "proxy
non-determinism (~44–53 lines)" without a cause.

Ruled out first, each by inspection or measurement:

- **Threading.** `chr21.fa` and `chr22.fa` each hold one record, and `main.py`
  takes the serial branch when `len(sequences) == 1`.
- **Time-based cutoffs.** Every `time.time()` in the tiers feeds a progress
  message; none gates detection.
- **Numba.** Not installed in the benchmark env (`HAVE_NUMBA` is False).
- **OpenMP / pthreads.** Absent from all four C extensions.
- **String-hash randomisation.** Every set/dict that reaches output is keyed by
  `int` or `Tuple[int, int]`, whose hashes are not randomised. The one
  `Set[str]` (`bwt_seed.seen_kmers`) is membership-only.

## 2. The cause

Pinning `PYTHONHASHSEED` **and** `MALLOC_PERTURB_` makes the pipeline
deterministic; changing *only* `MALLOC_PERTURB_` changes the output. glibc fills
freshly `malloc`'d memory with `MALLOC_PERTURB_ ^ 0xff`, so this is a direct
measurement that **detection depends on the contents of uninitialised heap
memory**. chr22, identical code, `PYTHONHASHSEED=0`:

| run | `MALLOC_PERTURB_` | rows | vs the first |
|-----|------------------:|-----:|--------------|
| A | 0   | 66 919 | — |
| B | 0   | 66 919 | **byte-identical** |
| D | 255 | 66 907 | **differs** |

The read is in `src/c_extensions/align_accel.c`, the C accelerator behind
`MotifUtils.align_repeat_region` (Tier 2 / satellite refinement — which is
exactly where the differing rows concentrate: 82/75 tier-2 and 16/13 satellite
rows of a 192-line raw diff).

`ptr_table` is `malloc`'d and never zeroed:

```c
char *ptr_table = (char *)malloc(((size_t)m + 1) * cols_sz * sizeof(char));
```

Row 0 is initialised. For every row `i >= 1` the fill loop writes only the band
`[j_min, j_max]`:

```c
for (int j = j_min; j <= j_max; j++) { ...; ptr_table[(size_t)i * cols_sz + j] = bp; }
```

So **column 0 of every row `i >= 1`, and every out-of-band column, is never
assigned** — yet the traceback reads exactly those cells:

```c
char op = ptr_table[(size_t)i * cols_sz + j];   /* line 162 */
```

A traceback that walks off the left edge (any alignment ending in deletions)
lands on column 0 with `i > 0` and reads whatever the allocator left there. If
the byte happens to be `'M'`, `'S'`, `'D'` or `'I'` the alignment continues along
a garbage path; otherwise it stops early. Both outcomes are silent.

Note the two sibling implementations get this right, which is why the C path is
the odd one out:

- `_accelerators.pyx` zero-fills its whole `ptr` table and then sets
  `ptr[i * cols] = 3` (`D`) for every row — an explicit column-0 backpointer.
- `MotifUtils._align_unit_to_window` (pure Python) does the same with its
  `ptr` list-of-lists.

`obs_bases` is also `malloc`'d, but every read is guarded by `obs_valid`, which
is `memset` to zero first. `bwt_accel.c`'s `rank` buffer is fully written before
use. `ptr_table` is the only unguarded uninitialised read found.

## 3. Why it is not fixed here

The obvious patch is two lines — `calloc` the table, and store the column-0
backpointer where the traceback will actually find it:

```c
-    char *ptr_table = (char *)malloc(((size_t)m + 1) * cols_sz * sizeof(char));
+    char *ptr_table = (char *)calloc(((size_t)m + 1) * cols_sz, sizeof(char));
...
     curr_row[0] = i;
     curr_ptr[0] = 'D';
+    ptr_table[(size_t)i * cols_sz] = 'D';
```

Built and tested (worktree, `gcc` from the `bwtandem` conda env):

| build | `tests/` result |
|-------|-----------------|
| current (`malloc`, no column-0 write) | 81 passed |
| `calloc` only | **1 failed** — `TestAdjacentGroundTruth::test_sensitivity` 81.8 % < 95 % |
| `calloc` + column-0 `'D'` | **1 failed** — same test, 90.9 % < 95 % |

`calloc` alone turns column 0 into the Stop code, so tracebacks that should have
walked up the left edge terminate early and deletions are undercounted — that is
the 81.8 %. Restoring the `'D'` backpointer recovers most of it. The residual
gap means the C DP still diverges from the `.pyx` somewhere else (`best_j`
selection range and the traceback termination condition are the next places to
look).

So: the current output is undefined, and the *correct* output is not what the
current code produces. Replacing one with the other is a **detection change**,
not a refactor, and it moves a ground-truth sensitivity floor. It needs its own
PR with a chr21/chr22 measurement and a decision about the published operating
points, not a drive-by commit. The patch is preserved at
`scratchpad/align_accel_ub.patch` and reproduced above.

## 4. Consequences for anyone verifying a change

- **BED byte-identity at chromosome scale is not an achievable acceptance
  criterion** with the current C extension. Two runs of the same commit differ.
- To compare two commits on real chromosomes, pin **both** `PYTHONHASHSEED` and
  a **non-zero** `MALLOC_PERTURB_`. A non-zero value makes the garbage a
  constant, which removes the run-to-run and path-to-path variation.
- The synthetic fixtures in `tests/fixtures/` do **not** trigger the read (their
  tracebacks never reach column 0), so `pytest` and fixture-level BED diffs are
  stable and remain a valid regression gate.
- Published recall/precision figures are reproducible to about ±0.01 pp, but the
  individual calls behind them are not.
