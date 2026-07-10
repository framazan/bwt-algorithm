# The caller is not reproducible: `align_accel.c` reads uninitialised memory

_Measured 2026-07-09, on `perf/exp1-human-sensitivity`. A **pre-existing** defect,
unrelated to the accelerator-fallback work landed the same day._

> **RESOLVED.** The first draft of this document said the fix "changes detection
> results, so it needs its own measured PR." That turned out to be half right:
> the uninitialised read was masking **two further divergences** between the C
> accelerator and the Python loop it is supposed to accelerate. Correcting the
> memory bug alone made a fixture regress; correcting all three made the two
> implementations agree **exactly** — 0 disagreements over 5000 random regions,
> where the shipped code disagreed on 31%. Section 5 records the remediation.
> `tests/test_align_parity.py` now pins it.

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

| run | `MALLOC_PERTURB_` | garbage byte | rows | vs run A |
|-----|------------------:|-------------:|-----:|----------|
| A | 0   | whatever the heap held | 66 919 | — |
| B | 0   | same process history   | 66 919 | **byte-identical** |
| D | 255 | `0x00`                 | 66 907 | **differs** |
| G | 1   | `0xfe`                 | 66 907 | **differs from A, identical to D** |

The last row is the tell. Neither `0x00` nor `0xfe` is one of the traceback's
opcodes (`'M'`, `'S'`, `'D'`, `'I'`), so both make it stop at the first
unwritten cell — deterministically, and identically. Only the *real* heap
garbage occasionally lands on a byte that looks like an opcode, which lets the
traceback keep walking a fabricated alignment path. **Those accidents produce 12
extra calls on chr22 alone.**

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

## 3. The memory bug was hiding two more

Fixing only the uninitialised read makes things *worse* on the fixtures, which is
how the other two divergences surfaced. Built and tested each step (worktree,
`gcc` from the `bwtandem` conda env):

| build | `tests/` | Adjacent sensitivity | C-vs-Python disagreement¹ |
|-------|----------|---------------------|---------------------------|
| shipped (`malloc`, no column-0 write) | 81 passed | 100 % (by accident) | **31 %** |
| `calloc` only | 1 failed | 81.8 % | — |
| `calloc` + column-0 `'D'` | 1 failed | 90.9 % | 15 % |
| + matching extension bound | 81 passed | 100 % | 15 % |
| + matching consensus tie-break | **84 passed** | 100 % | **0 %** |

¹ fraction of 1500 random repeat regions where `align_repeat_region` returns a
different summary with and without `libalign_accel` loaded. At 5000 trials the
final build still disagrees on **zero**.

**(a) The uninitialised read.** `calloc` alone turns column 0 into the Stop code,
so a traceback that should walk up the left edge halts and deletions go
uncounted. Writing the `'D'` backpointer into `ptr_table` — where the traceback
looks, rather than into the `curr_ptr` scratch row nothing reads — restores it.
The counting is now arithmetically checkable: 17 copies of a 3-mer consuming 44
bases must have 7 deletions. The shipped code reported **0**.

**(b) The extension bound.** `align_repeat_region_c` stopped at
`end + 3*motif_len + 4*max_indel`; `MotifUtils.align_repeat_region` stops at
`max(end, start + motif_len*min_copies) + max(3*motif_len, 4*max_indel)`. On the
`ACG` array of `synth_adjacent` that let the C run four bases further and claim
two extra copies (17/44 vs the Python loop's 15/40). With deletions now counted,
the longer C call tripped `_filter_overlaps`' `overlap / min(length) > 0.5` rule
against its neighbour and the `ACG` repeat was dropped — the 90.9 %.

**(c) The consensus tie-break.** `_consensus_from_counts` used
`Counter.most_common`, which breaks ties by insertion order. The C scans `A, C,
G, T` with a strict `>`, and `build_consensus_motif_array` takes `np.argmax` over
`np.unique`'s sorted values — both pick the lexicographically smallest base. So
the Python loop was the odd one out, and a single tied position produced a
different consensus, which then cascaded into every subsequent copy's alignment.
Aligning it to the other two costs nothing on the production path (the C consensus
is unchanged) and takes the disagreement to zero.

## 4. Consequences for anyone verifying a change

- Before this fix, **BED byte-identity at chromosome scale was not an achievable
  acceptance criterion**: two runs of the same commit differed, and with
  `MALLOC_PERTURB_=0` the output depended even on the directory the repo sat in.
  To compare two commits across that boundary, pin **both** `PYTHONHASHSEED` and
  a **non-zero** `MALLOC_PERTURB_`. That was validated with a same-code /
  different-path control, and used to show the accelerator refactor
  (`dfcdbcb` → `8838507`) was byte-identical at `MALLOC_PERTURB_` = 1 and 255.
- After this fix, chr22 output is identical at `MALLOC_PERTURB_` = 0, 1 and 255,
  so the pinning is no longer needed.
- The synthetic fixtures never reached the uninitialised cells, so `pytest` and
  fixture BED diffs were always a stable gate — they just could not see the bug.
  `tests/test_align_parity.py` can.
- `c_extensions/build.py` used to rebuild only when a `.so` was **missing**. The
  libraries are gitignored, so anyone who had already built would have kept
  running the old binary after pulling this fix, silently. It now rebuilds when
  the `.c` is newer.

## 5. What was done, and what it cost

Four changes, in `align_accel.c`, `motif_utils.py` and `c_extensions/build.py`,
described in §3 and pinned by `tests/test_align_parity.py`:

1. `calloc` the traceback table (defence — the out-of-band cells are provably
   never read, but a future traceback bug should not become a heap read);
2. write the column-0 backpointer into `ptr_table`, not just `curr_ptr`;
3. give the C loop the Python loop's extension bound;
4. break consensus ties on the smallest base everywhere.

Plus a stale-check in `build.py` so the fix actually reaches an existing checkout.

**Determinism, chr22, `PYTHONHASHSEED=0`:** the fixed build emits byte-identical
BED at `MALLOC_PERTURB_` = 0, 1 **and** 255. The undefined behaviour is gone; the
pinning workaround from §4 is no longer needed.

**Detection, chr22 vs the adotto ground truth (24 807 regions), catchH gate base:**

| build | calls | region recall | region precision | bp recall | bp precision |
|-------|------:|--------------:|-----------------:|----------:|-------------:|
| shipped (undefined) | 66 907 | 84.50 % | 52.34 % | 47.80 % | 32.63 % |
| memory bug only | 66 583 | 84.40 % | 52.60 % | 47.73 % | 32.66 % |
| **all four** | **66 854** | **84.38 %** | **52.74 %** | **47.49 %** | **32.65 %** |

It trades **0.12 pp of region recall for 0.40 pp of region precision** — 53 fewer
calls, most of them the fabricated ones the garbage traceback used to invent.
That is a net gain on the F1 the benchmark reports (64.6 % → 64.9 %), and it is
the first time the number means anything, because the previous one was a sample
from an undefined distribution.

The synthetic fixtures shift slightly too: the same number of calls in each, with
4–8 changed boundary/statistic lines in `synth_tier1`, `synth_mixed` and
`synth_adjacent`. Every ground-truth threshold still passes.
