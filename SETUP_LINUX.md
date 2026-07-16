# Linux / HPC Setup (native, no Docker)

The repo ships `environment.yml` (pinned `CONDA_SUBDIR: osx-64`, i.e. macOS) and a
`Dockerfile`. This document is the **native Linux / HPC recipe** — useful on
clusters where Docker is unavailable but `conda`/`micromamba` is. It was verified
end-to-end on a Linux x86-64 HPC login node (inside a Singularity container, no
system compiler, no `module` system).

## Why a dedicated recipe is needed

BWTandem's hot paths live in a Cython extension (`src/_accelerators.pyx`) and four
ctypes C libraries (`src/c_extensions/*.c`). **Without the compiled Cython
extension the pure-Python fallbacks return empty results and the tool reports 0
repeats** (see `src/accelerators.py`). So a working install must compile these,
which requires a C compiler.

Two gotchas on managed clusters:

1. **No system `gcc`** (and often no `module` command inside containers). Solution:
   pull the compiler from conda-forge (`c-compiler` → `gcc`).
2. **A `~/.condarc` with `channel_priority: strict` and `defaults` listed first
   makes the solver hang** trying to satisfy conda-forge compiler packages from
   `defaults` (observed: >10 min stuck at 96% CPU). Solution: solve against
   conda-forge only with `--override-channels`.

## Steps (verified)

```bash
# 0. From the repo root
cd /path/to/bwt-algorithm

# 1. Create an isolated env WITH a compiler.
#    --override-channels -c conda-forge avoids the strict-priority solver hang.
#    (Solve drops from >10 min to ~4 s.)
micromamba create -y -n bwtandem --override-channels -c conda-forge \
    python=3.11 numpy cython pip c-compiler
# (conda works the same: conda create -n bwtandem --override-channels -c conda-forge ...)

# 2. pydivsufsort (fast suffix-array construction). Ships manylinux wheels,
#    so no compiler is needed for this one.
micromamba run -n bwtandem pip install pydivsufsort

# 3. Build the REQUIRED Cython extension (else the tool finds 0 repeats).
micromamba run -n bwtandem python -c "from setuptools import setup, Extension; from Cython.Build import cythonize; import numpy as np; ext=[Extension('src._accelerators',['src/_accelerators.pyx'],include_dirs=[np.get_include()],extra_compile_args=['-std=c99'])]; setup(script_args=['build_ext','--inplace'], ext_modules=cythonize(ext, compiler_directives={'language_level':'3'}))"

# 4. Build the ctypes C extensions (tier1/tier2/bwt/align accelerators).
micromamba run -n bwtandem python -m src.c_extensions.build

# 5. Verify: should report a non-zero repeat count.
micromamba run -n bwtandem python -m src.main tests/fixtures/synth_tier3.fa \
    --format bed -o /tmp/bwt_smoke
wc -l /tmp/bwt_smoke.bed     # expect > 0 (8 on the bundled fixture)
```

## Running

Always run from the repo root (the tool is the `src` package):

```bash
micromamba run -n bwtandem python -m src.main <input.fa> \
    --min-period 1 --max-period 2000 --format bed -o <out_prefix>
```

`numba` is optional (JIT for rank/LCP queries). Install it for extra speed; it is
not required for correctness and is omitted above to keep the solve fast.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Total repeats found: 0` on real input | Cython `_accelerators` not built | Run step 3; confirm `src/_accelerators*.so` exists |
| `DEBUG: importing _accelerators failed` | `.so` missing / wrong platform (a macOS `.so` won't load on Linux) | Rebuild step 3 on this platform |
| `micromamba create` hangs for minutes at high CPU | `~/.condarc` `defaults` + `channel_priority: strict` | add `--override-channels -c conda-forge` |
| `gcc: command not found` during build | no system compiler | ensure `c-compiler` is in the env (step 1); run builds via `micromamba run -n bwtandem` so conda's gcc is on PATH |

## Verified toolchain

```
python 3.11.15 · numpy 2.4.6 · cython 3.2.5 · gcc 14.3.0 (conda-forge) · pydivsufsort 0.0.20
Built: src/_accelerators.cpython-311-x86_64-linux-gnu.so
       src/c_extensions/lib{tier1_scan,tier2_accel,bwt_accel,align_accel}.so
Smoke test (tests/fixtures/synth_tier3.fa): 8 repeats
```
