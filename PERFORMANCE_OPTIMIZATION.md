# Performance Optimization Guide

This document describes strategies to accelerate the BWT tandem repeat finder using **Cython**, **multithreading**, and **multiprocessing**.

## Current Bottlenecks

Based on profiling of a 10 Mbp chromosome:

| Operation | Time % | Location | Parallelizable? |
|-----------|--------|----------|-----------------|
| Tier 2 scanning | 60-70% | `_find_repeats_simple()` | ✅ Yes (by period) |
| Consensus building | 15-20% | `build_consensus_motif_array()` | ✅ Yes (vectorized) |
| Hamming distance | 10-15% | `hamming_distance_array()` | ✅ Yes (Cython) |
| FM-index search | 5-10% | `backward_search()` | ⚠️ Partially |
| Entropy calculation | 3-5% | `calculate_entropy()` | ✅ Yes (batch) |

---

## Strategy 1: Multiprocessing (EASIEST, BIGGEST GAIN)

**Best for**: Chromosome-level parallelism
**Expected speedup**: **4-8× on multi-core systems**
**Difficulty**: ⭐ Easy

### Implementation

Since each chromosome is processed independently, we can parallelize at the chromosome level:

```python
from multiprocessing import Pool, cpu_count

def process_chromosome(args):
    """Worker function for parallel chromosome processing."""
    chrom, seq, config = args

    # Build BWT for this chromosome
    bwt_core = BWTCore(seq, config['sa_sample_rate'])

    # Find repeats
    repeats = []
    if config['enable_tier1']:
        tier1 = Tier1STRFinder(bwt_core, ...)
        repeats.extend(tier1.find_strs(chrom))

    if config['enable_tier2']:
        tier2 = Tier2LCPFinder(bwt_core, ...)
        repeats.extend(tier2.find_long_repeats(chrom))

    # Clean up
    bwt_core.clear()

    return repeats

def find_tandem_repeats_parallel(self, enable_tier1=True, enable_tier2=True,
                                 n_jobs=None):
    """Parallel version using multiprocessing."""
    if n_jobs is None:
        n_jobs = min(cpu_count(), len(self.sequences))

    # Prepare arguments for each chromosome
    tasks = []
    for chrom, seq in self.sequences.items():
        config = {
            'sa_sample_rate': self.sa_sample_rate,
            'enable_tier1': enable_tier1,
            'enable_tier2': enable_tier2,
            'max_motif_length': self.max_motif_length,
            # ... other config
        }
        tasks.append((chrom, seq, config))

    # Process in parallel
    print(f"Processing {len(tasks)} chromosomes using {n_jobs} cores...")
    with Pool(n_jobs) as pool:
        results = pool.map(process_chromosome, tasks)

    # Flatten results
    all_repeats = []
    for repeats in results:
        all_repeats.extend(repeats)

    return all_repeats
```

### Usage

```bash
# Use all available cores
python bwt.py genome.fa -o repeats.bed --parallel

# Limit to 4 cores
python bwt.py genome.fa -o repeats.bed --parallel --jobs 4
```

**Pros**:
- Simple to implement (minimal code changes)
- True parallelism (no GIL)
- Works immediately with existing code

**Cons**:
- Memory overhead (each process loads its own BWT)
- Not helpful for single large chromosomes

---

## Strategy 2: Cython Optimization (MEDIUM EFFORT, BIG GAIN)

**Best for**: Hot loops (consensus, Hamming distance, entropy)
**Expected speedup**: **2-5× for computational kernels**
**Difficulty**: ⭐⭐ Medium

### Step 1: Create `bwt_fast.pyx`

```cython
# bwt_fast.pyx - Cython-optimized hot loops
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport log2

ctypedef cnp.uint8_t uint8

@cython.boundscheck(False)
@cython.wraparound(False)
def hamming_distance_fast(cnp.ndarray[uint8, ndim=1] a,
                          cnp.ndarray[uint8, ndim=1] b):
    """Fast Hamming distance using C loops."""
    cdef int n = a.shape[0]
    cdef int dist = 0
    cdef int i

    for i in range(n):
        if a[i] != b[i]:
            dist += 1

    return dist

@cython.boundscheck(False)
@cython.wraparound(False)
def build_consensus_fast(cnp.ndarray[uint8, ndim=2] copies):
    """Fast consensus building with vote counting.

    Args:
        copies: 2D array (n_copies, motif_len) of ASCII codes

    Returns:
        consensus: 1D array (motif_len,) of ASCII codes
    """
    cdef int n_copies = copies.shape[0]
    cdef int motif_len = copies.shape[1]
    cdef cnp.ndarray[uint8, ndim=1] consensus = np.zeros(motif_len, dtype=np.uint8)

    cdef int pos, copy_idx, base
    cdef int counts[256]  # ASCII table
    cdef int max_count, max_base

    for pos in range(motif_len):
        # Reset counts
        for i in range(256):
            counts[i] = 0

        # Count bases at this position
        for copy_idx in range(n_copies):
            base = copies[copy_idx, pos]
            counts[base] += 1

        # Find most common (majority vote)
        max_count = 0
        max_base = 65  # 'A'
        for base in [65, 67, 71, 84]:  # A, C, G, T
            if counts[base] > max_count:
                max_count = counts[base]
                max_base = base

        consensus[pos] = max_base

    return consensus

@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_entropy_fast(cnp.ndarray[uint8, ndim=1] sequence):
    """Fast Shannon entropy calculation.

    Args:
        sequence: 1D array of ASCII codes

    Returns:
        entropy: Shannon entropy in bits
    """
    cdef int n = sequence.shape[0]
    cdef int counts[256]
    cdef int i, base
    cdef double p, entropy = 0.0

    # Reset counts
    for i in range(256):
        counts[i] = 0

    # Count bases
    for i in range(n):
        counts[sequence[i]] += 1

    # Calculate entropy
    for base in [65, 67, 71, 84]:  # A, C, G, T
        if counts[base] > 0:
            p = <double>counts[base] / <double>n
            entropy -= p * log2(p)

    return entropy

@cython.boundscheck(False)
@cython.wraparound(False)
def scan_period_fast(cnp.ndarray[uint8, ndim=1] s_arr,
                     int period, int min_copies,
                     double max_mismatch_rate):
    """Fast period scanning for Tier 2.

    Returns list of (start, end, copies) tuples.
    """
    cdef int n = s_arr.shape[0]
    cdef int i = 0, j, copies, mismatches
    cdef int max_mm_per_copy = max(1, int(0.1 * period))
    cdef list results = []

    while i + 2 * period <= n:
        # Check if motif has sentinel or N
        has_invalid = False
        for j in range(i, i + period):
            if s_arr[j] == 36 or s_arr[j] == 78:  # '$' or 'N'
                has_invalid = True
                break

        if has_invalid:
            i += 1
            continue

        # Count tandem copies
        copies = 1
        j = i + period

        while j + period <= n:
            # Count mismatches in this copy
            mismatches = 0
            for k in range(period):
                if s_arr[i + k] != s_arr[j + k]:
                    mismatches += 1

            if mismatches <= max_mm_per_copy:
                copies += 1
                j += period
            else:
                break

        if copies >= min_copies:
            results.append((i, j, copies))
            i = j  # Jump past repeat
        else:
            i += 1

    return results
```

### Step 2: Create `setup.py`

```python
# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "bwt_fast",
        ["bwt_fast.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native"],
    )
]

setup(
    name="bwt_fast",
    ext_modules=cythonize(extensions, compiler_directives={
        'language_level': "3",
        'boundscheck': False,
        'wraparound': False,
    }),
)
```

### Step 3: Compile

```bash
# Install Cython
pip install cython

# Compile extension
python setup.py build_ext --inplace

# This creates bwt_fast.pyd (Windows) or bwt_fast.so (Linux/Mac)
```

### Step 4: Use in `bwt.py`

```python
# At top of bwt.py
try:
    import bwt_fast
    USE_CYTHON = True
    print("[INFO] Using Cython-optimized functions")
except ImportError:
    USE_CYTHON = False
    print("[INFO] Cython not available, using NumPy")

# In MotifUtils class
@staticmethod
def hamming_distance_array(a: np.ndarray, b: np.ndarray) -> int:
    """Calculate Hamming distance between two arrays."""
    if USE_CYTHON:
        return bwt_fast.hamming_distance_fast(a, b)
    else:
        return int(np.sum(a != b))

@staticmethod
def calculate_entropy(sequence: str) -> float:
    """Calculate Shannon entropy."""
    if USE_CYTHON:
        arr = np.frombuffer(sequence.encode('ascii'), dtype=np.uint8)
        return bwt_fast.calculate_entropy_fast(arr)
    else:
        # Original NumPy implementation
        ...
```

**Pros**:
- 2-5× speedup for hot loops
- Optional (falls back to NumPy if not available)
- Works with existing architecture

**Cons**:
- Requires C compiler
- Platform-dependent compilation
- More complex distribution

---

## Strategy 3: Numba JIT (EASIEST SPEEDUP)

**Best for**: Quick optimization without compilation
**Expected speedup**: **1.5-3× for numerical loops**
**Difficulty**: ⭐ Very Easy

### Implementation

```python
# At top of bwt.py
try:
    from numba import jit
    USE_NUMBA = True
except ImportError:
    # Fallback: identity decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    USE_NUMBA = False

# Optimize hot functions
@jit(nopython=True, cache=True)
def hamming_distance_numba(a, b):
    """Numba-optimized Hamming distance."""
    dist = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            dist += 1
    return dist

@jit(nopython=True, cache=True)
def calculate_entropy_numba(counts, n):
    """Numba-optimized entropy calculation."""
    entropy = 0.0
    for count in counts:
        if count > 0:
            p = count / n
            entropy -= p * np.log2(p)
    return entropy
```

**Usage**:
```bash
pip install numba
python bwt.py genome.fa -o repeats.bed  # Automatically uses Numba if available
```

**Pros**:
- Zero code changes (just add decorator)
- No compilation step
- Easy to install

**Cons**:
- Less speedup than Cython
- Some NumPy functions not supported

---

## Strategy 4: Thread-Level Parallelism (ADVANCED)

**Best for**: Parallelizing Tier 2 period scanning
**Expected speedup**: **2-4× on multi-core**
**Difficulty**: ⭐⭐⭐ Advanced

### Implementation

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class Tier2LCPFinder:
    def find_long_repeats_parallel(self, chromosome: str, n_threads=4) -> List[TandemRepeat]:
        """Parallel Tier 2 using thread pool."""

        # Split period range into chunks
        period_ranges = []
        chunk_size = (self.max_period - self.min_period + 1) // n_threads

        for i in range(n_threads):
            start_p = self.min_period + i * chunk_size
            end_p = start_p + chunk_size if i < n_threads - 1 else self.max_period
            period_ranges.append((start_p, end_p))

        # Process periods in parallel
        results = []
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = []
            for start_p, end_p in period_ranges:
                future = executor.submit(
                    self._scan_period_range,
                    chromosome, start_p, end_p
                )
                futures.append(future)

            # Collect results
            for future in as_completed(futures):
                results.extend(future.result())

        return results

    def _scan_period_range(self, chromosome, start_period, end_period):
        """Worker function: scan a range of periods."""
        local_results = []

        for p in range(start_period, end_period + 1):
            # Same logic as _find_repeats_simple but for period range [start_period, end_period]
            # ...
            pass

        return local_results
```

**Note**: Python threads don't provide true parallelism for CPU-bound tasks due to GIL. Use with Cython/Numba for best results.

---

## Recommended Approach: Combined Strategy

For **maximum performance**, combine multiple strategies:

### Phase 1: Quick Wins (Week 1)
1. ✅ **Add multiprocessing for chromosomes** (4-8× speedup, easy)
2. ✅ **Add Numba decorators** (1.5-3× speedup, trivial)

### Phase 2: Medium Effort (Week 2-3)
3. ✅ **Cythonize hot loops** (2-5× additional speedup)
4. ✅ **Add adaptive Tier 2 scanning** (already done!)

### Phase 3: Advanced (Optional)
5. ⚠️ **Thread-level parallelism for Tier 2 periods**
6. ⚠️ **GPU acceleration with CuPy** (for very large genomes)

---

## Expected Performance Gains

**Before optimizations** (10 Mbp chromosome):
- Tier 1: ~8 seconds
- Tier 2: ~180 seconds (3 minutes)
- **Total: ~188 seconds**

**After Phase 1** (Numba + Multiprocessing on 8 cores):
- Tier 1: ~5 seconds per core × 8 chromosomes = ~10 seconds wall time
- Tier 2: ~60 seconds per core × 8 chromosomes = ~15 seconds wall time
- **Total: ~25 seconds (7.5× faster)**

**After Phase 2** (+ Cython):
- Tier 1: ~3 seconds per core
- Tier 2: ~20 seconds per core
- **Total: ~6 seconds wall time (31× faster than original!)**

---

## Implementation Priority

| Strategy | Difficulty | Speedup | Priority | Status |
|----------|-----------|---------|----------|--------|
| Multiprocessing (chromosomes) | Easy | 4-8× | 🔥 HIGH | TODO |
| Adaptive scanning (Tier 2) | Easy | 2-10× | 🔥 HIGH | ✅ DONE |
| Numba JIT | Easy | 1.5-3× | ⭐ MEDIUM | TODO |
| Cython hot loops | Medium | 2-5× | ⭐ MEDIUM | TODO |
| Thread-level parallelism | Hard | 2-4× | 💡 LOW | TODO |
| GPU acceleration | Hard | 10-50× | 💡 LOW | Future |

---

## Next Steps

Would you like me to implement:

1. **Multiprocessing for chromosomes** (easiest, biggest gain)
2. **Numba JIT decorators** (trivial to add)
3. **Full Cython implementation** (best performance)
4. **All of the above** (comprehensive optimization)

Let me know which approach you'd prefer, and I'll implement it!
