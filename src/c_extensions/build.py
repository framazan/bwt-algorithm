"""Build C extensions for bwtandem."""
import os
import sys
import ctypes
from ctypes import c_int, c_double, POINTER

_EXT_DIR = os.path.dirname(os.path.abspath(__file__))


def _lib_path(name):
    ext = '.dylib' if sys.platform == 'darwin' else '.so'
    return os.path.join(_EXT_DIR, f'lib{name}{ext}')


def get_lib_path():
    """Get the path to the Tier 1 shared library."""
    return _lib_path('tier1_scan')


_SOURCES = (
    ('tier1_scan.c', 'tier1_scan'),
    ('tier2_accel.c', 'tier2_accel'),
    ('align_accel.c', 'align_accel'),
    ('bwt_accel.c', 'bwt_accel'),
)


def _compile(src_name, lib_name):
    """Compile a single C source to a shared library."""
    import subprocess
    src = os.path.join(_EXT_DIR, src_name)
    out = _lib_path(lib_name)
    cmd = ['gcc', '-O3', '-shared', '-fPIC', '-std=c99', '-o', out, src]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"C extension build failed ({src_name}): {result.stderr}", file=sys.stderr)
        return False
    return True


def _is_stale(src_name, lib_name):
    """True if the shared library is missing or older than its C source.

    The libraries are gitignored build artefacts, so a checkout that already has
    them would otherwise keep running an old binary after a `git pull` that
    changed the `.c` — silently, since nothing else notices.
    """
    out = _lib_path(lib_name)
    if not os.path.exists(out):
        return True
    return os.path.getmtime(os.path.join(_EXT_DIR, src_name)) > os.path.getmtime(out)


def build(force=False):
    """Compile any C extension whose source is newer than its library."""
    results = [_compile(src, lib) for src, lib in _SOURCES
               if force or _is_stale(src, lib)]
    return all(results)


def _ensure(lib_name, src_name):
    """Build `lib_name` if its library is missing or stale. Returns True on success."""
    if _is_stale(src_name, lib_name):
        return _compile(src_name, lib_name)
    return True


def _ensure_from_path(lib_path):
    """Build/refresh the library at `lib_path` if its C source is newer."""
    for src, lib in _SOURCES:
        if _lib_path(lib) == lib_path:
            return _ensure(lib, src)
    return os.path.exists(lib_path)


def load():
    """Load the Tier 1 C library, building if necessary."""
    lib_path = get_lib_path()
    if not _ensure_from_path(lib_path):
        return None
    try:
        lib = ctypes.CDLL(lib_path)
        lib.find_period_runs.restype = c_int
        lib.find_period_runs.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),  # text
            c_int,                            # n
            c_int,                            # k
            c_int,                            # min_seed_copies
            ctypes.POINTER(ctypes.c_ubyte),  # seen
            ctypes.POINTER(c_int),           # out_starts
            ctypes.POINTER(c_int),           # out_ends
            ctypes.POINTER(c_int),           # out_copies
            c_int,                            # max_candidates
        ]
        return lib
    except OSError as e:
        print(f"Failed to load C extension: {e}", file=sys.stderr)
        return None


def load_tier2():
    """Load the Tier 2 C library, building if necessary."""
    lib_path = _lib_path('tier2_accel')
    if not _ensure_from_path(lib_path):
        return None
    try:
        lib = ctypes.CDLL(lib_path)

        lib.smallest_period_str.restype = c_int
        lib.smallest_period_str.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte), c_int
        ]

        lib.smallest_period_str_approx.restype = c_int
        lib.smallest_period_str_approx.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte), c_int, c_double
        ]

        lib.hamming_distance.restype = c_int
        lib.hamming_distance.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.POINTER(ctypes.c_ubyte),
            c_int
        ]

        lib.batch_process_lcp_candidates.restype = c_int
        lib.batch_process_lcp_candidates.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),  # text
            c_int,                            # n
            ctypes.POINTER(c_int),           # periods
            ctypes.POINTER(c_int),           # seed_positions
            c_int,                            # n_candidates
            c_double,                         # max_mismatch_rate
            c_int,                            # min_copies
            ctypes.POINTER(ctypes.c_ubyte),  # covered_mask
            ctypes.POINTER(c_int),           # out_starts
            ctypes.POINTER(c_int),           # out_ends
            ctypes.POINTER(c_int),           # out_periods
            ctypes.POINTER(c_int),           # out_copies
            c_int,                            # max_results
        ]

        return lib
    except OSError as e:
        print(f"Failed to load Tier 2 C extension: {e}", file=sys.stderr)
        return None


def load_bwt():
    """Load the BWT acceleration C library, building if necessary."""
    lib_path = _lib_path('bwt_accel')
    if not _ensure_from_path(lib_path):
        return None
    try:
        lib = ctypes.CDLL(lib_path)

        lib.count_equal_range.restype = c_int
        lib.count_equal_range.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte), c_int, c_int, c_int
        ]

        lib.backward_search.restype = c_int
        lib.backward_search.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),  # bwt
            c_int,                            # n
            ctypes.POINTER(ctypes.c_ubyte),  # pattern
            c_int,                            # pat_len
            ctypes.POINTER(c_int),           # char_counts[256]
            ctypes.POINTER(c_int),           # char_totals[256]
            ctypes.POINTER(c_int),           # checkpoints_flat
            ctypes.POINTER(c_int),           # cp_offsets[256]
            ctypes.POINTER(c_int),           # cp_lengths[256]
            c_int,                            # sample_rate
            ctypes.POINTER(c_int),           # out_sp
            ctypes.POINTER(c_int),           # out_ep
        ]

        lib.batch_backward_search.restype = None
        lib.batch_backward_search.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),  # bwt
            c_int,                            # n
            ctypes.POINTER(ctypes.c_ubyte),  # patterns (concatenated)
            ctypes.POINTER(c_int),           # pat_offsets
            ctypes.POINTER(c_int),           # pat_lengths
            c_int,                            # n_patterns
            ctypes.POINTER(c_int),           # char_counts[256]
            ctypes.POINTER(c_int),           # char_totals[256]
            ctypes.POINTER(c_int),           # checkpoints_flat
            ctypes.POINTER(c_int),           # cp_offsets[256]
            ctypes.POINTER(c_int),           # cp_lengths[256]
            c_int,                            # sample_rate
            ctypes.POINTER(c_int),           # out_sps
            ctypes.POINTER(c_int),           # out_eps
        ]

        lib.kasai_lcp.restype = None
        lib.kasai_lcp.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),  # text_codes
            ctypes.POINTER(c_int),           # sa
            c_int,                            # n
            ctypes.POINTER(c_int),           # lcp_out
        ]

        return lib
    except OSError as e:
        print(f"Failed to load bwt_accel C extension: {e}", file=sys.stderr)
        return None


def load_align():
    """Load the alignment acceleration C library, building if necessary."""
    lib_path = _lib_path('align_accel')
    if not _ensure_from_path(lib_path):
        return None
    try:
        lib = ctypes.CDLL(lib_path)

        # AlignRegionResult struct layout: 5 ints + 1 double + padding
        class AlignRegionResult(ctypes.Structure):
            _fields_ = [
                ('copies', c_int),
                ('consumed_length', c_int),
                ('total_mismatches', c_int),
                ('total_insertions', c_int),
                ('total_deletions', c_int),
                ('max_errors_per_copy', c_int),
                ('mismatch_rate', c_double),
            ]

        lib.AlignRegionResult = AlignRegionResult

        lib.align_repeat_region_c.restype = c_int
        lib.align_repeat_region_c.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),  # text
            c_int,                            # text_len
            c_int,                            # start
            c_int,                            # end
            ctypes.POINTER(ctypes.c_ubyte),  # motif
            c_int,                            # motif_len
            c_double,                         # mismatch_frac
            c_int,                            # max_indel
            c_int,                            # min_copies
            ctypes.POINTER(AlignRegionResult), # out
            ctypes.POINTER(ctypes.c_ubyte),  # consensus_out
        ]

        return lib
    except OSError as e:
        print(f"Failed to load align_accel C extension: {e}", file=sys.stderr)
        return None


if __name__ == '__main__':
    if build():
        print("All C extensions built successfully.")
    else:
        sys.exit(1)
