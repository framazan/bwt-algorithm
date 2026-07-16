"""The compiled and pure-Python accelerator paths must agree, byte for byte.

`src/accelerators.py` binds `extend_with_mismatches`, `find_periodic_runs` and
friends to the Cython extension when it is importable and to Python fallbacks
otherwise. Those fallbacks used to be degenerate stubs — `return None` and
`return []` — so a build without the extension detected zero Tier-2 and zero
Tier-3 repeats and exited 0. Nothing caught it: the Tier-2/3 ground-truth cases
were skipped precisely when the extension was missing.

These tests run the whole pipeline twice over the fixtures, once per path, and
require the emitted BED and TRF output to be identical. Each run needs its own
process because `accelerators` binds its symbols at import time.
"""
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(REPO_ROOT, "tests", "fixtures")
FIXTURE_NAMES = ["tier1", "tier2", "tier3", "mixed", "adjacent"]

# Every emitted format, because each formatter reads different fields off
# TandemRepeat: bed the span, trf the consensus/variations/percent-match, vcf the
# derived REF, strfinder the allele coverage.
FORMAT_EXTENSIONS = {"bed": ".bed", "trf": ".dat", "vcf": ".vcf", "strfinder": ".csv"}

try:
    from src import _accelerators as _native  # noqa: F401
    NATIVE_BUILT = True
except Exception:
    NATIVE_BUILT = False

needs_native = pytest.mark.skipif(
    not NATIVE_BUILT,
    reason="compiled src/_accelerators is absent, so there is only one path to compare; "
           "build it (see CLAUDE.md) to run the parity check",
)


def run_pipeline(fasta: str, out_prefix: str, fmt: str, disable_native: bool) -> str:
    """Run src.main in a subprocess and return the contents of its output file."""
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if disable_native:
        env["BWT_DISABLE_NATIVE"] = "1"
    else:
        env.pop("BWT_DISABLE_NATIVE", None)

    proc = subprocess.run(
        [sys.executable, "-m", "src.main", fasta, "--format", fmt, "-o", out_prefix],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, f"src.main failed: {proc.stderr}"

    with open(out_prefix + FORMAT_EXTENSIONS[fmt]) as fh:
        return fh.read()


@needs_native
@pytest.mark.parametrize("fixture", FIXTURE_NAMES)
@pytest.mark.parametrize("fmt", sorted(FORMAT_EXTENSIONS))
def test_native_and_fallback_agree(tmp_path, fixture, fmt):
    """The two accelerator paths detect exactly the same repeats, in every format."""
    fasta = os.path.join(FIXTURES, f"synth_{fixture}.fa")

    native = run_pipeline(fasta, str(tmp_path / "native"), fmt, disable_native=False)
    fallback = run_pipeline(fasta, str(tmp_path / "fallback"), fmt, disable_native=True)

    assert native == fallback, (
        f"synth_{fixture} .{fmt}: compiled and pure-Python paths disagree "
        f"({len(native.splitlines())} vs {len(fallback.splitlines())} records)"
    )


def test_fallback_is_not_empty(tmp_path):
    """Guard the original defect: the fallback must not silently find nothing.

    A parity test alone would still pass if BOTH paths regressed to empty, and
    this one is meaningful even where the extension was never built.
    """
    fasta = os.path.join(FIXTURES, "synth_mixed.fa")
    fallback = run_pipeline(fasta, str(tmp_path / "fallback"), "bed", disable_native=True)
    assert len(fallback.splitlines()) >= 5, "pure-Python path detected almost nothing"


def test_rolling_extender_refuses_to_run_without_the_extension():
    """TIER1_EXT_ROLLING=1 exists only in the .pyx; the fallback must not pretend."""
    env = dict(os.environ)
    env["BWT_DISABLE_NATIVE"] = "1"
    env["TIER1_EXT_ROLLING"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    proc = subprocess.run(
        [sys.executable, "-c",
         "import numpy as np;"
         "from src.accelerators import extend_with_mismatches as e;"
         "e(np.zeros(10, dtype=np.uint8), 0, 2, 10, 0.1)"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode != 0
    assert "TIER1_EXT_ROLLING" in proc.stderr


@pytest.mark.parametrize("raw,rolling", [("0", False), ("00", False), (" 1 ", True), ("1", True)])
def test_rolling_flag_parses_like_the_pyx(monkeypatch, raw, rolling):
    """`bool(int(raw))`, same as _accelerators.pyx reads it at its own import.

    A looser parse here would mean `TIER1_EXT_ROLLING=00` picks the default
    extender when the extension is built and blows up when it is not.
    """
    from src import accelerators as acc

    monkeypatch.setenv("TIER1_EXT_ROLLING", raw)
    assert acc._rolling_extender_requested() is rolling


@pytest.mark.parametrize("raw", ["", "false", "yes"])
def test_rolling_flag_rejects_non_integers_like_the_pyx(monkeypatch, raw):
    from src import accelerators as acc

    monkeypatch.setenv("TIER1_EXT_ROLLING", raw)
    with pytest.raises(ValueError):
        acc._rolling_extender_requested()


def test_import_survives_warnings_as_errors_without_the_extension():
    """`-W error` must not make the package unimportable; the fallback is correct."""
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("BWT_DISABLE_NATIVE", None)
    # `None` in sys.modules makes the import of that name raise ImportError,
    # which hides the compiled extension without touching the file on disk.
    stub = (
        "import sys\n"
        "sys.modules['src._accelerators'] = None\n"
        "import warnings; warnings.simplefilter('error')\n"
        "import src.accelerators as a\n"
        "assert a.NATIVE_AVAILABLE is False\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", stub], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"import died under warnings-as-errors: {proc.stderr}"
    assert "ok" in proc.stdout


def test_all_public_accelerators_are_bound_in_both_modes():
    """Every symbol the tiers import must exist on both paths."""
    consumed = ["extend_with_mismatches", "find_periodic_runs", "lcp_tandem_candidates",
                "align_unit_to_window", "anchor_scan_boundaries"]
    for disable in ("0", "1"):
        env = dict(os.environ)
        env["BWT_DISABLE_NATIVE"] = disable
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.run(
            [sys.executable, "-c",
             f"import src.accelerators as a; [getattr(a, n) for n in {consumed!r}]"],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, f"BWT_DISABLE_NATIVE={disable}: {proc.stderr}"
