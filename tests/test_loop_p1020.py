"""Phase-2 lever (recall loop): a diverged period-13 array must be detected.

The period-10-20 notch: Tier2's exact-seeding paths (LCP plateau >=8bp, exact
9-mer recurrence) miss arrays whose every copy carries a substitution, because
no exact seed ever forms. The autocorrelation seeder (TIER2_APPROX_SEED=1)
detects local periodicity directly and must recover such arrays.

Run standalone:  python tests/test_loop_p1020.py
Run via pytest:  pytest tests/test_loop_p1020.py -q
"""
import os
import subprocess
import sys
import tempfile

PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOTIF = "ACGTGACTGACAT"      # 13 bp, primitive
N_COPIES = 40
SPAN = len(MOTIF) * N_COPIES  # 520 bp


def _diverged_array():
    """One substitution per copy at a cycling position -> ~8% divergence,
    no exact 9-mer recurs at period spacing (defeats exact seeding)."""
    sub = {"A": "C", "C": "G", "G": "T", "T": "A"}
    out = []
    for i in range(N_COPIES):
        m = list(MOTIF)
        pos = i % len(MOTIF)
        m[pos] = sub[m[pos]]
        out.append("".join(m))
    # flank with random-ish non-repetitive DNA so boundaries are real
    flank = "GCTAGCTAGCTTACCGATGCAT"
    return flank + "".join(out) + flank


def _run(approx: bool):
    seq = ">t\n" + _diverged_array() + "\n"
    d = tempfile.mkdtemp()
    fa = os.path.join(d, "t.fa")
    open(fa, "w").write(seq)
    env = dict(os.environ, TIER2_SHORT_REQ_COPIES="2")
    if approx:
        env["TIER2_APPROX_SEED"] = "1"
    else:
        env.pop("TIER2_APPROX_SEED", None)
    subprocess.run(
        [PY, "-m", "src.main", fa, "--min-period", "10", "--max-period", "20",
         "--tiers", "tier2", "--format", "bed", "-o", os.path.join(d, "t")],
        cwd=REPO, env=env, check=True, capture_output=True,
    )
    bed = os.path.join(d, "t.bed")
    rows = [l.split("\t") for l in open(bed)] if os.path.exists(bed) else []
    # best single call covering the array core (allow boundary slack)
    best = 0
    for r in rows:
        if len(r) >= 3:
            cov = int(r[2]) - int(r[1])
            best = max(best, cov)
    return best


def test_approx_seed_detects_diverged_p13():
    cov = _run(approx=True)
    assert cov >= SPAN * 0.25, f"approx seeder failed: best call covers {cov}bp of {SPAN}"


if __name__ == "__main__":
    base = _run(approx=False)
    approx = _run(approx=True)
    print(f"baseline best-call coverage = {base}bp / {SPAN}")
    print(f"approx   best-call coverage = {approx}bp / {SPAN}")
    assert approx >= SPAN * 0.25, "FAIL: approx seeder did not recover the diverged array"
    print("PASS")
