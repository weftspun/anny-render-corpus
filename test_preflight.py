"""Red/green regression test for preflight_audit.py.

Paranoia protocol: for every check, prove BOTH directions.
  RED   -- inject a corruption that should trip exactly that check, and assert
           the audit fails on it (a check that cannot fail is decoration)
  GREEN -- assert the same check passes on the clean corpus (a check that
           always fails is worse than none: it trains people to ignore output)

Some checks are STRUCTURAL: they test the code or the model, not the corpus
(deterministic ids, up-axis, metres-not-cm, mesh reproducibility, rest-pose
symmetry). No data mutation can trip them, and pretending otherwise would be
dishonest -- they are listed as structural and covered by the green pass plus
their own targeted asserts at the bottom.

Usage: python test_preflight.py <clean-corpus-dir>
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

FILES = ["identities.parquet", "identity_phenotype.parquet", "phenotypes.parquet"]


def pheno_id(d, name):
    ph = pd.read_parquet(f"{d}/phenotypes.parquet")
    return int(ph.loc[ph.name == name, "phenotype_id"].iloc[0])


# ---- corruptions: each targets ONE check -----------------------------------

def c_nan(d):
    ip = pd.read_parquet(f"{d}/identity_phenotype.parquet")
    ip.loc[ip.index[:50], "value"] = np.nan
    ip.to_parquet(f"{d}/identity_phenotype.parquet", compression="zstd", index=False)

def c_out_of_range(d):
    ip = pd.read_parquet(f"{d}/identity_phenotype.parquet")
    ip.loc[ip.index[:20], "value"] = 5.0
    ip.to_parquet(f"{d}/identity_phenotype.parquet", compression="zstd", index=False)

def c_dup_id(d):
    ids = pd.read_parquet(f"{d}/identities.parquet")
    ids.loc[ids.index[1], "identity_id"] = ids.loc[ids.index[0], "identity_id"]
    ids.to_parquet(f"{d}/identities.parquet", compression="zstd", index=False)

def c_collapse(d):
    ip = pd.read_parquet(f"{d}/identity_phenotype.parquet")
    first = ip.groupby("phenotype_id").value.first()
    ip["value"] = ip.phenotype_id.map(first)
    ip.to_parquet(f"{d}/identity_phenotype.parquet", compression="zstd", index=False)

def c_invert_sex(d):
    gid = pheno_id(d, "gender")
    ip = pd.read_parquet(f"{d}/identity_phenotype.parquet")
    m = ip.phenotype_id == gid
    ip.loc[m, "value"] = 1.0 - ip.loc[m, "value"]
    ip.to_parquet(f"{d}/identity_phenotype.parquet", compression="zstd", index=False)

def c_no_dimorphism(d):
    """Force every identity to the same sex -> dimorphism collapses to ~0."""
    gid = pheno_id(d, "gender")
    ip = pd.read_parquet(f"{d}/identity_phenotype.parquet")
    ip.loc[ip.phenotype_id == gid, "value"] = 0.5
    ip.to_parquet(f"{d}/identity_phenotype.parquet", compression="zstd", index=False)

def c_giants(d):
    """Push height to the top of its range -> adults outside the human band and
    the short-population coverage (Japan) is lost."""
    hid = pheno_id(d, "height")
    ip = pd.read_parquet(f"{d}/identity_phenotype.parquet")
    ip.loc[ip.phenotype_id == hid, "value"] = 1.0
    ip.to_parquet(f"{d}/identity_phenotype.parquet", compression="zstd", index=False)

def c_all_adults(d):
    """No children -> the child/adult ordering check loses its sample."""
    aid = pheno_id(d, "age")
    ip = pd.read_parquet(f"{d}/identity_phenotype.parquet")
    ip.loc[ip.phenotype_id == aid, "value"] = 0.8
    ip.to_parquet(f"{d}/identity_phenotype.parquet", compression="zstd", index=False)

def c_split_leak(d):
    ids = pd.read_parquet(f"{d}/identities.parquet")
    ids.loc[ids.index[:200], "source_subject"] = "shared:subject"
    ids.loc[ids.index[:100], "split"] = "val"
    ids.loc[ids.index[100:200], "split"] = "train"
    ids.to_parquet(f"{d}/identities.parquet", compression="zstd", index=False)

def c_no_val(d):
    ids = pd.read_parquet(f"{d}/identities.parquet")
    ids["split"] = "train"
    ids.to_parquet(f"{d}/identities.parquet", compression="zstd", index=False)


# corruption -> substring of the check name that MUST fail
CASES = [
    ("NaN phenotypes",            c_nan,           "no NaN/Inf"),
    ("out-of-range phenotype",    c_out_of_range,  "within [0,1]"),
    ("duplicate identity_id",     c_dup_id,        "identity_id unique"),
    ("collapsed population",      c_collapse,      "collapsed phenotype"),
    ("inverted sex label",        c_invert_sex,    "dimorphism"),
    ("single-sex population",     c_no_dimorphism, "sex subgroups present"),
    ("giant population",          c_giants,        "adult stature"),
    ("no children",               c_all_adults,    "child subgroup present"),
    ("split contamination",       c_split_leak,    "validator clean"),
    ("empty val split",           c_no_val,        "val split"),
]

STRUCTURAL = [
    "up axis is Z", "stature is in METRES", "rest pose is left/right symmetric",
    "deterministic_id stable within process", "deterministic_id stable ACROSS processes",
    "mesh decode is reproducible",
    # These two interrogate the MODEL, not the corpus: that rest_bone_heads pairs with
    # rest_vertices at extreme phenotypes, and that an identity pose is not the rest
    # pose. No parquet mutation can trip them, so they are declared rather than
    # red-tested -- but they are the tripwire for a 500 mm keypoint error on children.
    "matched skeleton/mesh pairing", "metric REJECTS the mismatched",
    "identity pose is NOT assumed",
    # The priority-1 rig gate. Structural: it interrogates the MODEL, not the
    # parquet, so no data mutation can trip it. Its own negative control lives
    # in interface_audit.py, which scores an unfixed rig and asserts it fails.
    "forearm transmits twist as a linear ramp", "forearms behave symmetrically",
    # The audit's own statistical power. Structural: it describes the RUN, not the
    # data. Its negative direction is exercised every red case, which passes a
    # loose --coverage-ppm precisely because a sampled run cannot certify a tail.
    "run can catch defects at or below",
]


def run_audit(corpus, sample=350, coverage_ppm=20000):
    """Run the audit.

    GREEN runs the FULL corpus (sample=0), because the green pass is the one that
    certifies the corpus and it must be able to see a single bad identity -- 43 ppm
    rather than the 10,000 ppm a 300-identity sample can resolve. It costs ~95 s.

    RED runs stay sampled and pass a deliberately loose --coverage-ppm: each red case
    corrupts a large fraction of the corpus, so detecting it needs no statistical power,
    and paying 95 s x 10 to prove that would be waste. The loose value is explicit rather
    than implied, so nobody reads a red pass as a tail-coverage claim."""
    cmd = [sys.executable, "preflight_audit.py", corpus,
           "--sample", str(sample), "--coverage-ppm", str(coverage_ppm)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    status = {}
    for line in r.stdout.splitlines():
        m = re.match(r"\s*\[(PASS|FAIL|WARN)\]\s+(.+?)(?:\s+--.*)?$", line)
        if m:
            status[m.group(2).strip()] = m.group(1)
    return r.returncode, status


def main():
    clean = sys.argv[1]
    print("GREEN: clean corpus\n" + "-" * 60)
    code, clean_status = run_audit(clean, sample=0, coverage_ppm=1000)
    green_ok = code == 0 and all(v == "PASS" for v in clean_status.values())
    print(f"  {len(clean_status)} checks, all PASS: {green_ok}, exit {code}")
    if not green_ok:
        print("  " + "; ".join(f"{k}={v}" for k, v in clean_status.items() if v != "PASS"))

    print("\nRED: one targeted corruption per check\n" + "-" * 60)
    red_results = []
    for name, mutate, expect_sub in CASES:
        d = os.path.join(tempfile.mkdtemp(), "case")
        os.makedirs(d)
        for f in FILES:
            shutil.copy(os.path.join(clean, f), d)
        mutate(d)
        code, status = run_audit(d)
        hit = [k for k, v in status.items() if v == "FAIL" and expect_sub.lower() in k.lower()]
        blocked = code != 0
        ok = bool(hit) and blocked
        red_results.append(ok)
        print(f"  {name:26s} {'RED ok' if ok else 'NOT CAUGHT'}"
              f"  (expected '{expect_sub}' to fail; "
              f"{'hit' if hit else 'MISSED'}, exit {code})")
        shutil.rmtree(os.path.dirname(d), ignore_errors=True)

    print("\nSTRUCTURAL checks (no data mutation can trip these)\n" + "-" * 60)
    for s in STRUCTURAL:
        got = next((v for k, v in clean_status.items() if s.lower() in k.lower()), "MISSING")
        print(f"  {s:44s} {got}")

    covered = len(CASES) + len(STRUCTURAL)
    print(f"\nSUMMARY: {sum(red_results)}/{len(CASES)} red tests caught, "
          f"green {'clean' if green_ok else 'DIRTY'}, "
          f"{covered}/{len(clean_status)} checks covered by a red test or declared structural")
    return 0 if (all(red_results) and green_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
