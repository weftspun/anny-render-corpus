# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Compare two azimuth-recovery runs on the views they both fitted.

`fit_ladder_azimuth.py` excludes a view it could not fit, which is right: a destroyed subject
is not a subject that failed to turn. But two runs then carry two different view sets, and a
slope from one is not comparable with a slope from the other. Comparing them anyway rewards
the run that lost the hardest views.

    python compare_camera_obedience.py --base <a.json> --other <b.json> [--self-test]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

import numpy as np


def load(path):
    d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    fitted = {float(r["asked_deg"]): r for r in d["rows"]}
    lost = {f["file"]: f["error"] for f in d.get("not_fitted", [])}
    return d, fitted, lost


def slope_over(rows, asked_set):
    """Slope of recovered against requested, over a named set of azimuths."""
    picked = [rows[a] for a in sorted(asked_set) if a in rows]
    if len(picked) < 3:
        return float("nan"), len(picked)
    asked = np.array([r["asked_deg"] for r in picked])
    got = np.degrees(np.unwrap(np.radians([r["recovered_deg"] for r in picked])))
    return float(np.polyfit(asked, got, 1)[0]), len(picked)


def compare(base_path, other_path, base_label="base", other_label="other", verbose=True):
    _, b_rows, b_lost = load(base_path)
    _, o_rows, o_lost = load(other_path)

    every = sorted(set(b_rows) | set(o_rows) | _asked_of(b_lost) | _asked_of(o_lost))
    common = sorted(set(b_rows) & set(o_rows))
    only_base = sorted(set(b_rows) - set(o_rows))
    only_other = sorted(set(o_rows) - set(b_rows))
    neither = [a for a in every if a not in b_rows and a not in o_rows]

    better = worse = 0
    table = []
    for a in every:
        b = b_rows.get(a)
        o = o_rows.get(a)
        if b and o:
            delta = o["error_deg"] - b["error_deg"]
            mark = "better" if delta < 0 else ("worse" if delta > 0 else "same")
            better += delta < 0
            worse += delta > 0
        else:
            delta, mark = None, ("lost by %s" % other_label if b else
                                 "lost by %s" % base_label if o else "unscored in both")
        table.append((a, b["error_deg"] if b else None, o["error_deg"] if o else None,
                      delta, mark))

    b_common, n_c = slope_over(b_rows, common)
    o_common, _ = slope_over(o_rows, common)
    b_full, n_bf = slope_over(b_rows, set(b_rows))
    o_full, n_of = slope_over(o_rows, set(o_rows))

    if verbose:
        print("  asked   %-10s %-10s   delta   " % (base_label, other_label))
        for a, be, oe, d, mark in table:
            print("  %5.0f   %-10s %-10s %8s   %s"
                  % (a,
                     "%.1f" % be if be is not None else "--",
                     "%.1f" % oe if oe is not None else "--",
                     "%+.1f" % d if d is not None else "",
                     mark))
        print()
        print("  scored in both      %d of %d azimuths" % (len(common), len(every)))
        print("  lost by %-11s %s" % (other_label,
                                      ", ".join("%.0f" % a for a in only_base) or "none"))
        print("  lost by %-11s %s" % (base_label,
                                      ", ".join("%.0f" % a for a in only_other) or "none"))
        print("  unscored in both    %s"
              % (", ".join("%.0f" % a for a in neither) or "none"))
        print()
        print("  SLOPE ON THE COMMON %d VIEWS   %s %.3f   %s %.3f"
              % (n_c, base_label, b_common, other_label, o_common))
        print("  slope on each run's own views  %s %.3f (n=%d)   %s %.3f (n=%d)"
              % (base_label, b_full, n_bf, other_label, o_full, n_of))
        if n_bf != n_of:
            print("  the two right-hand slopes are over different view sets and are NOT")
            print("  comparable with each other. The common-subset line above is.")
        print()
        print("  per-view: %d better, %d worse, %d lost by %s"
              % (better, worse, len(only_base), other_label))

    return {"common": common, "only_base": only_base, "only_other": only_other,
            "neither": neither, "better": better, "worse": worse,
            "slope_common": (b_common, o_common), "slope_full": (b_full, o_full),
            "n_common": n_c, "n_base": n_bf, "n_other": n_of, "table": table}


def _asked_of(lost):
    out = set()
    for name in lost:
        digits = "".join(c for c in name.split("_")[0] if c.isdigit())
        if digits:
            out.add(float(digits))
    return out


def _write(path, rows, lost=()):
    path.write_text(json.dumps({
        "rows": [{"file": "az%03d_A.png" % a, "asked_deg": float(a),
                  "recovered_deg": float(r), "error_deg": abs(float(r) - a)}
                 for a, r in rows],
        "not_fitted": [{"file": "az%03d_A.png" % a, "error": "no person detected"}
                       for a in lost]}), encoding="utf-8")


def self_test():
    """Four controls. Two must reject a comparison that the old reporting would have passed."""
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp)
        full = [(a, a) for a in (0, 45, 90, 135, 180, 225, 270, 315)]

        _write(d / "a.json", full)
        _write(d / "b.json", full)
        r = compare(d / "a.json", d / "b.json", verbose=False)
        results.append(("identical runs report no better and no worse",
                        r["better"] == 0 and r["worse"] == 0 and r["n_common"] == 8))

        # The planted defect: a run that loses its two hardest views scores a better slope on
        # its own view set while being worse on every view both runs kept.
        _write(d / "c.json", [(a, a + 1) for a, _ in full])
        _write(d / "d.json", [(a, a + 20) for a in (0, 90, 180, 225, 270, 315)],
               lost=(45, 135))
        r = compare(d / "c.json", d / "d.json", verbose=False)
        results.append(("a view lost by one run is excluded from the common slope",
                        r["n_common"] == 6 and r["only_base"] == [45.0, 135.0]))
        results.append(("the common-subset slopes are computed over the same views",
                        r["n_common"] == 6 and not np.isnan(r["slope_common"][0])
                        and not np.isnan(r["slope_common"][1])))
        results.append(("a run worse on every shared view is not reported as an improvement",
                        r["worse"] == 6 and r["better"] == 0))

    bad = sum(1 for _, ok in results if not ok)
    for name, ok in results:
        print("  %-4s control: %s" % ("ok" if ok else "FAIL", name))
    print("  %d of %d controls fired." % (len(results) - bad, len(results)))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base")
    ap.add_argument("--other")
    ap.add_argument("--base-label", default="base")
    ap.add_argument("--other-label", default="adapter")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.base or not args.other:
        return ap.error("--base and --other are required")
    compare(args.base, args.other, args.base_label, args.other_label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
