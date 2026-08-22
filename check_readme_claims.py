"""Re-measure every number the README asserts, and fail if reality has moved.

WHY THIS EXISTS. A README rots silently: the code changes, the numbers stay, and nobody
finds out until someone builds on a figure that stopped being true. This project has
already produced one actively misleading README -- `godot-soma-twist` opened with "no
custom code is needed, Godot already ships the correct solution" long after the shipping
answer had become the opposite -- and one docstring that claimed Delta Mush was "available
here" when grep found zero occurrences of it.

So the README's numbers are not prose. They are tagged claims:

    <!--claim:twist_rmse_90_L=3.6 tol=1.0-->

and this script re-derives each one from the live code and compares. Drift is a test
failure, not a discovery six months later. That is the whole mechanism: being wrong makes
the documentation fail loudly, which is what makes it worth trusting when it passes.

Usage:  python check_readme_claims.py [README.md]
Exit code is non-zero if any claim has drifted, so it can gate CI.

A claim with no measurement function registered is reported as UNVERIFIED rather than
skipped -- a silent skip reads exactly like a pass, which is a failure mode this project
has hit three separate times.
"""

import re
import sys


def measure_twist_rmse(side):
    import torch
    import anny_rig
    model = anny_rig.build_corpus_model(dtype=torch.float64)
    return anny_rig.twist_ramp_rmse(model, side)[0]


def measure_twist_rmse_stock(side):
    import torch
    import anny_rig
    model = anny_rig.build_corpus_model(dtype=torch.float64, apply_twist_fix=False)
    return anny_rig.twist_ramp_rmse(model, side)[0]


def measure_zero_twist_baseline():
    import anny_rig
    return anny_rig.ZERO_TWIST_RMSE


def measure_bone_count():
    import torch
    import anny_rig
    return float(anny_rig.build_corpus_model(dtype=torch.float32).bone_count)


def measure_rest_pose_shift_mm():
    """The re-weighting must not move the rest pose. It redistributes mass between bones
    that are both at identity at rest, so this is a property of the fix, not luck."""
    import numpy as np
    import torch
    import anny_rig
    a = anny_rig.build_corpus_model(dtype=torch.float64, apply_twist_fix=False)
    b = anny_rig.build_corpus_model(dtype=torch.float64)
    pose = anny_rig._identity_pose(a)
    with torch.no_grad():
        va = a(pose_parameters=pose)["rest_vertices"][0].numpy()
        vb = b(pose_parameters=pose)["rest_vertices"][0].numpy()
    return float(np.abs(va - vb).max() * 1000)


def measure_interfaces_total():
    import interface_audit
    interface_audit.RESULTS.clear()
    try:
        interface_audit.main()
    except SystemExit:
        pass
    return float(len(interface_audit.RESULTS))


def measure_interfaces_unchecked():
    import interface_audit
    interface_audit.RESULTS.clear()
    try:
        interface_audit.main()
    except SystemExit:
        pass
    return float(sum(1 for r in interface_audit.RESULTS if r[1] == "UNCHECKED"))


def measure_bvh_clip_count():
    import glob
    root = ("O:/Documents/Datasets/dataset-100style-mocap/unpacked/100STYLE")
    return float(len(glob.glob(root + "/*/*.bvh")))


MEASUREMENTS = {
    "twist_rmse_90_L": lambda: measure_twist_rmse("L"),
    "twist_rmse_90_R": lambda: measure_twist_rmse("R"),
    "twist_rmse_stock_L": lambda: measure_twist_rmse_stock("L"),
    "zero_twist_baseline": measure_zero_twist_baseline,
    "anny_bone_count": measure_bone_count,
    "rest_pose_shift_mm": measure_rest_pose_shift_mm,
    "interfaces_total": measure_interfaces_total,
    "interfaces_unchecked": measure_interfaces_unchecked,
    "bvh_clip_count": measure_bvh_clip_count,
}

CLAIM_RE = re.compile(r"<!--\s*claim:(\w+)=([-\d.]+)(?:\s+tol=([\d.]+))?\s*-->")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "README.md"
    with open(path, encoding="utf-8") as fh:
        text = fh.read()

    claims = CLAIM_RE.findall(text)
    if not claims:
        print("no tagged claims found in %s -- README numbers are unverifiable" % path)
        return 1

    print("%-24s %12s %12s %10s  %s" % ("claim", "README", "measured", "tol", "verdict"))
    print("-" * 76)
    bad = 0
    import io
    import contextlib
    for name, stated, tol in claims:
        stated = float(stated)
        tol = float(tol) if tol else max(abs(stated) * 0.05, 0.01)
        fn = MEASUREMENTS.get(name)
        if fn is None:
            # Not skipped: an unregistered claim is a hole in the check, and a silent
            # skip is indistinguishable from a pass.
            print("%-24s %12.3f %12s %10s  UNVERIFIED (no measurement registered)"
                  % (name, stated, "-", "-"))
            bad += 1
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                got = float(fn())
        except Exception as exc:                                   # noqa: BLE001
            print("%-24s %12.3f %12s %10s  ERROR %s"
                  % (name, stated, "-", "-", type(exc).__name__))
            bad += 1
            continue
        ok = abs(got - stated) <= tol
        bad += 0 if ok else 1
        print("%-24s %12.3f %12.3f %10.3f  %s"
              % (name, stated, got, tol, "ok" if ok else "DRIFTED"))

    print()
    if bad:
        print("%d claim(s) drifted or unverified -- the README is making statements the"
              % bad)
        print("code no longer supports. Fix the README or fix the code; do not ignore it.")
    else:
        print("all %d claims re-derived from live code and still hold." % len(claims))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
