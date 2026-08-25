"""Did the body turn? Recover an azimuth from each generated view and compare it to the ask.

THE GATE ON THE WHOLE SWEEP. `ladder_camera_obedience.py` asks OmniGen2 for eight named camera
directions. Whether it obeyed is not a question about how the pictures look, so this does not
look at them: it detects keypoints, fits ANNY, and reads the azimuth off the fitted body. If
requested azimuth does not predict recovered azimuth, generating 96 views would produce 96
copies of one pose and the sweep should not run.

WHY THE FIT AND NOT A SHORTCUT. Shoulder width alone collapses as the subject turns, which
looks like a measurement until you notice it cannot tell front from back -- both give a wide
shoulder line, and a body facing away scores the same as one facing you. The fit does not have
that ambiguity because the keypoints are labelled: when `left_shoulder` appears to the right of
`right_shoulder` in image space, the subject is seen from behind, and the solver uses that.

WHAT IS BEING READ. `fit_2d` optimises pose rotations plus a weak-perspective camera, and the
camera it fits carries scale and translation but no orientation. So a change of viewpoint is
absorbed by the body rotating instead, and the recovered azimuth is the fitted subject's own
facing direction in the horizontal plane, measured against the same subject fitted to the
source frame rather than against an absolute zero.

BASELINE, BECAUSE A NUMBER WITHOUT ONE IS NOT A MEASUREMENT. The source frame is fitted too,
and its recovered azimuth is subtracted from every other. That removes any constant offset in
how the fit reads a facing, so what remains is the change the camera instruction produced.

    python fit_ladder_azimuth.py --dir <ladder dir> --source <front-view.png>
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]
                       / "7-service" / "service-livebook" / "priv" / "python"))

# A run this far from the ask is not a turn, it is a coincidence. Stated before the numbers are
# read so it is a threshold rather than a rationalisation of whatever came out.
AGREEMENT_DEG = 30.0


def facing_azimuth(points3d, names):
    """The subject's facing, in degrees, from the shoulder and hip lines.

    Both lines are used because either alone can be degenerate: shoulders in a shrug, hips in
    a twist. Their average is the torso's own left-right axis, and the facing is perpendicular
    to it in the horizontal plane.
    """
    index = {n: i for i, n in enumerate(names)}
    need = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    missing = [n for n in need if n not in index]
    if missing:
        raise KeyError("the fit has no %s, so a facing cannot be read" % missing)
    across = ((points3d[index["left_shoulder"]] - points3d[index["right_shoulder"]])
              + (points3d[index["left_hip"]] - points3d[index["right_hip"]])) / 2.0
    # Up is the largest rest extent by convention here, and the horizontal plane is the other
    # two components. `image_axes` measures which those are rather than assuming Z.
    return across


def azimuth_from(across, right_axis, depth_axis):
    return math.degrees(math.atan2(float(across[depth_axis]), float(across[right_axis])))


def wrap(delta):
    """Into (-180, 180]. A turn from 350 to 10 degrees is 20, not 340."""
    return (delta + 180.0) % 360.0 - 180.0


def recover(path, model, regressor, names, axes):
    import torch
    from loop1_fit import detect_keypoints, fit_2d

    xy, confidence, detected = detect_keypoints(str(path), threshold=0.3)
    target = torch.tensor(np.asarray(xy), dtype=torch.float64)
    result = fit_2d(model, target, list(detected),
                    confidence=torch.tensor(np.asarray(confidence), dtype=torch.float64),
                    regressor=regressor)
    # `fit_2d` returns the 4x4 `pose_parameters` it solved, not a rotation vector, and the
    # model takes `pose_parameters=`. Calling it `pose_rotvec=` raised rather than quietly
    # fitting an unposed body, which is the failure mode worth having.
    with torch.no_grad():
        posed = model(pose_parameters=result["pose"])
        points = regressor(posed)[0].detach().cpu().numpy()
    return facing_azimuth(points, names)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--source", required=True, help="the frame the views were generated from")
    ap.add_argument("--condition", default="A")
    ap.add_argument("--agreement", type=float, default=AGREEMENT_DEG)
    args = ap.parse_args()

    import torch
    from loop1_fit import build_model, coco_regressor, image_axes

    model = build_model()
    regressor, names = coco_regressor(model)
    right, up = image_axes(model)
    depth = [a for a in (0, 1, 2) if a not in (right, up)][0]
    axes = (right, up)

    frames = sorted(glob.glob(str(pathlib.Path(args.dir) / ("az*_%s.png" % args.condition))))
    if not frames:
        sys.exit("FAIL  no az*_%s.png under %s" % (args.condition, args.dir))

    rows, failed = [], []
    try:
        base_across = recover(args.source, model, regressor, names, axes)
        base = azimuth_from(base_across, right, depth)
    except Exception as error:  # noqa: BLE001
        sys.exit("FAIL  the source frame could not be fitted, so there is no baseline to "
                 "subtract and every number below would be uncalibrated: %s: %s"
                 % (type(error).__name__, error))
    print("baseline from the source frame: %.1f deg (subtracted from every row)" % base)

    for frame in frames:
        asked = float(re.search(r"az(\d+)_", pathlib.Path(frame).name).group(1))
        try:
            across = recover(frame, model, regressor, names, axes)
            got = wrap(azimuth_from(across, right, depth) - base)
        except Exception as error:  # noqa: BLE001
            # A VIEW WITH NO DETECTION IS NOT A VIEW THAT DID NOT TURN. Counting it as zero
            # would pull the fit toward "obeyed" for exactly the frames where the subject was
            # destroyed, so it is named and excluded from the correlation instead.
            failed.append((pathlib.Path(frame).name,
                           "%s: %s" % (type(error).__name__, str(error)[:90])))
            print("  --   %-16s asked %5.1f  NO FIT  %s"
                  % (pathlib.Path(frame).name, asked, str(error)[:60]))
            continue
        error_deg = abs(wrap(got - asked))
        rows.append({"file": pathlib.Path(frame).name, "asked_deg": asked,
                     "recovered_deg": round(got, 1), "error_deg": round(error_deg, 1)})
        print("  ok   %-16s asked %5.1f  recovered %6.1f  off by %5.1f"
              % (pathlib.Path(frame).name, asked, got, error_deg))

    print()
    if len(rows) < 3:
        print("BAD  only %d view(s) fitted; that is too few to say whether the body turns, "
              "and the missing ones are listed above rather than treated as zeros" % len(rows))
        verdict = "undecided"
        slope = float("nan")
    else:
        asked = np.array([r["asked_deg"] for r in rows])
        got = np.unwrap(np.radians([r["recovered_deg"] for r in rows]))
        got = np.degrees(got)
        slope = float(np.polyfit(asked, got, 1)[0])
        within = sum(1 for r in rows if r["error_deg"] <= args.agreement)
        print("slope of recovered against requested: %.2f   (1.00 is perfect obedience, "
              "0.00 is a body that never turned)" % slope)
        print("%d of %d views within %.0f degrees of the ask" % (within, len(rows),
                                                                 args.agreement))
        verdict = ("the body turns" if slope > 0.5 and within >= len(rows) * 0.6
                   else "the body does NOT follow the camera instruction")
        print("VERDICT: %s" % verdict)

    out = pathlib.Path(args.dir) / ("azimuth_recovery_%s.json" % args.condition)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"condition": args.condition, "baseline_deg": round(base, 1),
                   "agreement_deg": args.agreement, "slope": slope,
                   "verdict": verdict, "rows": rows,
                   "not_fitted": [{"file": f, "error": e} for f, e in failed]}, fh, indent=2)
    print("wrote %s" % out)
    print("%d view(s) could not be fitted and are excluded, not counted as zero" % len(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
