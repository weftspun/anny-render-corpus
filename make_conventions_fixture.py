"""Rebuild `fixtures-conventions.usda`, the rig `check_conventions.py --self-test` runs on.

The gate's six negative controls each break one convention and require it to be rejected,
and its positive control requires a clean rig to be accepted. Both need a rig, and the
repository had none, so the whole suite was unrunnable.

It is the canonical model at identity, not a hand-built stand-in: the controls are only
worth anything if the thing they accept is the thing the corpus renders.

TWO ENVIRONMENTS, because no one of them has both halves. `anny` has torch and the model;
`usd` has pxr. Stage one writes an npz, stage two archives it:

    pixi run -e anny python make_conventions_fixture.py --npz build/rig.npz
    pixi run -e usd  python mesh_to_usda.py build/rig.npz fixtures-conventions.usda \
        --names build/rig.names.json --subject conventions-fixture

The npz is a build artifact and is not committed: it is a zip, which the archive rule does
not accept. The `.usda` is the committed form, which is what that rule asks for.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def write_npz(out):
    import torch

    import anny_rig

    model = anny_rig.build_corpus_model(dtype=torch.float64)
    with torch.no_grad():
        rest = model(pose_parameters=anny_rig._identity_pose(model))
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             verts=rest["rest_vertices"][0].numpy(),
             faces=np.asarray(model.faces),
             bone_poses=rest["rest_bone_poses"][0].numpy(),
             parents=np.asarray(model.bone_parents))
    names = out.with_suffix(".names.json")
    names.write_text(json.dumps(list(model.bone_labels)), encoding="utf-8")
    print("wrote %s and %s: %d joints" % (out, names, len(model.bone_labels)))
    return 0


def self_test():
    """Two controls, on the fixture as committed rather than on one built here."""
    import check_conventions

    r = []
    rig = check_conventions.read_rig("fixtures-conventions.usda")
    problems, m = check_conventions.check(rig)
    r.append(("the committed fixture is accepted by the gate", not problems))
    r.append(("it carries faces and bases, so no check reports NOT RUN",
              m["signed_volume"] is not None and m["negative_bases"] is not None))
    for name, ok in r:
        print("  %-4s control: %s" % ("ok" if ok else "FAIL", name))
    bad = sum(1 for _, ok in r if not ok)
    print("  %d of %d controls fired." % (len(r) - bad, len(r)))
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", help="write the intermediate npz here, then run mesh_to_usda.py")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    raise SystemExit(self_test() if args.self_test else write_npz(args.npz or "build/rig.npz"))
