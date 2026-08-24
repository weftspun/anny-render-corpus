"""Red tests for T06: topology_id is a foreign key, and a keypoint can be anchored two ways.

WHAT T06 ADDED AND WHY A TEST IS OWED. `KEYPOINTS_2D` used to key on `bone_id`, which asserts
that every keypoint follows a bone. The nose and both ears do not -- they have no bone and are
fixed mesh vertices -- so three of the 104 points the corpus exists to label were not merely
unvalidated, they were INEXPRESSIBLE. Nothing errored, because a schema that cannot say a thing
does not complain about not saying it.

So keypoints gained an identity, the anchoring moved into two satellite relations rather than
nullable columns, and `topology_id` became a foreign key into `TOPOLOGIES`.

THE POINT OF THIS FILE IS THAT `validate()` ACTUALLY REJECTS THE BROKEN CASES. It iterates
RELATIONS and FOREIGN_KEYS, so the new relations are covered the moment they are registered --
which is exactly the kind of coverage that looks free and can be wrong. A check that passes on
known-broken input certifies the defect.

WHY A DANGLING TOPOLOGY_ID IS THE ONE THAT MATTERS. ANNY's two topologies share zero vertices,
19,158 basemesh against 13,718 body. A vertex_id carrying the wrong topology is not ambiguous,
it is wrong -- and wrong at an index, silently, which is why the schema spends a byte on it.

Run:  pixi run -e corpus python test_topology_fk.py
"""

import os
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

import anny_render_schema as S


def write(root, name, rows):
    schema = S.RELATIONS[name]
    cols = {f.name: [r[f.name] for r in rows] for f in schema}
    pq.write_table(pa.table(cols, schema=schema),
                   os.path.join(root, f"{name}.parquet"), compression="zstd")


def clean(root):
    """The smallest corpus that exercises both anchor kinds, and nothing else."""
    write(root, "topologies", [
        {"topology_id": 0, "name": "makehuman", "vertex_count": 19158},
        {"topology_id": 1, "name": "anny", "vertex_count": 13718},
    ])
    write(root, "bones", [{"bone_id": 0, "name": "upperarm01.L", "parent_bone_id": -1}])
    write(root, "keypoint_defs", [
        {"keypoint_id": 0, "name": "left_shoulder"},   # bone-anchored
        {"keypoint_id": 1, "name": "nose"},            # vertex-anchored, HAS NO BONE
    ])
    write(root, "keypoint_bone_anchor", [{"keypoint_id": 0, "bone_id": 0}])
    write(root, "keypoint_vertex_anchor",
          [{"keypoint_id": 1, "topology_id": 0, "vertex_id": 4021}])
    return root


def c_dangling_topology(root):
    """A vertex anchor naming a topology that is not declared."""
    write(root, "keypoint_vertex_anchor",
          [{"keypoint_id": 1, "topology_id": 7, "vertex_id": 4021}])


def c_dangling_keypoint(root):
    """An anchor for a keypoint nothing defines."""
    write(root, "keypoint_bone_anchor", [{"keypoint_id": 99, "bone_id": 0}])


def c_null_topology(root):
    """A NULL where the foreign key goes. ETNF forbids it and the validator must say so."""
    schema = S.RELATIONS["keypoint_vertex_anchor"]
    pq.write_table(
        pa.table({"keypoint_id": [1], "topology_id": [None], "vertex_id": [4021]},
                 schema=schema),
        os.path.join(root, "keypoint_vertex_anchor.parquet"), compression="zstd")


def c_dangling_mesh_topology(root):
    """The same key one level up: a mesh whose topology is not declared."""
    write(root, "renders", [{"render_id": 1, "camera_id": 1, "run_id": 1,
                             "width": 8, "height": 8}])
    write(root, "meshes", [{"render_id": 1, "topology_id": 9, "geometry": b"x"}])


CONTROLS = [
    ("a vertex anchor names an undeclared topology", c_dangling_topology, "topology_id"),
    ("an anchor names an undefined keypoint", c_dangling_keypoint, "keypoint_id"),
    ("a NULL sits where the foreign key goes", c_null_topology, "NULL"),
    ("a mesh names an undeclared topology", c_dangling_mesh_topology, "topology_id"),
]


def relevant(problems):
    """`validate()` reports every absent relation; this fixture declares only a few on purpose,
    so those are filtered. Filtering the SIGNAL would be the bug, so the substring each control
    expects is asserted separately rather than inferred from the count."""
    return [p for p in problems if not p.startswith("missing relation:")]


def main():
    failures = 0

    # GREEN FIRST. A red suite that passes against a fixture the validator rejects for some
    # unrelated reason proves nothing about the checks.
    root = tempfile.mkdtemp(prefix="topo-fk-clean-")
    clean(root)
    green = relevant(S.validate(root))
    print(f"GREEN  clean corpus -> {len(green)} problems")
    for p in green:
        print(f"         {p}")
    if green:
        failures += 1

    for label, corrupt, expect in CONTROLS:
        r = tempfile.mkdtemp(prefix="topo-fk-red-")
        clean(r)
        corrupt(r)
        got = relevant(S.validate(r))
        hit = any(expect in p for p in got)
        print(f"RED  {'ok  ' if hit else 'MISS'}  {label}")
        if hit:
            print(f"         {next(p for p in got if expect in p)}")
        else:
            print(f"         expected a problem mentioning {expect!r}, got {got or 'nothing'}")
            failures += 1

    print(f"\n{len(CONTROLS)} controls, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
