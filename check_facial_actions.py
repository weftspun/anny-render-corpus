"""Is ANNY's facial-action set ARKit's 52? Compared by NAME, both directions.

WHY THIS IS NOT THE CHECK THAT ALREADY EXISTS. `interface_audit.py` names the interface
"ANNY facial_actions <-> ARKit-52" and then tests `len(fa) == 52`. A count is the
convenient proxy; two sets of 52 can be 52 different things, and the quantity that decides
it is which names are in both. `coco.pth`'s weight map got `searchsorted` and a maximum
positional difference of zero exactly. This interface got cardinality.

WHY IT MATTERS RATHER THAN BEING PEDANTRY. ARKit is FACS-DERIVED and is not FACS: it comes
from FaceShift's shapes, and the translation is many-to-many in four ways at once. Two
ARKit shapes collapse to one action unit (mouthSmileLeft and mouthSmileRight are both AU12);
one ARKit shape spans two (jawOpen is AU26 or AU27); one action unit splits into two ARKit
shapes (AU17 into mouthShrugUpper and mouthShrugLower); and several map to AD and M codes
rather than to action units at all.

So a rig whose labels are FACS action units and a rig whose labels are ARKit blendshapes are
not interchangeable however their counts compare, and getting this wrong is documented rather
than hypothetical: ICT-FaceKit ships mouthShrugUpper labelled "upper lip raiser" and the
actual upper-lip-raiser shapes labelled "nasolabial furrow deepener".

Source for the 52: Apple's ARFaceAnchor.BlendShapeLocation. Cross-checked against Ozel's
ARKit-to-FACS cheat sheet, which tabulates 51 of them and omits tongueOut -- the same 51/52
split DETAILS.md already records for MediaPipe, from a different direction.
"""

import sys

# Apple's ARFaceAnchor.BlendShapeLocation, all 52.
ARKIT_52 = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
    "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight", "tongueOut",
]


def norm(s):
    """Compare on lowercase alphanumerics only, so browDownLeft, brow_down_left and
    BrowDown_L are one name. A cosmetic difference is not a semantic one, and reporting it
    as one would bury the real finding."""
    return "".join(c for c in str(s).lower() if c.isalnum())


def compare(anny_names, arkit=ARKIT_52):
    """The two sets by normalised name, both directions. Split out so a control can put
    known-broken input through the same comparison the real run uses."""
    a = {norm(x): x for x in arkit}
    n = {norm(x): x for x in anny_names}
    return sorted(set(a) - set(n)), sorted(set(n) - set(a)), a, n


def self_test():
    """Six controls. Five must reject a set that is not ARKit's 52 by name."""
    r = []
    good = list(ARKIT_52)
    r.append(("the identical set is accepted", compare(good)[:2] == ([], [])))
    r.append(("cosmetic spelling is not a difference",
              compare([x.lower().replace("Left", "_left") for x in good])[:2] == ([], [])))

    renamed = good[:-1] + ["cheekPuffed"]
    missing_arkit, extra_anny, _, _ = compare(renamed)
    r.append(("a renamed action is caught in both directions",
              len(missing_arkit) == 1 and len(extra_anny) == 1))
    r.append(("a dropped action is reported as absent from ANNY",
              compare(good[:-1])[0] == [norm(good[-1])]))
    r.append(("an added action is reported as absent from ARKit",
              compare(good + ["tongueOut2"])[1] == ["tongueout2"]))

    # THE CARDINALITY TRAP THIS FILE EXISTS FOR: 52 names that are not the 52 names.
    fifty_two_others = good[:51] + ["notAnArkitShape"]
    r.append(("52 of the wrong names is still rejected",
              len(fifty_two_others) == 52 and compare(fifty_two_others)[:2] != ([], [])))

    for name, ok in r:
        print("  %-4s control: %s" % ("ok" if ok else "FAIL", name))
    bad = sum(1 for _, ok in r if not ok)
    print("  %d of %d controls fired." % (len(r) - bad, len(r)))
    return 1 if bad else 0


def main():
    assert len(ARKIT_52) == 52, f"the ARKit list itself is {len(ARKIT_52)}, not 52"
    assert len(set(map(norm, ARKIT_52))) == 52, "the ARKit list has a duplicate"

    try:
        import anny
    except ImportError as exc:
        sys.exit(f"FAIL  cannot import anny ({exc}). Run under `pixi run -e anny`.")

    print(f"anny {getattr(anny, '__version__', 'version unknown')}")

    # THE SHARED BUILDER, NOT A BARE `anny.Anny()`. `anny_rig.CORPUS_CONFIG` is the pinned
    # configuration every corpus stage agrees on -- its own comment says changing any field
    # invalidates already-rendered shards and is a schema change. `facial_actions="all"` is
    # one of those fields.
    #
    # An earlier version of this script called `anny.Anny()` directly, got zero facial
    # actions because the constructor defaults them to "none", and reported that absence as
    # a property of the MODEL rather than of the CALL. `interface_audit.py` warns against
    # exactly this on the line above its own build: "Audit the model the corpus ACTUALLY
    # SHIPS, not a bare anny.Anny."
    #
    # So the guard existed, in this repository, one file away, and a new entry point walked
    # around it. Going through the builder is what makes that impossible rather than
    # discouraged.
    import anny_rig
    model = anny_rig.build_corpus_model()
    labels = getattr(model, "facial_action_labels", None)
    print("  ..   built via anny_rig.build_corpus_model() (CORPUS_CONFIG)")

    if labels is None:
        # An unmet precondition is a FAIL, never a skip. Reporting "no mismatch found"
        # because the attribute could not be read is the silent-skip failure exactly.
        sys.exit("FAIL  could not read facial_action_labels. The interface stays UNCHECKED; "
                 "this run measured nothing and must not be read as agreement.")

    labels = [str(x) for x in labels]
    print(f"  ..   ANNY reports {len(labels)} facial actions")

    a, n = {norm(x): x for x in ARKIT_52}, {norm(x): x for x in labels}
    shared = sorted(set(a) & set(n))
    only_arkit = sorted(set(a) - set(n))
    only_anny = sorted(set(n) - set(a))

    print(f"\n  shared names        {len(shared)} of 52")
    print(f"  ARKit-only          {len(only_arkit)}")
    print(f"  ANNY-only           {len(only_anny)}")
    if only_arkit:
        print("\n  in ARKit, absent from ANNY:")
        for k in only_arkit:
            print(f"    {a[k]}")
    if only_anny:
        print("\n  in ANNY, absent from ARKit:")
        for k in only_anny:
            print(f"    {n[k]}")

    if not only_arkit and not only_anny:
        print("\n  ok    the two sets are the same 52 names. 'Same definition' holds at "
              "name level.\n        Semantics still unverified: a shared name is not a "
              "shared meaning, which is\n        what ICT-FaceKit got wrong.")
        return 0
    print(f"\n  FAIL  the sets differ. ANNY's actions are not ARKit's 52 by name, so the "
          f"runtime\n        path needs a measured map rather than an assumed identity.")
    return 1


if __name__ == "__main__":
    sys.exit(self_test() if "--self-test" in sys.argv else main())
