"""Views come from the sequence. Choosing one because it looked better is not a view choice.

WHY THIS EXISTS. `render_view.py` draws its cameras from `sphere_hammersley_sequence`: view i
of n is a yaw and a pitch nobody argued for. The T01 entry records the failure it replaced. A
front view picked by hand flattened the thing being judged -- mean foot separation is 0.356 m
along the travel axis, about five stacked soda cans, against 0.230 m across it, about three
and a half.

The sequence does not stop somebody hand-picking a MEMBER of it, and that is the same failure
one level down. It happened again while testing depth conditioning: view 0 came out looking
straight down, so index 3 was used instead because it looked more like a person. That is
selection on appearance, dressed in the sequence's clothes, and the T01 entry already names
the honest alternative -- "narrowing the pitch band would be a decision to write down rather
than a default to inherit".

WHAT IS CHECKED. A caller may render the whole sequence, or a contiguous prefix of it, or a
band declared in `PITCH_BANDS` with a reason. What it may not do is name indices. A set of
indices that is neither the whole sequence nor a declared band is a hand-picked set however
it was arrived at.

THE SEQUENCE IS NOT UNIFORM IN PITCH, WHICH IS WHY THIS MATTERS. Measured at n=8:

    index   0     1     2     3     4     5     6     7
    pitch -90   -30     0   9.6  19.5    30  41.8  56.4

so a "representative" index chosen by eye is a pitch choice, and dropping index 0 alone
removes the only view below -30 degrees. Any subset silently changes the pitch distribution
the corpus is trained on.
"""

from __future__ import annotations

import sys

# A band is a decision with a reason attached, which is what makes it different from a
# hand-picked set. Empty on purpose: nobody has needed one yet, and adding an entry is the
# act of writing the decision down.
PITCH_BANDS: dict[str, tuple[float, float, str]] = {}


def check(indices, n_views, band=None):
    """Return a list of problems. Empty means the selection is defensible."""
    problems = []
    idx = sorted(set(indices))
    whole = idx == list(range(n_views))
    prefix = idx == list(range(len(idx)))

    if whole:
        return problems
    if band is not None:
        if band not in PITCH_BANDS:
            problems.append(
                "band %r is not declared in PITCH_BANDS, so it carries no reason. A band "
                "without a written reason is a hand-picked set with a name." % band
            )
        return problems
    if prefix:
        # A prefix is defensible -- it is "the first k of the sequence" -- but it still
        # truncates the pitch distribution, so it is reported rather than waved through.
        problems.append(
            "indices %s are a prefix of %d, which is defensible but truncates the pitch "
            "range: the sequence puts its extreme pitches late, so a prefix is a narrower "
            "band than it looks." % (idx, n_views)
        )
        return problems
    problems.append(
        "indices %s are neither the whole sequence of %d nor a prefix nor a declared band. "
        "Selecting members of a Hammersley sequence by eye is the hand-picked camera the "
        "sequence was adopted to replace." % (idx, n_views)
    )
    return problems


def self_test():
    """Each control must FAIL. A gate that cannot fail certifies the defect."""
    cases = [
        ("the whole sequence", list(range(8)), 8, None, False),
        ("a prefix", [0, 1, 2], 8, None, True),
        ("the set that was actually hand-picked", [3], 8, None, True),
        ("scattered indices", [1, 4, 7], 8, None, True),
        ("an undeclared band", [2, 3], 8, "waist-height", True),
    ]
    bad = 0
    print("view-selection gate")
    for label, idx, n, band, want_fail in cases:
        got = check(idx, n, band)
        ok = bool(got) == want_fail
        print("  %-38s %s" % (label, "ok" if ok else "MISS"))
        if got:
            print("      %s" % got[0][:96])
        if not ok:
            bad += 1
    print("\n%d case(s) wrong" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(self_test())
