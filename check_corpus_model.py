"""Every consumer builds the corpus model through the shared builder. Enforced, not asked.

THE FLAW THIS CLOSES. `anny_rig.CORPUS_CONFIG` pins the model every corpus stage must
agree on -- rig, topology, phenotypes, local_changes, facial_actions, skinning_method -- and
its own comment says changing a field invalidates already-rendered shards and is a schema
change. `interface_audit.py` warns against going around it: "Audit the model the corpus
ACTUALLY SHIPS, not a bare anny.Anny."

Both of those are prose. A new script that calls `anny.Anny(...)` directly gets a DIFFERENT
model and no complaint from anything, and the difference is silent because the defaults are
plausible: `facial_actions` defaults to "none", so a bare construction reports zero facial
actions and looks like a model that has none. That happened -- `check_facial_actions.py`
did it and concluded ANNY ships no expression at all, which was wrong and got recorded in
the plan before it was caught.

The failure is not that somebody ignored a comment. It is that a shared configuration had
no mechanism, so every new entry point was a fresh chance to disagree with it.

WHAT IS CHECKED, AND BOTH HALVES ARE NEEDED.

  1. No file outside the allowlist constructs `anny.Anny(` directly. Going through
     `build_corpus_model()` is then the only way to get a model.
  2. `CORPUS_CONFIG` still pins every field it is supposed to. Otherwise the builder stays
     the single entry point while quietly ceasing to configure anything, which would pass
     check 1 while reintroducing the exact defect.

ENUMERATION, NOT SAMPLING. The population is every tracked `.py` in this repository, which
is fixed and small, so PITFALLS 5 says enumerate. Every file is read and the count is
reported; nothing is skipped for being unlikely.

Usage:  python check_corpus_model.py [--self-test]
"""

import os
import re
import sys

# `anny_rig` owns the configuration and must construct the model; that is its job. This
# file names the pattern in its own prose and builds a fixture containing it, so it would
# otherwise report itself -- which would be a gate that always fails, and PITFALLS 2 says
# that is worse than none because it trains people to ignore output.
ALLOWLIST = {"anny_rig.py", "check_corpus_model.py"}

# The fields CORPUS_CONFIG exists to pin. A field that drops out of it silently reverts to
# the constructor default, which is how `facial_actions` would go back to "none".
REQUIRED_FIELDS = ("rig", "topology", "phenotypes", "local_changes",
                   "facial_actions", "skinning_method")

BARE_CONSTRUCTION = re.compile(r"\banny\s*\.\s*Anny\s*\(")
# An exemption must state a reason after the colon. A bare marker does not qualify, because
# a marker with no reason is the comment this gate was written to replace.
EXEMPT = re.compile(r"#\s*corpus-model-exempt:\s*\S")


def scan(root):
    """Returns (problems, files_read). A file that cannot be read is a problem, never a
    skip -- an unreadable file is an unchecked file and reads exactly like a clean one."""
    problems, read = [], 0
    for name in sorted(os.listdir(root)):
        if not name.endswith(".py"):
            continue
        path = os.path.join(root, name)
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:                                     # noqa: BLE001
            problems.append(f"{name}: unreadable ({exc}), so it is unchecked")
            continue
        read += 1
        if name in ALLOWLIST:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue                    # a comment quoting the pattern is not a call
            if EXEMPT.search(line):
                # A DECLARED exception, not an inferred one. Some constructions are
                # deliberately NOT the corpus model -- asserting a fact about a different
                # topology, for instance. The gate cannot read intent, so intent is written
                # down on the line, where a reviewer sees it in the diff.
                continue
            if BARE_CONSTRUCTION.search(line):
                problems.append(
                    f"{name}:{i} constructs anny.Anny() directly. Use "
                    f"anny_rig.build_corpus_model(); a bare construction takes the "
                    f"constructor defaults and silently disagrees with CORPUS_CONFIG.")
    return problems, read


def check_config(root):
    """CORPUS_CONFIG must still pin every field. Read as text rather than imported, so this
    gate runs without torch, anny or a GPU -- a gate nobody can run is not a gate."""
    path = os.path.join(root, "anny_rig.py")
    if not os.path.exists(path):
        return [f"missing {path}: the pinned configuration cannot be checked"]
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    m = re.search(r"CORPUS_CONFIG\s*=\s*dict\((.*?)\)\s*\n", text, re.S)
    if not m:
        return ["anny_rig.py declares no CORPUS_CONFIG dict"]
    body = m.group(1)
    missing = [f for f in REQUIRED_FIELDS if not re.search(rf"\b{f}\s*=", body)]
    return [f"CORPUS_CONFIG no longer pins: {', '.join(missing)}. Those revert to "
            f"constructor defaults with nothing reporting it."] if missing else []


def main(root="."):
    problems, read = scan(root)
    problems += check_config(root)
    print(f"checked {read} python files, allowlist {sorted(ALLOWLIST)}")
    for p in problems:
        print(f"  FAIL  {p}")
    if not problems:
        print(f"  ok    every consumer goes through anny_rig.build_corpus_model(), and "
              f"CORPUS_CONFIG pins all {len(REQUIRED_FIELDS)} fields")
    return 1 if problems else 0


def self_test():
    """Each control must FAIL. A gate that cannot fail certifies the defect."""
    import shutil, tempfile
    src = os.path.dirname(os.path.abspath(__file__))
    controls = []

    def _bare_construction(d):
        """The actual defect: a new script builds its own model."""
        with open(os.path.join(d, "zz_new_script.py"), "w", encoding="utf-8") as fh:
            fh.write("import anny\nm = anny.Anny(facial_actions='all')\n")

    def _unpinned_field(d):
        """The builder stays the only entry point and stops configuring anything."""
        p = os.path.join(d, "anny_rig.py")
        t = open(p, encoding="utf-8").read().replace('facial_actions="all",', "")
        open(p, "w", encoding="utf-8").write(t)

    controls = [("a new script constructs anny.Anny() directly", _bare_construction,
                 "constructs anny.Anny() directly"),
                ("CORPUS_CONFIG stops pinning facial_actions", _unpinned_field,
                 "no longer pins")]

    print("negative controls (each must FAIL):")
    bad = 0
    for label, mutate, expect in controls:
        d = tempfile.mkdtemp(prefix="corpus-model-")
        try:
            shutil.rmtree(d)
            shutil.copytree(src, d, ignore=shutil.ignore_patterns(
                ".pixi", "__pycache__", "coco_*", "*.parquet"))
            mutate(d)
            probs, _ = scan(d)
            probs += check_config(d)
            hit = [p for p in probs if expect in p]
            print(f"  {'ok  ' if hit else 'MISS'} {label}: "
                  f"{hit[0][:88] if hit else 'check did not fire'}")
            if not hit:
                bad += 1
        finally:
            shutil.rmtree(d, ignore_errors=True)
    print(f"\n{'All' if not bad else bad} control(s) {'fired.' if not bad else 'did not fire.'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(self_test() if "--self-test" in sys.argv else main(
        os.path.dirname(os.path.abspath(__file__))))
