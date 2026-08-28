"""Does what was published actually work for someone who only has the hub?

AN UPLOAD THAT REPORTED SUCCESS IS NOT A REPOSITORY THAT SERVES. Those are different claims,
and only the second matters to whoever arrives next. This checks the second, from the hub's
side, downloading what it needs rather than reading the local copies it was built from.

The strongest check here is the last one: every path in the training records is resolved
against the file list the hub actually serves. A record pointing at a file that is not in the
repository is the exact defect that absolute paths were -- a set that looks complete and is
inert for everybody but us -- and it survives a card review, because cards do not resolve
paths.

    python check_published.py --namespace chibifire
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

REVISION = "df5dca8a981d74e6c3af214c145f5c735fe72367"


def hf_token(item="rkuylld4umpmaxvlvbp5q7kgii"):
    out = subprocess.run(["op", "item", "get", item, "--fields", "label=credential", "--reveal"],
                         capture_output=True, text=True, timeout=90)
    token = out.stdout.strip()
    if not token.startswith("hf_"):
        sys.exit("FAIL  no usable token from 1Password: %s" % out.stderr.strip()[:160])
    return token


def resolve_records(rows, served):
    """Every image a record points at, against the file list the hub actually serves.

    Split out from `main` so a control can drive it: the network path cannot be a control,
    and this is the check that matters -- a record naming a local absolute path is inert for
    everybody but us, and survives a card review because cards do not resolve paths.
    """
    referenced, missing, absolute = set(), [], []
    for r in rows:
        for q in list(r.get("input_images", [])) + [r["output_image"]]:
            referenced.add(q)
            if ":" in q or q.startswith("/"):
                absolute.append(q)
            elif q not in served:
                missing.append(q)
    return referenced, missing, absolute


def self_test():
    """Six controls. Five must reject records a hub could not serve."""
    served = {"images/a.png", "images/b.png"}
    ok_rows = [{"input_images": ["images/a.png"], "output_image": "images/b.png"}]
    r = []

    ref, missing, absolute = resolve_records(ok_rows, served)
    r.append(("records the repo serves are accepted",
              not missing and not absolute and ref == served))
    r.append(("a record naming a file the repo does not serve is caught",
              resolve_records([{"input_images": ["images/gone.png"],
                                "output_image": "images/b.png"}], served)[1]
              == ["images/gone.png"]))
    r.append(("a windows absolute path is caught, not counted as served",
              resolve_records([{"input_images": ["O:/local/a.png"],
                                "output_image": "images/b.png"}], served)[2]
              == ["O:/local/a.png"]))
    r.append(("a posix absolute path is caught",
              resolve_records([{"input_images": ["/home/me/a.png"],
                                "output_image": "images/b.png"}], served)[2]
              == ["/home/me/a.png"]))
    r.append(("the output image is checked, not only the inputs",
              resolve_records([{"input_images": ["images/a.png"],
                                "output_image": "images/gone.png"}], served)[1]
              == ["images/gone.png"]))
    r.append(("a record with no inputs still checks its output",
              resolve_records([{"output_image": "images/gone.png"}], served)[1]
              == ["images/gone.png"]))

    for name, ok in r:
        print("  %-4s control: %s" % ("ok" if ok else "FAIL", name))
    bad = sum(1 for _, ok in r if not ok)
    print("  %d of %d controls fired." % (len(r) - bad, len(r)))
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--namespace", default="chibifire")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    token = hf_token()
    api = HfApi(token=token)
    ns = args.namespace
    problems = []

    repos = [
        ("%s/anny-render-corpus" % ns, "dataset"),
        ("%s/anny-render-corpus-generated" % ns, "dataset"),
        ("%s/anny-camera-lora" % ns, "model"),
        ("%s/omnigen2-base-df5dca8a" % ns, "model"),
    ]

    listings = {}
    print("repositories")
    for repo_id, kind in repos:
        try:
            info = api.repo_info(repo_id, repo_type=kind, files_metadata=True)
        except Exception as error:  # noqa: BLE001
            problems.append("%s does not resolve: %s" % (repo_id, str(error)[:90]))
            print("  BAD  %-44s %s" % (repo_id, str(error)[:60]))
            continue
        names = [s.rfilename for s in info.siblings]
        listings[repo_id] = names
        gib = sum(s.size or 0 for s in info.siblings) / 2 ** 30
        print("  ok   %-44s %-8s %3d files  %6.2f GiB  private=%s"
              % (repo_id, kind, len(names), gib, info.private))
        if info.private:
            problems.append("%s is private; it was published to be readable" % repo_id)
        for required in ("README.md", "CITATION.cff"):
            if required not in names:
                problems.append("%s has no %s" % (repo_id, required))

    # THE CARD MUST NAME THE CODE. RFD 1141's whole point is that an artifact reaches back.
    print("\ncards reach the code")
    for repo_id, kind in repos:
        if repo_id not in listings:
            continue
        try:
            path = hf_hub_download(repo_id, "README.md", repo_type=kind, token=token,
                                   force_download=True)
            text = open(path, encoding="utf-8").read()
        except Exception as error:  # noqa: BLE001
            problems.append("%s README unreadable: %s" % (repo_id, str(error)[:80]))
            continue
        # The mirror is upstream's work and names upstream instead; that is correct for it.
        want = "OmniGen2/OmniGen2" if repo_id.endswith("omnigen2-base-df5dca8a") \
            else "github.com/weftspun"
        mark = "ok  " if want in text else "BAD "
        if want not in text:
            problems.append("%s README does not name %s" % (repo_id, want))
        print("  %s %-44s names %s" % (mark, repo_id, want))
        for leak in ("ernes", "C:" + chr(92), "C:/"):
            if leak in text:
                problems.append("%s README contains %r" % (repo_id, leak))

    # THE ADAPTER MUST BE LOADABLE-SHAPED. `task_type: null` is what the hub complained about.
    print("\nadapter config")
    try:
        cfg = json.load(open(hf_hub_download("%s/anny-camera-lora" % ns, "adapter_config.json",
                                             token=token, force_download=True), encoding="utf-8"))
        for key, want in (("peft_type", "LORA"), ("r", 8), ("lora_alpha", 8)):
            got = cfg.get(key)
            print("  %s %-16s %s" % ("ok  " if got == want else "BAD ", key, got))
            if got != want:
                problems.append("adapter_config %s is %r, not %r" % (key, got, want))
        tt = cfg.get("task_type")
        ok = isinstance(tt, str) and tt
        print("  %s task_type       %r" % ("ok  " if ok else "BAD ", tt))
        if not ok:
            problems.append("adapter_config task_type is %r; the hub requires a string" % tt)
    except Exception as error:  # noqa: BLE001
        problems.append("adapter_config unreadable: %s" % str(error)[:90])

    # THE RECORDS MUST RESOLVE AGAINST WHAT THE HUB SERVES. This is the check that would have
    # caught the absolute paths, and it catches a rename or a missed file just as well.
    print("\ntraining records resolve against the published file list")
    ds = "%s/anny-render-corpus" % ns
    served = set(listings.get(ds, []))
    for name in ("records/train_formA.jsonl", "records/val_formA.jsonl"):
        try:
            path = hf_hub_download(ds, name, repo_type="dataset", token=token,
                                   force_download=True)
        except Exception as error:  # noqa: BLE001
            problems.append("%s missing: %s" % (name, str(error)[:80]))
            continue
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        referenced, missing, absolute = resolve_records(rows, served)
        print("  %s %-28s %3d rows, %3d distinct files, %d missing, %d absolute"
              % ("ok  " if not missing and not absolute else "BAD ",
                 name, len(rows), len(referenced), len(missing), len(absolute)))
        if missing:
            problems.append("%s references %d file(s) the repo does not serve, e.g. %s"
                            % (name, len(missing), missing[0]))
        if absolute:
            problems.append("%s still carries %d absolute path(s)" % (name, len(absolute)))

    # THE MIRROR MUST BE THE REVISION IT CLAIMS, byte for byte where the hub can tell us.
    print("\nmirror matches upstream at %s" % REVISION[:8])
    mirror = "%s/omnigen2-base-df5dca8a" % ns
    try:
        up = {s.rfilename: s.lfs.get("sha256") if s.lfs else None
              for s in api.repo_info("OmniGen2/OmniGen2", revision=REVISION,
                                     files_metadata=True).siblings}
        mine = {s.rfilename: s.lfs.get("sha256") if s.lfs else None
                for s in api.repo_info(mirror, files_metadata=True).siblings}
        shared = [k for k in up if k in mine and up[k]]
        differing = [k for k in shared if up[k] != mine[k]]
        absent = [k for k in up if k not in mine]
        print("  %s %d upstream file(s), %d compared by sha256, %d differing, %d absent here"
              % ("ok  " if not differing and not absent else "BAD ",
                 len(up), len(shared), len(differing), len(absent)))
        if differing:
            problems.append("mirror differs from upstream on %d file(s), e.g. %s"
                            % (len(differing), differing[0]))
        if absent:
            problems.append("mirror is missing %d upstream file(s), e.g. %s"
                            % (len(absent), absent[0]))

        # THE SMALL FILES ARE HASHED HERE RATHER THAN TAKEN ON TRUST. The hub exposes sha256
        # for LFS objects only, so the config and tokenizer files came back as "present with
        # the right name", which is not the same claim as "identical". They are a few hundred
        # kilobytes in total, so downloading both copies and hashing them costs almost
        # nothing, and a mirror whose config.json drifted would load a different model while
        # every weight matched.
        import hashlib

        small = sorted(k for k in up if k in mine and not up[k])
        checked, mismatched = 0, []
        for name in small:
            try:
                a = hf_hub_download("OmniGen2/OmniGen2", name, revision=REVISION, token=token,
                                    force_download=True)
                b = hf_hub_download(mirror, name, token=token, force_download=True)
            except Exception as error:  # noqa: BLE001
                mismatched.append((name, "unreadable: %s" % str(error)[:60]))
                continue
            ha = hashlib.sha256(open(a, "rb").read()).hexdigest()
            hb = hashlib.sha256(open(b, "rb").read()).hexdigest()
            checked += 1
            if ha != hb:
                mismatched.append((name, "%s vs %s" % (ha[:12], hb[:12])))
        print("  %s %d small file(s) hashed locally, %d differing"
              % ("ok  " if not mismatched else "BAD ", checked, len(mismatched)))
        for name, why in mismatched[:6]:
            print("       %-44s %s" % (name, why))
        if mismatched:
            problems.append("%d small file(s) differ from upstream, e.g. %s"
                            % (len(mismatched), mismatched[0][0]))
    except Exception as error:  # noqa: BLE001
        problems.append("could not compare the mirror: %s" % str(error)[:110])
        print("  BAD  %s" % str(error)[:90])

    print()
    for p in problems:
        print("  BAD  %s" % p)
    print("%d problem(s)" % len(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    sys.exit(main())
