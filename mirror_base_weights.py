"""Mirror the exact OmniGen2 revision this project trained against.

WHAT A MIRROR BUYS, AND WHAT IT DOES NOT. The canonical weights are on the hub already, and
our cards cite the commit, so this does not make them easier to reach. What it protects
against is upstream moving or deleting them: an adapter is 19.52 MiB of deltas against a base
it cannot function without, so a card that cites a revision nobody can fetch any more
describes a model that no longer exists.

APACHE-2.0 PERMITS THIS, AND THE ATTRIBUTION IS THE CONDITION. OmniGen2 is Apache-2.0 in both
weights and code. Redistribution is allowed with the licence and the notices intact, so the
card names the upstream repository, the exact revision, and the licence rather than presenting
the files as ours. Nothing here is modified: this is a copy, and a copy that had been altered
would be a different artifact needing a different name.

THE REVISION IS IN THE REPOSITORY NAME. `omnigen2-base-df5dca8a` says which commit it is
without opening it, and a second mirror of a different revision cannot quietly overwrite it.

SIZE, MEASURED: 29.11 GiB across 36 files. `upload_large_folder` is used rather than
`upload_folder` because it resumes, and an upload this size will be interrupted.

    python mirror_base_weights.py --namespace chibifire [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

REVISION = "df5dca8a981d74e6c3af214c145f5c735fe72367"
UPSTREAM = "OmniGen2/OmniGen2"
SOURCE_REPO = "https://github.com/weftspun/anny-render-corpus"


def hf_token(item="rkuylld4umpmaxvlvbp5q7kgii"):
    out = subprocess.run(["op", "item", "get", item, "--fields", "label=credential", "--reveal"],
                         capture_output=True, text=True, timeout=90)
    token = out.stdout.strip()
    if not token.startswith("hf_"):
        sys.exit("FAIL  1Password returned no usable token; unlock the desktop app. %s"
                 % out.stderr.strip()[:160])
    return token


def card(gib, files):
    return """---
license: apache-2.0
tags:
  - mirror
  - base-model
---

# omnigen2-base-df5dca8a

An unmodified mirror of [`%s`](https://huggingface.co/%s) at revision
[`%s`](https://huggingface.co/%s/tree/%s).

**%.2f GiB across %d files.** Nothing here is changed. If it were, it would need a different
name, because a modified copy is a different artifact.

## Why this exists

[`chibifire/anny-camera-lora`](https://huggingface.co/chibifire/anny-camera-lora) is 19.52 MiB
of deltas against these weights and cannot function without them. Its card cites this exact
commit, and a cited revision that can no longer be fetched describes a model that no longer
exists. This is the copy that keeps that from happening.

Use the upstream repository if you can. This is the fallback, not the front door.

## Licence and attribution

OmniGen2 is **Apache-2.0** in both weights and code, which permits redistribution with the
licence and notices intact. The work is by [VectorSpaceLab](https://huggingface.co/OmniGen2);
this repository claims no authorship of it and adds nothing to it.

## Related

- Adapter: [`chibifire/anny-camera-lora`](https://huggingface.co/chibifire/anny-camera-lora)
- Corpus: [`chibifire/anny-render-corpus`](https://huggingface.co/datasets/chibifire/anny-render-corpus)
- Generated outputs, held apart: [`chibifire/anny-render-corpus-generated`](https://huggingface.co/datasets/chibifire/anny-render-corpus-generated)
- Code: [`weftspun/anny-render-corpus`](%s), on the `6-datasource` side of the hexagon
""" % (UPSTREAM, UPSTREAM, REVISION[:8], UPSTREAM, REVISION, gib, files, SOURCE_REPO)


def citation():
    return """cff-version: 1.2.0
message: "Cite the original work, not this mirror."
type: software
title: "OmniGen2 (mirror at df5dca8a)"
authors:
  - name: "VectorSpaceLab"
license: Apache-2.0
url: "https://huggingface.co/%s"
abstract: >-
  An unmodified copy of %s at revision %s, kept so that a cited
  revision stays fetchable. This repository adds nothing and claims no authorship. Cite the
  upstream work.
""" % (UPSTREAM, UPSTREAM, REVISION)


def snapshot_dir():
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import IncompleteSnapshotError

    # OFFLINE FIRST, BECAUSE THE POINT IS TO MIRROR WHAT WAS TRAINED AGAINST. Fetching
    # whatever the hub serves now would produce a repository that claims a revision and holds
    # something else. The revision is pinned in both calls, so the fallback cannot drift.
    try:
        return snapshot_download(UPSTREAM, revision=REVISION, local_files_only=True)
    except IncompleteSnapshotError as error:
        # A MIRROR MISSING FILES IS NOT A MIRROR. The local cache holds every weight because
        # training needed them, and never fetched the six README images because nothing did.
        # Completing the snapshot at the same pinned commit is the difference between a copy
        # and a subset, and saying which happened matters more than either.
        print("  local cache is incomplete, completing it at the pinned revision:")
        print("    %s" % str(error).split("incomplete:")[-1].strip()[:160])
        return snapshot_download(UPSTREAM, revision=REVISION)


def measure(path):
    total, count = 0, 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.realpath(os.path.join(root, f)))
                count += 1
            except OSError:
                continue
    return total / 2 ** 30, count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--namespace", default="chibifire")
    ap.add_argument("--repo", default="omnigen2-base-df5dca8a")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    snap = snapshot_dir()
    gib, files = measure(snap)
    print("mirroring %s @ %s" % (UPSTREAM, REVISION[:8]))
    print("  local snapshot %s" % snap)
    print("  %.2f GiB across %d files" % (gib, files))

    repo_id = "%s/%s" % (args.namespace, args.repo)
    card_text, cff = card(gib, files), citation()
    if args.dry_run:
        print("\ndry run: would create %s and upload the snapshot" % repo_id)
        print(card_text.splitlines()[6])
        return 0

    from huggingface_hub import HfApi

    api = HfApi(token=hf_token())
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)

    # The card and the citation go up first, so a partial upload is still a repository that
    # says what it is and who wrote the weights, rather than an unlabelled pile of shards.
    for name, text in (("README.md", card_text), ("CITATION.cff", cff)):
        api.upload_file(path_or_fileobj=text.encode("utf-8"), path_in_repo=name,
                        repo_id=repo_id, repo_type="model",
                        commit_message="Say what this mirror is before the weights arrive")
    print("  card and citation uploaded")

    api.upload_large_folder(folder_path=snap, repo_id=repo_id, repo_type="model",
                            ignore_patterns=["*.lock", ".cache*"])

    info = api.repo_info(repo_id, repo_type="model", files_metadata=True)
    got = sum(s.size or 0 for s in info.siblings) / 2 ** 30
    print("\n%s: %d file(s), %.2f GiB on the hub, private=%s"
          % (repo_id, len(info.siblings), got, info.private))
    names = [s.rfilename for s in info.siblings]
    for required in ("README.md", "CITATION.cff"):
        print("  %s %s" % ("ok  " if required in names else "BAD ", required))
    # THE SIZE IS COMPARED, NOT ASSUMED. An upload that reported success and a repository that
    # serves 29 GiB are different claims, and only the second one is any use to a reader.
    if got < gib * 0.98:
        print("  BAD  the hub holds %.2f GiB against %.2f GiB locally: the upload is "
              "incomplete. upload_large_folder resumes, so run it again." % (got, gib))
        return 1
    print("  https://huggingface.co/%s" % repo_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
