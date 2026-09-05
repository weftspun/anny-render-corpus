"""urn:oid:1.3.6.1.4.1.66606.1.2.2165.3 -- emit MaskScore Text-stub parquets.

Fills the last of MASKSCORE.md's eight stubs (mesh, depth, pose, keypoints,
multimodal, speech, video, text). Walking-skeleton shape per the Mesh stub:
one root row with a canonical text as INPUT, three candidates on a new
`text_edit` axis, WER against the canonical as the score, controls asserted
before write.

    identity     : the canonical text verbatim; WER must be 0
    paraphrase   : a mild edit; WER > 0 and strictly less than wrong_all
    wrong_all    : a garbled string; WER strictly worse than paraphrase

Five ZSTD parquets in ETNF form via the shared schema module:

    maskscore_text.parquet
    maskscore_text_candidates.parquet
    maskscore_text_scores.parquet
    maskscore_text_root_extras.parquet          canonical_text as text_utf8
    maskscore_text_candidate_extras.parquet     candidate_target_text per rank

Usage:
    pixi run --environment anny-mac python emit_text_stub.py --out-dir .
"""
from __future__ import annotations

import argparse
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq

import maskscore_stub_schema as ms


CANONICAL = "The quick brown fox jumps over the lazy dog."

CANDIDATES = [
    # (rank, kind, target_text)
    (1, "identity",    CANONICAL),
    (3, "paraphrase",  "A quick brown fox jumps above the lazy dog."),
    (5, "wrong_all",   "banana banana banana banana banana banana banana."),
]


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate: Levenshtein over tokens, normalised by reference length.

    Same shape as emit_speech_stub.py's wer(); duplicated here to keep the
    text stub independent of the speech stub's build.
    """
    r = reference.lower().split()
    h = hypothesis.lower().split()
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = prev[j - 1] if rw == hw else 1 + min(prev[j - 1], prev[j], cur[j - 1])
        prev = cur
    return prev[-1] / len(r)


def assert_controls(scores: list[tuple[int, str, float]]) -> None:
    by_kind = {kind: score for _, kind, score in scores}
    if by_kind["identity"] != 0.0:
        raise SystemExit(
            f"identity control failed: identity WER {by_kind['identity']} != 0.0"
        )
    if not (by_kind["paraphrase"] < by_kind["wrong_all"]):
        raise SystemExit(
            f"negative control failed: wrong_all WER ({by_kind['wrong_all']}) "
            f"not strictly worse than paraphrase ({by_kind['paraphrase']})"
        )


def build_tables(out_dir: pathlib.Path):
    key = "text/canonical/pangram/en"
    scores_by_rank = [
        (rank, kind, wer(CANONICAL, target))
        for rank, kind, target in CANDIDATES
    ]
    assert_controls(scores_by_rank)

    root = pa.table({
        "key":              [key],
        "task_type":        ["instruction_edit"],
        "dimension":        ["instruction_following"],
        "input_column":     ["input_text"],
        "input_asset":      ["(inline canonical; see maskscore_text_root_extras.parquet)"],
        "input_asset_kind": ["text_utf8"],
    }, schema=ms.ROOT)

    cands = pa.table({
        "row_key":              [key] * len(CANDIDATES),
        "candidate_axis":       ["text_edit"] * len(CANDIDATES),
        "rank":                 [rank for rank, _, _ in CANDIDATES],
        "candidate_asset":      ["(inline candidate; see maskscore_text_candidate_extras.parquet)"] * len(CANDIDATES),
        "candidate_asset_kind": ["text_utf8"] * len(CANDIDATES),
        "candidate_kind":       [kind for _, kind, _ in CANDIDATES],
    }, schema=ms.CANDIDATES)

    scores = pa.table({
        "row_key":        [key] * len(scores_by_rank),
        "candidate_axis": ["text_edit"] * len(scores_by_rank),
        "candidate_rank": [rank for rank, _, _ in scores_by_rank],
        "view_index":     [-1] * len(scores_by_rank),
        "metric_name":    ["wer"] * len(scores_by_rank),
        "metric_value":   [score for _, _, score in scores_by_rank],
    }, schema=ms.SCORES)

    root_extras = pa.table({
        "row_key":     [key],
        "extra_name":  ["canonical_text"],
        "extra_value": [CANONICAL],
        "extra_kind":  ["text_utf8"],
    }, schema=ms.ROOT_EXTRAS)

    cand_extras = pa.table({
        "row_key":        [key] * len(CANDIDATES),
        "candidate_axis": ["text_edit"] * len(CANDIDATES),
        "candidate_rank": [rank for rank, _, _ in CANDIDATES],
        "extra_name":     ["candidate_target_text"] * len(CANDIDATES),
        "extra_value":    [target for _, _, target in CANDIDATES],
        "extra_kind":     ["text_utf8"] * len(CANDIDATES),
    }, schema=ms.CANDIDATE_EXTRAS)

    ms.validate(root_t=root, cands_t=cands, scores_t=scores,
                root_extras_t=root_extras, cand_extras_t=cand_extras)

    p_root, p_cands, p_scores, p_root_extras, p_cand_extras = ms.paths_for("text", out_dir)
    for tbl, path in ((root, p_root), (cands, p_cands), (scores, p_scores),
                      (root_extras, p_root_extras), (cand_extras, p_cand_extras)):
        pq.write_table(tbl, path, compression="zstd")

    return (p_root, p_cands, p_scores, p_root_extras, p_cand_extras), scores_by_rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=".", type=pathlib.Path)
    args = ap.parse_args()
    paths, scored = build_tables(args.out_dir)
    for p in paths:
        print(f"wrote {p}")
    for rank, kind, score in scored:
        print(f"  rank {rank} {kind}: wer = {score}")


if __name__ == "__main__":
    main()
