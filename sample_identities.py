"""Populate the identity relations of the ANNY render corpus.

Writes, per anny_render_schema.py:
  phenotypes.parquet          interned phenotype names
  identities.parquet          identity_id, seed, source_subject, split
  identity_phenotype.parquet  long-form values (no NULLs, schema-stable)

Population source is PLUGGABLE (`--source`) because the choice is a live
decision, not a fixed dependency:

  anny  (default) -- anny.shape_distribution.SimpleShapeDistribution. Calibrated
        to WHO growth curves, spans infants to elders, and is explicitly built
        to avoid demographic bias. Same parameter space the renderer consumes,
        so there is no cross-model mapping to get wrong.

  Deliberately NOT used: AddBiomechanics .b3d. Twice disqualified -- its
  subjects are biomechanics-lab volunteers (young, healthy, able-bodied, from
  the Camargo/Carter/Falisse/Fregly/Hamner studies), so it is neither diverse
  nor equitable as an identity population; and nimblephysics, its only reader,
  ships no Windows wheels and no Windows CI, so reading it here would mean an
  unsupported from-source build. Adding a new source means adding a function
  below, not reshaping the schema.

Determinism: identity_id = blake2b(source, index) so re-running appends
idempotently instead of duplicating. Each identity also carries the seed that
regenerates it alone.

Split hygiene: assigned HERE, at identity level, so no downstream stage can
leak the same identity across train/val. The validator in anny_render_schema
re-checks this.
"""

import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq

import anny_render_schema as S


def sample_anny(n: int, seed: int):
    """ANNY's own population prior. Returns (labels, rows) where rows[i] is a
    dict phenotype_name -> float in [0,1]."""
    import torch
    import anny
    from anny.shape_distribution import SimpleShapeDistribution

    torch.manual_seed(seed)
    model = anny.Anny(local_changes="default", facial_actions="all")
    dist = SimpleShapeDistribution(model)
    _age_years, pheno = dist.sample(n)          # (chronological age, {name: tensor})
    labels = list(model.phenotype_labels)       # gender, age, muscle, weight, height, proportions
    rows = [{k: float(pheno[k][i]) for k in labels} for i in range(n)]
    return labels, rows


SOURCES = {"anny": sample_anny}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=23000,
                        help="identities; ~23k x ~35 views reaches the 800k budget")
    parser.add_argument("--source", default="anny", choices=sorted(SOURCES))
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    args = parser.parse_args()

    labels, rows = SOURCES[args.source](args.count, args.seed)
    os.makedirs(args.out, exist_ok=True)

    pheno_id = {name: i for i, name in enumerate(labels)}
    pq.write_table(
        pa.table({"phenotype_id": pa.array(list(pheno_id.values()), pa.int16()),
                  "name": pa.array(list(pheno_id.keys()))}, schema=S.PHENOTYPES),
        os.path.join(args.out, "phenotypes.parquet"), compression="zstd")

    ids, seeds, subjects, splits = [], [], [], []
    ip_identity, ip_pheno, ip_value = [], [], []
    for i, row in enumerate(rows):
        # blake2b over (source, index): stable across machines and re-runs.
        ident = S.deterministic_id(args.source, args.seed, i) % (2**31 - 1)
        ids.append(ident)
        seeds.append(S.deterministic_id("seed", args.source, args.seed, i))
        # Each sampled identity is an INDEPENDENT draw, so it is its own
        # subject. A shared literal here would make the validator's cross-split
        # contamination check fire on every row (one "subject" spanning both
        # splits) -- a false positive that would block a clean corpus.
        subjects.append(f"{args.source}:sampled:{ident}")
        # Deterministic split from the id itself -- never random per run, so a
        # resumed run assigns the same identity to the same side.
        splits.append("val" if (ident % 10000) < int(args.val_fraction * 10000) else "train")
        for name, value in row.items():
            ip_identity.append(ident)
            ip_pheno.append(pheno_id[name])
            ip_value.append(value)

    pq.write_table(
        pa.table({"identity_id": pa.array(ids, pa.int32()),
                  "seed": pa.array(seeds, pa.int64()),
                  "source_subject": pa.array(subjects),
                  "split": pa.array(splits)}, schema=S.IDENTITIES),
        os.path.join(args.out, "identities.parquet"), compression="zstd")

    pq.write_table(
        pa.table({"identity_id": pa.array(ip_identity, pa.int32()),
                  "phenotype_id": pa.array(ip_pheno, pa.int16()),
                  "value": pa.array(ip_value, pa.float32())}, schema=S.IDENTITY_PHENOTYPE),
        os.path.join(args.out, "identity_phenotype.parquet"), compression="zstd")

    n_val = splits.count("val")
    print(f"{len(ids)} identities ({len(ids)-n_val} train / {n_val} val), "
          f"{len(ip_identity)} phenotype rows -> {args.out}")
    print(f"unique ids: {len(set(ids))} (collisions: {len(ids)-len(set(ids))})")


if __name__ == "__main__":
    main()
