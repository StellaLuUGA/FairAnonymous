# FairGap smoke-test data

This directory contains lightweight anonymized smoke-test data for the FairGap
artifact. The files are intended for artifact review, schema inspection, and
pipeline verification, not for estimating full benchmark results.

## Domains and attributes

The smoke-test package includes three recommendation domains:

- `movielens_smoke/`
- `goodreads_smoke/`
- `steam_smoke/`

Each domain contains three counterfactual protected-attribute settings:

- `age/`
- `gender/`
- `race/`

Each domain-attribute folder follows the same lightweight file schema:

- `profiles_sample.jsonl`
- `pairs_sample.jsonl`
- `ranked_lists_sample.jsonl`
- `split_sample.jsonl`
- `output_distance_sample.eval.jsonl`
- `internal_distance_layers_sample.eval.jsonl`
- `internal_distance_sample.eval.jsonl`
- `match10_sample.eval.jsonl`
- `fairgap_records_sample.jsonl`
- `README.md`

## Use

These files support lightweight inspection of the FairGap pipeline, including
counterfactual prompt construction, parsed recommendation lists, observable
output shift, internal representation shift, Match@10 utility, and compact
joined FairGap records.

Full benchmark reproduction requires the original public recommendation datasets,
open-weight model checkpoints, and GPU resources. Large hidden-state arrays,
full raw generations, and local run logs are intentionally excluded from this
anonymous artifact.
