# movielens_smoke / gender smoke-test files

This folder contains lightweight FairGap smoke-test files for the
`movielens_smoke` domain and `gender` counterfactual attribute setting.

Included files:

- profiles_sample.jsonl
- pairs_sample.jsonl
- ranked_lists_sample.jsonl
- split_sample.jsonl
- output_distance_sample.eval.jsonl
- internal_distance_layers_sample.eval.jsonl
- internal_distance_sample.eval.jsonl
- match10_sample.eval.jsonl
- fairgap_records_sample.jsonl

These files are intended for verifying the FairGap data schema and metric
pipeline, not for reproducing the full benchmark estimates. Raw generations,
hidden vectors, full user-level outputs, and large intermediate files are
intentionally excluded from the anonymous artifact.
