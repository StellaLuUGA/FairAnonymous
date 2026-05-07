# FairGap Anonymous Artifact

This repository contains the anonymized artifact for FairGap, a benchmark for
auditing hidden-output fairness gaps in LLM-based recommenders.

## Cross-domain quadrant annotation files

For quick inspection, the repository root contains two consolidated files:

- `fairgap_allrecords_with_quadrants.jsonl`
- `fairgap_allrecords_with_quadrants.csv`

The same two files are also available under `data/`.

These files summarize smoke-sample records across Goodreads, MovieLens, and
SteamReviews. Each row includes `user_id`, `dataset`, `attribute`, `obs`, `ibs`,
and a derived diagnostic `quadrant` label (Q1–Q4). The quadrant labels are
computed from observable output shift (OBS) and internal representation shift
(IBS); they are not human-annotated ground-truth fairness labels.

## Artifact scope

FairGap evaluates three recommendation domains and three protected-attribute
counterfactual settings:

- Domains: MovieLens, Goodreads, SteamReviews
- Attributes: gender, age, race

For review, this artifact provides:

- Evaluation scripts for the FairGap pipeline.
- Lightweight smoke-test records for all nine domain-attribute settings.
- Selected MovieLens summary result files for artifact inspection.
- Configuration examples for representative smoke-test runs.
- Responsible AI notes and Croissant metadata for the included benchmark artifact.

The smoke-test files are intended to verify the file schema and metric pipeline
without requiring reviewers to rerun full LLM generation or hidden-state
extraction. Full-scale reproduction requires the original public recommendation
datasets and the corresponding open-weight LLM checkpoints.

## FairGap records

The lightweight smoke-test folders include records for:

- anonymized user/profile identifiers,
- counterfactual prompt pairs,
- parsed paired recommendation lists,
- observable output-shift scores,
- internal representation-shift scores,
- Match@10 utility scores, and
- compact joined FairGap records combining Match@10, output shift, and internal shift.

Summary-level hidden-output quadrant diagnostics are provided in the selected
result summaries under `results/`.

Demographic cues are synthetic counterfactual prompt perturbations used for
auditing model behavior. They should not be interpreted as verified or inferred
protected attributes of real users.

## Directory structure

- `scripts/`: core FairGap evaluation scripts.
- `configs/`: example configuration files.
- `data/`: lightweight smoke-test records for each domain-attribute condition.
- `results/`: selected MovieLens summary result files.
- `metadata/`: Responsible AI notes and Croissant metadata.

## Reproducibility

The included smoke-test files allow reviewers to inspect the data schema and run
metric aggregation steps such as output-shift scoring, internal-shift
aggregation, Match@10 scoring, quadrant summarization, and main metric
summarization.

Full benchmark reproduction requires external public datasets, open-weight model
checkpoints, and GPU resources. The smoke artifact includes precomputed
intermediate files so reviewers can verify the downstream metric pipeline
without rerunning full generation or hidden-state extraction.

## Anonymity

This artifact has been prepared for double-blind review. Author names,
institutional identifiers, local paths, and private credentials should not
appear in the repository.

## Internal representation note

For lightweight review, the artifact provides precomputed internal-distance
files such as `internal_distance_sample.eval.jsonl` and
`internal_distance_layers_sample.eval.jsonl`. Full hidden-state arrays are
omitted because they are large and expensive to regenerate.

## Goodreads data source

The Goodreads profile builder expects locally downloaded Goodreads Book Graph
files. For lightweight review, precomputed Goodreads smoke-test profiles are
provided under `data/goodreads_smoke/`.

## GPU-dependent generation and hidden-state extraction

Steps 3 and 4 require open-weight LLM checkpoints, Hugging Face access when
models are gated, and GPU resources. The lightweight smoke-test package provides
precomputed ranked lists and internal-distance files so reviewers can verify the
metric pipeline without rerunning full generation or hidden-state extraction.

## Prompt formatting in Step 3

Step 3 supports both raw-prompt generation and chat-template generation.

For the simple prompt family, use raw prompting by passing `--no_chat_template`.
This matches the flat, instruction-minimal prompt described in the paper.

For the structured and optimized prompt families, the script may use the
tokenizer's official `apply_chat_template` interface when available. In that
case, pass the corresponding system message through `--system_prompt`. If a
model does not provide a reliable chat template, use `--no_chat_template` and
provide a prompt where the system and user blocks are concatenated explicitly.
