# FairGap Anonymous Artifact

This repository contains the anonymized artifact for FairGap, a benchmark for
auditing hidden-output fairness gaps in LLM-based recommenders.

## Artifact scope

FairGap evaluates three recommendation domains and three protected-attribute
counterfactual settings:

- Domains: MovieLens, Goodreads, SteamReviews
- Attributes: gender, age, race

For review, this artifact provides:

1. Evaluation scripts for the FairGap pipeline.
2. A lightweight smoke-test instance for validating the metric pipeline, with
   folder structure for all nine domain-attribute settings.
3. Precomputed summary files for artifact inspection when provided.
4. Responsible AI notes and metadata for the included benchmark artifact.

The smoke-test files are intended to verify the metric pipeline without requiring
reviewers to rerun full LLM generation or hidden-state extraction. Full-scale
reproduction requires the original public recommendation datasets and the
corresponding open-weight LLM checkpoints.

## FairGap record

A FairGap record consists of:

- an anonymized user/profile identifier,
- a counterfactual prompt pair,
- paired recommendation lists,
- output-side shift scores,
- internal representation shift scores,
- Match@10 utility scores,
- and quadrant labels for hidden-output mismatch analysis.

Demographic cues are synthetic counterfactual prompt perturbations used for
auditing model behavior. They should not be interpreted as verified or inferred
protected attributes of real users.

## Directory structure

- `scripts/`: core FairGap evaluation scripts.
- `configs/`: example configuration files.
- `data/`: smoke-test benchmark records for each domain-attribute condition.
- `results/`: summary files for the reported benchmark tables and robustness checks.
- `metadata/`: Responsible AI notes and Croissant metadata.

## Reproducibility

The included smoke-test files allow reviewers to run the metric aggregation
steps, including output-shift scoring, internal-shift aggregation, quadrant
assignment, and main metric summarization. Full benchmark reproduction requires
external public datasets and model checkpoints.

## Anonymity

This artifact has been prepared for double-blind review. Author names,
institutional identifiers, local paths, and private credentials should not
appear in the repository.

## Internal probe re-fitting note

For lightweight review, we provide precomputed `probe_weights.json` and
`internal_distance_sample.eval.jsonl`. Re-fitting probe weights requires
`internal_vectors.npz`, which is omitted from the smoke-test package because it
contains large hidden-state arrays.

## Goodreads data source

The Goodreads profile builder expects locally downloaded Goodreads Book Graph
files. The source page is:
https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html

For lightweight review, precomputed Goodreads smoke-test profiles are provided
under `data/goodreads_smoke/`.

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
