# Responsible AI Notes for the FairGap Smoke Artifact

## Intended use

This artifact is intended for research review and reproducibility checking of the FairGap benchmark pipeline. It supports inspection of how counterfactual recommendation prompts, observable output shift, internal representation shift, and hidden-output quadrant assignments are computed.

## Out-of-scope use

This artifact is not intended for production recommendation, user profiling, demographic inference, fairness certification, or automated decision-making.

## Data sources

The full FairGap pipeline uses public recommendation benchmark sources, including MovieLens, Goodreads Book Graph, and Steam recommendation data. The smoke package includes a lightweight anonymized subset for artifact review.

## Protected-attribute cues

Protected-attribute information is introduced synthetically through counterfactual prompt cues. These cues should not be interpreted as verified demographic labels of real users.

## Privacy and anonymization

Smoke-test user identifiers are anonymized. The artifact omits large hidden-state arrays and local run logs. Source dataset licenses and terms should be respected when reconstructing full datasets.

## Bias and limitations

Recommendation datasets and LLM outputs may reflect historical, cultural, popularity, and demographic biases. The smoke subset is not statistically representative and should only be used to validate the computational pipeline. Full fairness conclusions require the complete benchmark setup described in the paper.

## Model and compute considerations

Full generation and hidden-state extraction require open-weight LLM checkpoints, Hugging Face access for gated models when applicable, and GPU resources. The smoke artifact includes precomputed intermediate files to support lightweight review.
