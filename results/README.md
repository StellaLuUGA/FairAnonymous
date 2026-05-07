# Selected summary results

This directory contains selected summary-level result files for inspecting the
FairGap benchmark outputs and robustness analyses. It is not intended to contain
all domain, attribute, model, or raw-output results from the complete benchmark.

The included MovieLens summaries are representative examples of the result-file
schema used across benchmark conditions. Lightweight smoke-test files for
pipeline verification are provided under `data/`. Full benchmark reproduction
requires the released scripts, configuration files, public source datasets, and
open-weight model checkpoints.

Summary files may include model suffixes such as `_llama8` to indicate the
evaluated model condition and to avoid overwriting summaries from different
model families.
