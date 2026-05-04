# FairGap Anonymous Review Release

This repository contains anonymized code for the FairGap benchmark pipeline.

## Contents
- Core pipeline scripts are in fairgap/.
- The pipeline supports counterfactual pair construction, LLM generation, ranked-list parsing, internal-vector extraction, OBS/IBS/ROA computation, Match@10 scoring, quadrant analysis, and case selection.

## Installation
pip install -r requirements.txt

## Notes
Full-scale reproduction requires GPU access and open-weight LLM checkpoints. Reviewers can inspect the scripts and apply them to toy or full intermediate files under examples/toy_out/.
