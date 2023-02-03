# DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature

Implementation of the experiments in the [DetectGPT paper](https://arxiv.org/abs/2301.11305v1).

An interactive demo of DetectGPT can be found [here](https://detectgpt.ericmitchell.ai).

## Instructions

First, install the Python dependencies:

    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt

Second, run any of the scripts (or just individual commands) in `paper_scripts/`.

If you'd like to run the WritingPrompts experiments, you'll need to download the WritingPrompts data from [here](https://www.kaggle.com/datasets/ratthachat/writing-prompts). Save the data into a directory `data/writingPrompts`.

**Intermediate results are saved in `tmp_results/`. If your experiment completes successfully, the results will be moved into the `results` directory.

## Citing the paper
