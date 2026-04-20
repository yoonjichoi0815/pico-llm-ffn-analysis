# pico-llm-ffn-analysis
Neuron-level analysis of Transformer FFN layers as key–value memories in a small language model (pico-LLM).

This repository focuses on my contribution to a course project on small Transformer interpretability.
I analyzed FFN neurons in a pico-LLM model trained on TinyStories, inspired by the key–value memory view of Transformer FFNs.

## My Contribution
- Implemented / adapted FFN activation logging for neuron-level analysis
- Collected top-k activation-triggered contexts for selected neurons
- Examined layer-wise token specificity using a token purity metric
- Visualized differences across layers with boxplots
- Interpreted the results in relation to the key–value memory view of FFNs

## Main Finding
Early-layer FFN neurons showed high token purity, suggesting lexical selectivity,
while higher-layer neurons showed lower and more variable purity,
consistent with a shift toward more abstract semantic or discourse-level behavior.

## Repository Structure

- `src/`: core implementation for FFN analysis (logging and visualization)
- `data/`: sample activation logs used for analysis
- `results/figures/`: final visualizations
- `results/poster/`: project poster

## Attribution
This work was developed as part of a course project at NYU.
The original project was completed collaboratively with teammates, and this repository focuses specifically on my FFN analysis contribution.
Only the components relevant to my contribution are included here.

The full course project codebase is not included in this repository.
This repository is intended to document my analysis work and findings.