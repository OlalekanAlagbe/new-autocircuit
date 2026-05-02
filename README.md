# Analogical Reasoning Circuit in Gemma-2-2B

**Mechanistic Interpretability · March 2026**

Olalekan Alagbe · Joseph Lawrence · Anish Maheshwar · Konstantinos Krampis

---

## Overview

This repository contains the code, data, and paper for our mechanistic interpretability analysis of analogical reasoning in Gemma-2-2B using Neuronpedia SAE attribution graphs.

We identify a shared **180-feature analogical reasoning circuit** active across five diverse analogy prompts, discover dedicated analogy-concept features at layers 5, 8, 9, and 13, and provide causal validation through feature steering experiments.

**Read the paper:** [olalekanalagbe.github.io/autocircuit](https://olalekanalagbe.github.io/autocircuit)

---

## Key Findings

- **Shared circuit:** 180 SAE features appear in all 5 attribution graphs; 510 in at least 3
- **Dedicated analogy features:** L5 #5793 ("analogies"), L8 #13766 ("analogies or comparisons"), L9 #13344, L13 #10969 — all domain-agnostic
- **Three-phase architecture:** Circuit templates (L0–4) → Analogy recognition (L5–9) → Relational integration (L10–13)
- **Causal validation:** Suppressing all Phase 2 features causes capital analogy circuits to output "France" instead of the correct target country — direct evidence of relational transfer

## Live Resources

- [Attribution Graphs on Neuronpedia](https://www.neuronpedia.org/gemma-2-2b/graph?slug=analog_berlin)
- [Core Circuit Feature List](https://neuronpedia.org/quick-list/?name=Analogical%20Reasoning%20Core%20Circuit)
- [Interactive Presentation](https://kkrampis.github.io/autocircuit/presentation.html)

---

## Repository Structure

```
├── index.md                        # Paper (rendered as GitHub Pages site)
├── autocircuit_tools_new.py        # Python tooling for attribution graph analysis
├── run_pending_experiments.py      # Steering experiment runner
├── graphs/
│   ├── analog_comparison.json      # Cross-graph feature overlap (510/277/180)
│   └── analog_steering_validation.json  # All steering experiment results
└── circuits/
    ├── analogical_circuit_180features.json   # 180 core features with per-graph breakdown
    ├── analogical_circuit_core_5of5.json     # 119 features in all 5 distinct graphs
    └── analogical_reasoning_circuit.json     # Named circuit nodes
```

## Citation

```bibtex
@article{alagbe2026analogical,
  title   = {Mechanistic Interpretability of Analogical Reasoning in {Gemma-2-2B}:
             A Sparse Autoencoder Attribution Graph Analysis},
  author  = {Alagbe, Olalekan and Lawrence, Joseph and Maheshwar, Anish and Krampis, Konstantinos},
  year    = {2026},
  month   = {March},
  note    = {Neuronpedia API \texttt{gemmascope-transcoder-16k} SAE analysis}
}
```
