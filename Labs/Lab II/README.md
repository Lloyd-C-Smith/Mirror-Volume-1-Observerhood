# Mirror Observerhood Lab II

**Channel-Decomposed Reliability and the Limits of Reliability Tracking Alone**

This folder contains the paper, code, fixed-seed data and figures for **Mirror Observerhood Lab II**, part of the Computational Observerhood Labs of the Mirror Programme, Volume I: Observerhood.

DOI: <https://doi.org/10.5281/zenodo.21159572>

## Citation

Smith, L. C. (2026). *Mirror Observerhood Lab II: Channel-Decomposed Reliability and the Limits of Reliability Tracking Alone*. Zenodo. <https://doi.org/10.5281/zenodo.21159572>

## Contents

```text
Mirror_Observerhood_Lab_II.pdf
mirror_lab_ii.py
requirements.txt
data/
  Mirror_Observerhood_Lab_II_results.csv
  Mirror_Observerhood_Lab_II_summary_by_condition.csv
figures/
  make_publication_figures.py
  mirror_observerhood_lab_ii_viability.pdf
  mirror_observerhood_lab_ii_viability.png
  mirror_observerhood_lab_ii_survival.pdf
  mirror_observerhood_lab_ii_survival.png
  mirror_observerhood_lab_ii_decomposed_advantage.pdf
  mirror_observerhood_lab_ii_decomposed_advantage.png
CITATION.cff
LICENSE-CODE.txt
LICENSE-DATA-FIGURES.txt
```

## Experiment summary

Lab II tests whether decomposing reliability into self-state, sensor-channel and world-map components is sufficient to improve viability under channel-specific perturbation.

The experiment compares four architectures in a controlled grid-world:

1. predictor-only agent;
2. self-model agent without reliability tracking;
3. scalar Mirror reliability agent; and
4. decomposed Mirror reliability agent.

Across 5,760 fixed-seed episodes, decomposed reliability does not dominate scalar reliability or simpler baselines. The result is a constructive negative finding: channel decomposition is useful for attribution, but reliability becomes viability-relevant only when diagnosis changes action, repair or commitment in a cost-sensitive way.

## Reproducing the experiment

Install the requirements:

```bash
pip install -r requirements.txt
```

Run the experiment with the publication settings:

```bash
python mirror_lab_ii.py --episodes 240 --seed 9101 --outdir outputs/lab_II
```

This writes raw episode results, summary tables and figures to:

```text
outputs/lab_II/data/
outputs/lab_II/figures/
```

To regenerate the publication-style figures from the packaged data without re-running the experiment:

```bash
python figures/make_publication_figures.py
```

The fixed-seed outputs in `data/` are the outputs used for the paper figures and tables.

## Relationship to the Mirror Programme

Lab II follows Lab I. Lab I showed that scalar self-model reliability can help under false self-location, but can become brittle under generic sensor degradation. Lab II tests the natural refinement: decomposing reliability into self, sensor and map channels.

The result motivates Lab III: reliability variables must become actionable and cost-sensitive before they can support a disciplined observerhood predicate.

## Licensing

Paper, generated data and figures: CC BY 4.0.

Source code: MIT License.
