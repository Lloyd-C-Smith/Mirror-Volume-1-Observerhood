# Lab I: Recursive Self Model Reliability

**Recursive Self-Model Reliability Improves Viability Under Self-Relevant Perturbation**

This folder contains the paper, code, fixed-seed data and figures for Mirror Observerhood Lab I, part of the Computational Observerhood Labs of Mirror Programme, Volume I: Observerhood.

## Citation

Smith, L. C. (2026). *Mirror Observerhood Lab I: Recursive Self-Model Reliability Improves Viability Under Self-Relevant Perturbation*. Zenodo. https://doi.org/10.5281/zenodo.21157523

## Contents

```text
Mirror_Observerhood_Lab_I.pdf
mirror_lab_i.py
requirements.txt
data/
  Mirror_Observerhood_Lab_I_results.csv
  Mirror_Observerhood_Lab_I_summary_by_condition.csv
figures/
  make_publication_figures.py
  mirror_observerhood_lab_i_viability.pdf
  mirror_observerhood_lab_i_viability.png
  mirror_observerhood_lab_i_survival.pdf
  mirror_observerhood_lab_i_survival.png
  mirror_observerhood_lab_i_mirror_advantage.pdf
  mirror_observerhood_lab_i_mirror_advantage.png
CITATION.cff
LICENSE-CODE.txt
LICENSE-DATA-FIGURES.txt
```

## Run

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

Reproduce the 6,000-episode experiment:

```bash
python mirror_lab_i.py --episodes 300 --seed 12345 --outdir outputs/lab_I
```

This writes raw results, summary tables and generated figures to:

```text
outputs/lab_I/data/
outputs/lab_I/figures/
```

To regenerate the publication figures from the fixed-seed data already included in this folder:

```bash
python figures/make_publication_figures.py
```

## Headline result

Across 6,000 deterministic episodes, the Mirror reliability agent shows its largest advantage under false self-location:

- strongest non-Mirror mean viability: 139.2;
- Mirror mean viability: 223.6;
- strongest non-Mirror survival: 7.7%;
- Mirror survival: 63.0%.

The same scalar reliability mechanism underperforms under generic sensor degradation, supporting the negative result that reliability tracking is not a general-purpose performance booster.

## Licensing

Paper, generated data and figures: CC BY 4.0.

Source code: MIT License.
