# Mirror Observerhood Lab V

**Recursive Reliability Under Estimator Corruption**

Author: Lloyd Christopher Smith  
Mirror Programme, Volume I: Observerhood - Computational Observerhood Labs  
Version: v1  
Date: July 2026  
DOI: https://doi.org/10.5281/zenodo.21166653

This folder contains the paper, code, data and figures for Mirror Observerhood Lab V.

## Summary

Lab V tests what happens when a first-order reliability estimate can itself become unreliable. A no-reliability baseline, a first-order reliability agent, a recursive reliability agent and an oracle upper bound are compared across seven reliability-failure conditions.

The default run produces 39,200 deterministic episodes.

## Reproduce the publication run

```bash
pip install -r requirements.txt
python mirror_lab_v.py --episodes 1400 --seed 780000 --outdir outputs/lab_v
```

The command writes:

- `outputs/lab_v/data/Mirror_Observerhood_Lab_V_results.csv`
- `outputs/lab_v/data/Mirror_Observerhood_Lab_V_summary_by_condition.csv`
- `outputs/lab_v/data/Mirror_Observerhood_Lab_V_recursive_advantage.csv`
- generated figures in `outputs/lab_v/figures/`

## Contents

- `Mirror_Observerhood_Lab_V.pdf` — publication paper
- `mirror_lab_v.py` — experiment code
- `requirements.txt` — Python dependencies
- `data/` — fixed-seed publication outputs
- `figures/` — generated figures in PNG and PDF formats
- `CITATION.cff` — citation metadata
- `LICENSE-CODE.txt` — MIT licence for code
- `LICENSE-DATA-FIGURES.txt` — CC BY 4.0 notice for data and figures

## Headline results

- 39,200 total deterministic episodes.
- Recursive reliability outperforms first-order reliability by 78.3 viability points under estimator corruption.
- Recursive reliability outperforms first-order reliability by 40.1 points under false-alarm reliability and 23.0 points under reliability blindness.
- Recursive reliability underperforms when first-order reliability is already truthful and when meta-diagnosis is too costly.
- The high-meta-cost condition shows a -61.4 viability disadvantage relative to first-order reliability.

## Citation

Smith, L. C. (2026). *Mirror Observerhood Lab V: Recursive Reliability Under Estimator Corruption*. Zenodo. https://doi.org/10.5281/zenodo.21166653

## Licence

Paper, data and figures: Creative Commons Attribution 4.0 International (CC BY 4.0).  
Code: MIT License.
