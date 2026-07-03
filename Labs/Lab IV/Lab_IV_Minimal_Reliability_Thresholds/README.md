# Mirror Observerhood Lab IV

**Minimal Reliability Thresholds in Viability-Constrained Agents**

Author: Lloyd Christopher Smith  
Mirror Programme, Volume I: Observerhood — Computational Observerhood Labs  
Version: v1  
Date: July 2026  
DOI: https://doi.org/10.5281/zenodo.21164691

This folder contains the paper, code, data and figures for Mirror Observerhood Lab IV.

## Summary

Lab IV tests when reliability-gated self-repair becomes viability-positive. A non-repair baseline and a Mirror repair agent are compared across a sweep of self-state perturbation rate, repair cost and reliability threshold.

The experiment uses common random numbers across variants so that comparisons are architectural rather than stochastic. The default run produces 28,800 deterministic episodes.

## Run

```bash
pip install -r requirements.txt
python mirror_lab_iv.py --episodes 90 --seed 4404 --outdir outputs/lab_iv
```

The command writes:

- `outputs/lab_iv/data/Mirror_Observerhood_Lab_IV_results.csv`
- `outputs/lab_iv/data/Mirror_Observerhood_Lab_IV_summary_by_condition.csv`
- `outputs/lab_iv/data/Mirror_Observerhood_Lab_IV_all_thresholds_advantage.csv`
- `outputs/lab_iv/data/Mirror_Observerhood_Lab_IV_best_threshold_advantage.csv`
- `outputs/lab_iv/data/Mirror_Observerhood_Lab_IV_threshold_summary.csv`
- publication figures in `outputs/lab_iv/figures/`

## Contents

- `Mirror_Observerhood_Lab_IV.pdf` — standalone paper
- `mirror_lab_iv.py` — experiment code
- `requirements.txt` — Python dependencies
- `data/` — fixed-seed publication outputs
- `figures/` — generated publication figures
- `CITATION.cff` — citation metadata
- `LICENSE-CODE.txt` — MIT licence for code
- `LICENSE-DATA-FIGURES.txt` — CC BY 4.0 licence for data and figures

## Publication outputs

The `data/` folder contains the fixed-seed outputs used for the paper. The `figures/` folder contains the generated publication figures in PNG and PDF formats.

Key headline results from the default run:

- 28,800 total deterministic episodes.
- Maximum Mirror advantage of 59.4 viability points at repair cost 0 and perturbation rate 0.16.
- Maximum Mirror advantage of 57.9 viability points at repair cost 2 and perturbation rate 0.12.
- The positive reliability-repair region shrinks as repair cost rises.
- At repair cost 15, no positive perturbation rate is observed in the best-threshold summary.

## Citation

Smith, L. C. (2026). *Mirror Observerhood Lab IV: Minimal Reliability Thresholds in Viability-Constrained Agents*. Zenodo. https://doi.org/10.5281/zenodo.21164691

## Licence

Code: MIT License.  
Paper, data and figures: Creative Commons Attribution 4.0 International (CC BY 4.0).
