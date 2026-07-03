# Mirror Observerhood Lab III

**Actionable Reliability and Cost-Sensitive Repair in Viability-Constrained Agents**

This folder contains the paper, code, data and figures for Mirror Observerhood Lab III, part of the Computational Observerhood Labs of Mirror Programme, Volume I: Observerhood.

Lab III tests whether reliability variables become viability-relevant only when they are coupled to actionable and cost-sensitive repair. It follows Lab I, which showed that scalar self-model reliability can help under false self-location, and Lab II, which showed that decomposed reliability alone is insufficient.

DOI: https://doi.org/10.5281/zenodo.21161631

Canonical citation:

Smith, L. C. (2026). *Mirror Observerhood Lab III: Actionable Reliability and Cost-Sensitive Repair in Viability-Constrained Agents*. Zenodo. https://doi.org/10.5281/zenodo.21161631

## Contents

- `Mirror_Observerhood_Lab_III.pdf` - publication paper.
- `mirror_lab_iii.py` - single-file simulation, experiment runner and figure generator.
- `requirements.txt` - Python dependencies.
- `data/` - fixed-seed publication outputs.
- `figures/` - publication figures generated from the fixed-seed outputs.
- `CITATION.cff` - citation metadata for the release.
- `LICENSE-CODE.txt` - MIT license for the source code.
- `LICENSE-DATA-FIGURES.txt` - CC BY 4.0 license notice for data and figures.

## Run

```bash
pip install -r requirements.txt
python mirror_lab_iii.py --episodes 200 --seed 12345 --outdir outputs/lab_iii
```

The default command reproduces the 4,800-episode dataset, summary tables, cost-sensitive advantage table and figures used in the paper.

## Publication settings

- Conditions: control, false self-location, sensor degradation, map corruption, mixed low cost, mixed high cost.
- Agent architectures: no reliability, passive reliability, threshold repair, cost-sensitive Mirror.
- Episodes: 200 per agent-condition pair.
- Total episodes: 4,800.
- Base seed: 12345.

## License

Paper, figures and data: Creative Commons Attribution 4.0 International (CC BY 4.0).  
Code: MIT License.
