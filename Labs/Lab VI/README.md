# Mirror Observerhood Lab VI

**Neuro-Symbolic Persistence in Language-Like Agents**

This reproducibility package accompanies Mirror Observerhood Lab VI, part of the Computational Observerhood Labs of Mirror Programme, Volume I: Observerhood.

Lab VI tests whether a reliability-weighted symbolic commit layer improves robustness in a language-like agent setting. The experiment does not call a production language model. It simulates a stochastic semantic front-end that emits candidate propositions about goal, memory, self-state and tool state. Four architectures are compared: a stateless semantic agent, a memory-only agent, a persistent self-model agent and a Mirror neuro-symbolic agent with channel reliability, symbolic commit gates and diagnostic repair.

## Run

```bash
pip install -r requirements.txt
python mirror_lab_vi.py --episodes 1200 --seed 6102026 --outdir outputs/lab_vi
```

The canonical paper run uses 1,200 episodes per condition, across eight conditions and four agents, for 38,400 agent-condition episodes.

## Contents

- `mirror_lab_vi.py` - single-file simulation and output generator.
- `requirements.txt` - Python package requirements.
- `data/` - fixed-seed raw results, summary data, Mirror viability advantage data and success advantage data.
- `figures/` - publication figures and figure-generation helper.
- `CITATION.cff` - citation metadata.
- `LICENSE-CODE.txt` - MIT licence for code.
- `LICENSE-DATA-FIGURES.txt` - CC BY 4.0 licence notice for data and figures.

## Main result

The Mirror neuro-symbolic agent improves task success under several semantic perturbations, especially tool unreliability and mixed stress. Net viability improves only where reliability-weighted commitment and repair avoid more loss than they cost. Under high repair cost, the Mirror agent improves success but sharply underperforms on mean viability because diagnostic repair is too expensive.

## Citation

Smith, L. C. (2026). Mirror Observerhood Lab VI: Neuro-Symbolic Persistence in Language-Like Agents. Zenodo. https://doi.org/10.5281/zenodo.21170478

## Licence

Code is released under the MIT Licence. Data and figures are released under CC BY 4.0.
