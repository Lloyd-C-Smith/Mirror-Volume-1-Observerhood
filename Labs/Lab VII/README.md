# Mirror Observerhood Lab VII

**Autobiographical Continuity Under Memory Injection and Goal Drift**

**DOI:** https://doi.org/10.5281/zenodo.21171374

This reproducibility package accompanies Mirror Observerhood Lab VII, part of the Computational Observerhood Labs of Mirror Programme, Volume I: Observerhood.

Lab VII tests the transition from memory storage to autobiographical continuity. Language-like agents receive true reminders, false memories, corrupted summaries, identity injections, goal-drift prompts and fake tool outputs across long horizons. Five architectures are compared: a stateless language agent, a naive memory agent, a self-model memory agent, a Mirror commit-gated agent and a recursive Mirror agent.

## Run

```bash
pip install -r requirements.txt
python mirror_lab_vii.py --runs 100 --seed 77000 --episodes 60 --outdir outputs/lab_vii
```

The canonical paper run uses 100 long-horizon runs per agent-condition cell, across eight conditions and five agents. Each run contains 60 simulated episodes, producing 4,000 long-horizon run rows and 240,000 simulated agent episodes.

## Contents

- `mirror_lab_vii.py` - single-file simulation and output generator.
- `requirements.txt` - Python package requirements.
- `data/` - fixed-seed run-level results, summary data and identity-advantage data.
- `figures/` - publication figures and figure-generation helper.
- `CITATION.cff` - citation metadata.
- `LICENSE-CODE.txt` - MIT licence for code.
- `LICENSE-DATA-FIGURES.txt` - CC BY 4.0 licence notice for data and figures.

## Main result

The best Mirror agent outperforms the strongest non-Mirror baseline across all tested conditions. The effect is small under control and large under adversarial autobiographical perturbation. Commit-gated agents preserve viability, identity continuity and memory integrity by refusing to treat every candidate memory as durable self-memory.

The recursive Mirror agent does not uniformly dominate the simpler commit-gated agent. Additional audit can become costly, reinforcing the broader Mirror threshold result: recursion and repair are useful only where the avoided loss exceeds the cost of checking.

## Citation

Smith, L. C. (2026). Mirror Observerhood Lab VII: Autobiographical Continuity Under Memory Injection and Goal Drift. Zenodo. https://doi.org/10.5281/zenodo.21171374

## Licence

Code is released under the MIT Licence. Data and figures are released under CC BY 4.0.
