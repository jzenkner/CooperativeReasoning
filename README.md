# TIIPS: Transductively Informed Inductive Program Synthesis

**ECML PKDD 2026 · Naples** &nbsp;|&nbsp; [🌐 Project Page](https://jzenkner.github.io/CooperativeReasoning/) &nbsp;|&nbsp; [📄 arXiv](https://arxiv.org/abs/2505.14744) &nbsp;|&nbsp; [📑 Paper (ECML)](#) &nbsp;|&nbsp; [📎 Appendix](Appendix.pdf)

---

> *When neither paradigm should unconditionally own the trajectory.*

Official implementation of **TIIPS**, introduced in:

> **Beyond Either-Or Reasoning: Transduction and Induction as Cooperative Problem-Solving Paradigms**  
> Janis Zenkner, Tobias Sesterhenn, Christian Bartelt  
> *Clausthal University of Technology*  
> ECML PKDD 2026, Naples, Italy

---

## Overview

Programming-by-example (PBE) solvers have long been split into two camps. **Inductive** approaches search over symbolic program spaces — interpretable and reusable, but vulnerable to combinatorial explosion. **Transductive** approaches predict outputs directly from examples — flexible, but producing no inspectable artifact.

Existing hybrids (e.g. ExeDec) combine the two **hierarchically**: the transductive model fixes the full trajectory, and the inductive model is constrained to realise each prescribed subgoal. A single wrong transductive prediction cascades into failure.

**TIIPS** introduces a different principle: **cooperative** transductive-inductive problem solving. Rather than one paradigm dictating to the other, both contribute dynamically along the solution trajectory. Each transductive step acts as a *search horizon reset* — a springboard, not a cage.

### The Three Cooperation Criteria

| Criterion | What it rules out |
|---|---|
| **Dual Agency** — both paradigms are active solvers | Approaches where induction is merely a preprocessing step |
| **Interleaved Granularity** — switching at individual steps | Ensembles where paradigms never interact within a trajectory |
| **Search Autonomy Preservation** — each transductive step is a new search root | Hybrids that lock induction into a transductively prescribed corridor |

### Key Results

- **+10 pp** over ExeDec on DeepCoder (avg. across all compositional generalisation categories)
- **+13 pp** over ExeDec on LambdaBeam
- **221 tasks** solved by TIIPS on DeepCoder that are unreachable by *any* ensemble of baselines — the cooperative dividend
- On RobustFill (single-trace domain), TIIPS and ExeDec converge exactly as the framework predicts

---

## Installation

```bash
pip install numpy tensorflow absl-py
pip install flax==0.5.3
pip install jax==0.3.25 jaxlib==0.3.25 -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install tqdm
```

> **Note:** adjust relative paths to absolute paths for model checkpointing.

---

## Datasets

TIIPS is evaluated on the compositional generalisation splits from [ExeDec](https://arxiv.org/abs/2307.13883) across three domains:

| Domain | Task type | Key property |
|---|---|---|
| **RobustFill** | String manipulation | Single canonical solution trace; tests convergence with ExeDec |
| **DeepCoder** | List manipulation | Multiple valid decompositions; cooperation outperforms subordination |
| **LambdaBeam** | List manipulation with lambdas and `If` | Branching execution; amplifies gains of search autonomy |

Each domain includes six generalisation splits: `NONE` (in-distribution), `LENGTH_GENERALIZATION`, `COMPOSE_DIFFERENT_CONCEPTS`, `SWITCH_CONCEPT_ORDER`, `COMPOSE_NEW_OP`, `ADD_OP_FUNCTIONALITY`.

---

## Training

First generate data:

```bash
bash tasks/deepcoder/dataset/run_data_generation.sh
bash tasks/robustfill/dataset/run_data_generation.sh
bash tasks/lambdabeam/dataset/run_data_generation.sh
```

Then train the TIIPS models (each script trains both the inductive synthesizer and the transductive guide):
- Training mode for inductive synthesizer: model_type=synthesizer_model
- Training mode for transductive decomposition model: model_type=decomposition_model
- Training mode for inductive baseline model: model_type=joint_model

```bash
bash spec_decomposition/run_deepcoder_training.sh
bash spec_decomposition/run_robustfill_training.sh
bash spec_decomposition/run_lambdabeam_training.sh
```

Adjust scripts for your environment (TPU, SLURM, Docker) as needed.

---

## Evaluation

Evaluate trained checkpoints end-to-end:

```bash
bash spec_decomposition/run_deepcoder_end_to_end_predict.sh
bash spec_decomposition/run_robustfill_end_to_end_predict.sh
bash spec_decomposition/run_lambdabeam_end_to_end_predict.sh
```

- Evaluation mode for TIIPS: prediction_type=tiips
- Evaluation mode for ExeDec: prediction_type=separate
- Evaluation mode for inductive baseline: prediction_type=baseline

`end_to_end_predict.py` runs the incremental intervention schedule: it attempts purely inductive synthesis first (j = 0), then progressively introduces transductive guidance steps (j = 1 … J), returning the first successful program found. j = 0 recovers the inductive baseline; j = J recovers ExeDec; all gains come from the cooperative regime in between.

---

## Citation

```bibtex
@inproceedings{zenkner2026tiips,
  title     = {Beyond Either-Or Reasoning: Transduction and Induction
               as Cooperative Problem-Solving Paradigms},
  author    = {Zenkner, Janis and Sesterhenn, Tobias and Bartelt, Christian},
  booktitle = {Proceedings of the European Conference on Machine Learning
               and Principles and Practice of Knowledge Discovery
               in Databases (ECML PKDD 2026)},
  year      = {2026},
  note      = {To appear}
}
```
