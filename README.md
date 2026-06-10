# TIIPS: Transductively Informed Inductive Program Synthesis
This repository contains the official implementation of *TIIPS*, a framework that treats transduction (neural "hunches") and induction (symbolic rules) as cooperative agents rather than hierarchical ones. 
By interleaving these two reasoning modes, TIIPS overcomes the "cascading error" problem common in hybrid models, where a single incorrect neural prediction irrevocably prunes the correct solution.OverviewTIIPS instantiates the concept of Cooperative Transductive-Inductive Problem Solving.
It satisfies three key criteria:
- Dual Agency: Both inductive and transductive modes function as active solvers.
- Interleaved Granularity: The system can switch between modes at individual reasoning steps.
- Search Autonomy Preservation: Each transductive transition acts as a "search horizon reset," allowing the inductive solver to explore the full remaining search space unconstrained.

Across benchmarks like DeepCoder and LambdaBeam, TIIPS consistently outperforms state-of-the-art baselines by significant margins (up to 10 percentage points over hybrid models).

## ⚙️ Installation

Install required dependencies:

```bash
pip install numpy tensorflow absl-py
pip install flax==0.5.3
pip install jax==0.3.25 jaxlib==0.3.25 -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install tqdm
```

## 📁 Datasets

We evaluate **TIIPS** using the compositional generalization splits from **ExeDec** in three domains:

- **RobustFill**: string manipulation
- **DeepCoder**: list manipulation
- **LambdaBeam**: harder list manipulation with arbitray lambda functions and If-Else branching

Each domain includes 5 generalization types:

- `NONE` (test on training distribution)
- `LENGTH_GENERALIZATION`  
- `COMPOSE_DIFFERENT_CONCEPTS`  
- `SWITCH_CONCEPT_ORDER`  
- `COMPOSE_NEW_OP`  
- `ADD_OP_FUNCTIONALITY`  

## 🏋️ Training
Adapt the paths according to your preferences.
To generate training data, run:
```bash
bash tasks/deepcoder/dataset/run_data_generation.sh
bash tasks/robustfill/dataset/run_data_generation.sh
bash tasks/lambdabeam/dataset/run_data_generation.sh
```

To train TIIPS models, run:

```bash
bash tiips/run_deepcoder_training.sh
bash tiips/run_robustfill_training.sh
bash tiips/run_lambdabeam_training.sh
```

These scripts: Load data and pretrain the transductive and inductive model.
You may need to adjust them for your environment (e.g., TPU, SLURM, Docker).


## 🧪 Evaluation
To evaluate trained checkpoints:

```bash
bash tiips/run_deepcoder_end_to_end_predict.sh
bash tiips/run_robustfill_end_to_end_predict.sh
bash tiips/run_lambdabeam_end_to_end_predict.sh
```

These use end_to_end_predict.py, which: Synthesizes programs using the inductive model, invokes transductive model if synthesis fails, and iteratively builds a full solution.

