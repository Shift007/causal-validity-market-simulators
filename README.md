# Measuring the Causal Validity of Macro-Conditioned Market Simulators

Code for the paper by Marco Gervatini, Francesco Puoti and Fabrizio Pittorino,
7th ACM International Conference on AI in Finance (ICAIF '26).

The repository contains

- a synthetic macro-finance environment whose response to a macro intervention
  is known exactly (`scripts/synthetic_dgp_complex.py`), in a linear-mean and a
  regime-switching variant;
- the evaluation protocol: prior-marginalised and latent-preserving estimates of
  the interventional response, scored against the exact truth
  (`src/generation/counterfactual_synthetic.py`);
- a conditional diffusion model on wavelet images with two macro encoders,
  Temporal and Snapshot (`src/diffusion/lab.py`);
- the MacroVAE baseline of Kubiak et al. (ICAIF '25), used from the authors'
  repository as a git submodule;
- the scripts that compute every table and figure of the paper.

## Setup

```bash
git clone --recurse-submodules https://github.com/Shift007/causal-validity-market-simulators.git
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.12, PyTorch 2.11 (CUDA 12.8) on one 8 GB GPU. Install the
PyTorch build that matches your CUDA version before the requirements.

The real-data experiments download prices from Yahoo Finance and macro series
from FRED. Get a free key at <https://fred.stlouisfed.org> and export it as
`FRED_API_KEY`. The synthetic experiments need no external data.

## Layout

```
configs/    one YAML per experiment branch (data, checkpoints and results folders)
scripts/    command-line entry points: data, training, evaluation, scoring, figures
src/        model, preprocessing, generation and metrics
MacroVAE/   the baseline, as a submodule of the authors' repository
tests/      unit tests of the metrics
```

Each config names its own `processed_data/`, `checkpoints/` and `results/`
sub-folders, so experiments never overwrite each other. Evaluation runs are
written to `results/<branch>/<task>/run_<timestamp>/` together with the arrays
they were computed from; the scoring scripts read those arrays.

## Pipeline

All commands run from the repository root. See [`docs/REPRODUCE.md`](docs/REPRODUCE.md)
for the full list behind each table.

```bash
CFG=configs/synthetic_complex_filmonly/baseline.yaml

# 1. data: simulate the environment and build the training windows
python scripts/synthetic_dgp_complex.py --config $CFG
python scripts/preprocess.py            --config $CFG

# 2. train the Temporal diffusion model (about 25 minutes on a laptop GPU)
python scripts/train.py --config $CFG --restart --epochs 1000 --wf_epochs 80 \
    --batch 1024 --lr 1e-3 --seed 42 --macro_encoder symmetric

# 3. interventional response against the exact truth
python scripts/generate.py counterfactual_synthetic --config $CFG --model diffusion --latent_mode fresh    # prior-marginalised
python scripts/generate.py counterfactual_synthetic --config $CFG --model diffusion --latent_mode invert   # latent-preserving

# 4. scores with bootstrap intervals
python scripts/score_paper_causal.py --results results --out results/paper_scores/causal.json
```

## Using the benchmark with another generator

The environment and the scoring are independent of the models in this
repository. A generator only has to produce, for each evaluation window and
shock, the difference between its output under the shocked and the factual
macro state, as an array of shape `(windows, shocks, days, assets)`.
`TheoreticalCounterfactualComplex` in `scripts/synthetic_dgp_complex.py` returns
the closed-form average response for the same windows (on the regime-switching
variant it reads the regime on the expected macro path; `scripts/score_nl_exact_truth.py`
gives the exact average by re-simulation), and `scripts/score_unit_level.py`
the exact response of each individual window.

## Citation

```bibtex
@inproceedings{gervatini2026causal,
  title     = {Measuring the Causal Validity of Macro-Conditioned Market Simulators},
  author    = {Gervatini, Marco and Puoti, Francesco and Pittorino, Fabrizio},
  booktitle = {Proceedings of the 7th ACM International Conference on AI in Finance (ICAIF '26)},
  year      = {2026},
  doi       = {10.1145/3858063.3858831}
}
```

## License

The code in this repository is released under the MIT License (see `LICENSE`).
The `MacroVAE` submodule is the code of Kubiak et al., kept in its authors'
repository; it is not part of this repository and is not covered by this
license.
