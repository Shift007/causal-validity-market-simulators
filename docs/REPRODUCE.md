# Reproducing the paper

All commands run from the repository root. Training uses seed 42 unless stated.
A GPU with 8 GB of memory is enough; times are for a laptop RTX 5070.

## Experiment branches

| Config | Data | Model |
|---|---|---|
| `configs/real_filmonly/baseline.yaml` | real panel, 14 assets | Temporal diffusion |
| `configs/real_noqt_filmonly/baseline.yaml` | real panel, no quantile transform | Temporal diffusion (Raw) |
| `configs/real/baseline.yaml` | real panel | MacroVAE |
| `configs/synthetic_complex_filmonly/baseline.yaml` | synthetic, linear mean | Temporal diffusion |
| `configs/synthetic_complex_snapfilm/baseline.yaml` | synthetic, linear mean | Snapshot diffusion |
| `configs/synthetic_complex/baseline.yaml` | synthetic, linear mean | MacroVAE |
| `configs/synthetic_complex_nl_filmonly/baseline.yaml` | synthetic, regime switch | Temporal diffusion |
| `configs/synthetic_complex_nl_snapfilm/baseline.yaml` | synthetic, regime switch | Snapshot diffusion |
| `configs/synthetic_complex_nl/baseline.yaml` | synthetic, regime switch | MacroVAE |
| `configs/seeds/*.yaml` | as above | additional training seeds |

## Data

```bash
# synthetic branches
python scripts/synthetic_dgp_complex.py --config <config>
python scripts/preprocess.py            --config <config>

# real branches (needs FRED_API_KEY)
python scripts/ingest.py     --config <config>
python scripts/preprocess.py --config <config>
```

## Training

```bash
# diffusion: --macro_encoder symmetric (Temporal) or snapshot (Snapshot)
python scripts/train.py --config <config> --restart --epochs 1000 --wf_epochs 80 \
    --batch 1024 --lr 1e-3 --seed 42 --macro_encoder symmetric

# MacroVAE, synthetic branches
python scripts/train_macrovae.py --config <config> --restart --hidden_dim 128 --latent_dim 32 \
    --cond_embed_dim 8 --batch 64 --lr 4e-4 --epochs 1000 --wf_epochs 50 \
    --kl_weight 0.01 --kl_anneal_eps 1000 --seed 42

# MacroVAE, real panel
python scripts/train_macrovae.py --config configs/real/baseline.yaml --restart --hidden_dim 128 \
    --latent_dim 32 --cond_embed_dim 8 --lr 4e-4 --epochs 2000 --wf_epochs 100 \
    --kl_weight 0.01 --kl_anneal_eps 2000 --seed 42
```

`--restart` deletes the checkpoints of that branch before training.

## Table 2: distributional fidelity

```bash
python scripts/generate.py wf_eval --config <config> --model diffusion    # or --model macrovae
python scripts/score_fidelity.py --real <run>/pooled/real.npy \
    --model "Temporal=<run>/pooled/diff_synth.npy" \
    --model "MacroVAE=<run>/pooled/mvae_synth.npy" \
    --pooling sample --out results/paper_scores/fidelity.json
```

`<run>` is the `results/<branch>/wf_eval/run_<timestamp>` folder written by the
first command. The same output file holds the within-window statistics quoted
in the text (autocorrelation, leverage, cross-asset correlation).

## Table 3: interventional response, linear environment

```bash
python scripts/generate.py counterfactual_synthetic --config <config> --model diffusion --latent_mode fresh
python scripts/generate.py counterfactual_synthetic --config <config> --model diffusion --latent_mode invert
python scripts/generate.py counterfactual_synthetic --config configs/synthetic_complex/baseline.yaml \
    --model macrovae --mvae_latent posterior
python scripts/generate.py counterfactual_synthetic --config configs/synthetic_complex/baseline.yaml \
    --model macrovae --mvae_latent prior --mvae_prior_samples 1
python scripts/score_paper_causal.py --results results --out results/paper_scores/causal.json
```

`score_paper_causal.py` lists the run folders it reads at the top of the file;
edit them to point at your own runs.

Robustness checks quoted in the text, each written to its own results branch:

```bash
# more latent draws per window
python scripts/cf_variant.py --tag M16 --config <config> --model diffusion --ate_samples 16
python scripts/cf_variant.py --tag prior16 --config configs/synthetic_complex/baseline.yaml \
    --model macrovae --mvae_latent prior --mvae_prior_samples 16
# held-out windows only
python scripts/cf_variant.py --tag oos --config <config> --model diffusion --window_pool test --n_windows 447
# exact unit-level counterfactual, from saved runs (CPU)
python scripts/score_unit_level.py --config <config> \
    --run "Temporal=<run>:diff_effects.npy" --out results/paper_scores/unit_level.json
# regime-switching environment: asymmetry index against the exact re-simulated average (Table 4, CPU)
python scripts/score_nl_exact_truth.py --config configs/synthetic_complex_nl_filmonly/baseline.yaml \n    --run "Temporal=<run>:diff_effects.npy" --out results/paper_scores/nl_exact_truth.json
```

Linear references quoted in the text (OLS, ridge and VAR on the same environment, CPU):

```bash
python scripts/baselines_synthetic.py            # writes results/synthetic_complex_baselines/baselines.json
python scripts/svar_synthetic_benchmark.py --config configs/synthetic_complex_nl/baseline.yaml
```

The baselines are scored on 1000 boundary days drawn from the simulation, not on
the 2000 model windows.

## Table 4: regime-switching environment

The same counterfactual commands on the `synthetic_complex_nl*` configs, then
`score_paper_causal.py` for the asymmetry index with its bootstrap interval, and

```bash
python scripts/baseline_nl_mlp.py --reference <temporal non-linear run>
```

for the MLP regressor.

## Figures

```bash
python scripts/fig_kde_paper.py --real-run <run> --temporal-run <run> --raw-run <run> --out fig_kde_ablations.png
python scripts/fig_paper_causal.py trajectories  --temporal <run> --snapshot <run> --macrovae <run> --out fig_fidelity_extreme_shocks.png
python scripts/fig_paper_causal.py dose_response --temporal <run> --snapshot <run> --macrovae <run> --out fig_nl_unified.png
```

Each figure script writes a JSON file next to the image with the runs it read.
