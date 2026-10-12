#!/usr/bin/env bash
# Sequential GPU queue for the robustness runs of the paper. Run from the repository root.
#   1. walk-forward fidelity generations on the three synthetic models
#   2. two extra training seeds (43, 44) for the synthetic diffusion models,
#      each followed by its counterfactual evaluation
#   3. two extra training seeds for the MacroVAE (linear and non-linear data)
# Every run writes to its own checkpoints/results branch; paper runs are untouched.
set -u
export PYTHONUTF8=1
PY=${PY:-python}
L=results/_logs
mkdir -p "$L"

step() {  # step <log name> <command...>
    local name=$1; shift
    echo "[$(date +%T)] start $name"
    "$@" > "$L/$name.log" 2>&1
    echo "[$(date +%T)] done  $name exit=$?"
}

DIFF_TRAIN="--restart --epochs 1000 --wf_epochs 80 --batch 1024 --lr 1e-3"
MVAE_TRAIN="--restart --hidden_dim 128 --latent_dim 32 --cond_embed_dim 8 --batch 64 --lr 4e-4 --epochs 1000 --wf_epochs 50 --kl_weight 0.01 --kl_anneal_eps 1000"

# ── 1. fidelity generations on the synthetic models ─────────────────────────
step wfeval_synthetic_complex_filmonly $PY scripts/generate.py wf_eval --config configs/synthetic_complex_filmonly/baseline.yaml --model diffusion
step wfeval_synthetic_complex_snapfilm $PY scripts/generate.py wf_eval --config configs/synthetic_complex_snapfilm/baseline.yaml --model diffusion
step wfeval_synthetic_complex_macrovae $PY scripts/generate.py wf_eval --config configs/synthetic_complex/baseline.yaml --model macrovae

# ── 1b. Temporal (with QT) on the real panel retrained on this machine ──────
# Same recipe as the paper checkpoint; gives a same-machine partner for the
# head-free Raw model (real_noqt_filmonly) in the quantile-transform ablation.
step train_real_filmonly_rerun $PY scripts/train.py --config configs/seeds/real_filmonly_rerun.yaml $DIFF_TRAIN --seed 42 --macro_encoder symmetric
step wfeval_real_filmonly_rerun $PY scripts/generate.py wf_eval --config configs/seeds/real_filmonly_rerun.yaml --model diffusion

# ── 2. diffusion seeds ──────────────────────────────────────────────────────
for s in 43 44; do
    for spec in "synthetic_complex_filmonly symmetric lin" "synthetic_complex_snapfilm snapshot lin" "synthetic_complex_nl_filmonly symmetric nl"; do
        set -- $spec; b=$1; enc=$2; kind=$3; cfg=configs/seeds/${b}_seed${s}.yaml
        step train_${b}_seed${s} $PY scripts/train.py --config $cfg $DIFF_TRAIN --seed $s --macro_encoder $enc
        step cf_ate_${b}_seed${s} $PY scripts/generate.py counterfactual_synthetic --config $cfg --model diffusion --latent_mode fresh
        if [ "$kind" = lin ]; then
            step cf_ite_${b}_seed${s} $PY scripts/generate.py counterfactual_synthetic --config $cfg --model diffusion --latent_mode invert
        fi
    done
done

# ── 3. MacroVAE seeds ───────────────────────────────────────────────────────
for s in 43 44; do
    for b in synthetic_complex synthetic_complex_nl; do
        cfg=configs/seeds/${b}_seed${s}.yaml
        step train_mvae_${b}_seed${s} $PY scripts/train_macrovae.py --config $cfg $MVAE_TRAIN --seed $s
        step cf_post_mvae_${b}_seed${s} $PY scripts/generate.py counterfactual_synthetic --config $cfg --model macrovae --mvae_latent posterior
        step cf_prior_mvae_${b}_seed${s} $PY scripts/generate.py counterfactual_synthetic --config $cfg --model macrovae --mvae_latent prior --mvae_prior_samples 1
    done
done
echo "[$(date +%T)] queue finished"
