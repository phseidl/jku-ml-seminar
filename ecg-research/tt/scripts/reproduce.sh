#!/usr/bin/env bash
# =============================================================================
# reproduce.sh -- regenerate the three headline results of the xLSTM-ECG
# reproduction from scratch, portably: no hard-coded machine paths, no `screen`
# busy-waits, configurable GPUs, resumable.
#
# Each headline is produced the same way the reported numbers were:
#   1. train N seeds with `run_training` (the recipe = config.yaml defaults +
#      a few explicit overrides),
#   2. average the per-seed sigmoid scores with `ensemble_eval`,
#   3. put a 95% bootstrap CI on the ensemble AND on the best single seed with
#      `bootstrap_ci`.
# This script just wraps that chain into one command per headline.
#
# USAGE
#   bash scripts/reproduce.sh <target>
#     ptbxl     PTB-XL headline   results_matching 10-seed ensemble  -> ~0.9088, CI [0.8990, 0.9187]
#     combined  PTB-XL footnote   combined-axis 10-seed ensemble -> ~0.9100, CI [0.9000, 0.9199]
#     georgia   Georgia depth sweep  results_matching nb=2/4/6 (nb=6 headline) -> ~0.9460, CI [0.9334, 0.9576]
#     ablations ablation tables + as-described baseline (paper Tables 2-5 + leave-one-out)
#     all       every target above, in order (the full reported program; hours)
#
# ENV OVERRIDES (all optional)
#   GPUS="0"          comma-separated GPU ids; seeds round-robin across them and
#                     run that many at a time in parallel (default: 1 GPU, serial).
#   SEEDS="42 123 ..." space-separated seeds (default: the 10 reported seeds;
#                     set e.g. SEEDS="42" for a fast single-seed smoke test).
#   PYTHON=".venv/bin/python"   interpreter to use.
#   EXTRA="data.data_dir=/p data.cache_dir=/p"  extra run_training overrides,
#                     e.g. to point at where YOU downloaded the data (no spaces
#                     inside a path). PTB-XL paths also live in config/config.yaml.
#
# RESUMABILITY: a seed whose result.json already exists is skipped; delete its
# results/repro_*/s<seed>/result.json to force a retrain. Training writes to
# results/repro_*/ (gitignored output, not part of the code).
# =============================================================================
set -euo pipefail

# Resolve the submission root from this script's location (scripts/ -> ..),
# so the script works from any working directory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-.venv/bin/python}"
read -r -a SEEDS    <<< "${SEEDS:-42 123 456 789 999 11 13 17 23 29}"
IFS=',' read -r -a GPU_ARR <<< "${GPUS:-0}"
NGPU=${#GPU_ARR[@]}
GPUS_RAW="${GPUS:-0}"   # raw GPU/worker list passed straight to the sweep harnesses' --gpus
if [ -n "${EXTRA:-}" ]; then read -r -a EXTRA_ARR <<< "$EXTRA"; else EXTRA_ARR=(); fi

PTBXL_CLASSES=(NORM MI STTC CD HYP)
GEORGIA_CLASSES=(NSR AF IAVB LBBB RBBB SB STach)

# train_seeds <outdir> <run_training override tokens...>
train_seeds () {
  local outdir="$1"; shift
  local overrides=("$@")
  local tag; tag="$(basename "$outdir")"
  mkdir -p "$outdir" logs
  local pids=() i=0
  for seed in "${SEEDS[@]}"; do
    local d="$outdir/s$seed"
    if [ -f "$d/result.json" ]; then
      echo "[$tag] s$seed already done -- skipping (delete $d/result.json to retrain)"
      i=$((i + 1)); continue
    fi
    local gpu=${GPU_ARR[$((i % NGPU))]}
    mkdir -p "$d/checkpoints" "$d/logs"
    echo "[$tag] training seed $seed on GPU $gpu  (log: logs/${tag}_s${seed}.log)"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m scripts.run_training \
      "${overrides[@]}" ${EXTRA_ARR[@]+"${EXTRA_ARR[@]}"} \
      training.seed="$seed" \
      training.checkpoint_dir="$d/checkpoints" \
      training.log_dir="$d/logs" \
      training.result_file="$d/result.json" \
      > "logs/${tag}_s${seed}.log" 2>&1 &
    pids+=("$!"); i=$((i + 1))
    # Throttle to NGPU concurrent jobs (one wave per GPU set).
    if [ "${#pids[@]}" -ge "$NGPU" ]; then wait "${pids[@]}"; pids=(); fi
  done
  [ "${#pids[@]}" -gt 0 ] && wait "${pids[@]}"
  return 0
}

# ensemble_and_ci <outdir> <class names...>
ensemble_and_ci () {
  local outdir="$1"; shift
  local classes=("$@")
  local tag; tag="$(basename "$outdir")"
  local ckpts
  ckpts=$(ls -1 "$outdir"/s*/checkpoints/best.pt 2>/dev/null | sort)
  if [ -z "$ckpts" ]; then echo "[$tag] ERROR: no checkpoints under $outdir"; return 1; fi
  echo "[$tag] ensembling $(echo "$ckpts" | wc -l) checkpoints"
  CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" $PY -m scripts.ensemble_eval \
    --checkpoints $ckpts \
    --out "$outdir/ensemble.json"
  # CI on the averaged ensemble scores (sidecar npy written by ensemble_eval).
  $PY -m scripts.bootstrap_ci \
    --labels "$outdir/ensemble_labels.npy" \
    --scores "$outdir/ensemble_scores.npy" \
    --class-names "${classes[@]}" \
    --out "$outdir/ci_ensemble.json"
  # CI on the best single seed (by final test_auroc in its result.json).
  local best
  best=$($PY -c "import json,glob;print(max(glob.glob('$outdir/s*/result.json'),key=lambda p:json.load(open(p))['test_auroc']).replace('result.json','checkpoints/best.pt'))")
  echo "[$tag] best individual checkpoint: $best"
  CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" $PY -m scripts.bootstrap_ci \
    --checkpoint "$best" \
    --out "$outdir/ci_best.json"
  echo "[$tag] DONE -> ensemble: $outdir/ensemble.json | CIs: $outdir/ci_ensemble.json, $outdir/ci_best.json"
}

repro_ptbxl () {
  echo "=== PTB-XL headline: results_matching (paper 4.3 + dropout 0.1 + emb 512 + nb 4; keeps paper StepLR) ==="
  local out=results/repro_ptbxl_results_matching
  train_seeds "$out" model.dropout=0.1 model.embedding_dim=512 model.num_blocks=4
  ensemble_and_ci "$out" "${PTBXL_CLASSES[@]}"
}

repro_combined () {
  echo "=== PTB-XL footnote: combined-axis (results_matching + cosine_warmup(5)) ==="
  local out=results/repro_ptbxl_combined
  train_seeds "$out" model.dropout=0.1 model.embedding_dim=512 model.num_blocks=4 \
    training.lr_scheduler=cosine_warmup training.warmup_epochs=5
  ensemble_and_ci "$out" "${PTBXL_CLASSES[@]}"
}

repro_georgia () {
  echo "=== Georgia: results_matching depth sweep nb=2/4/6 (paper-strict split); nb=6 is the headline ==="
  local base=results/repro_georgia_results_matching
  # The corrected (results_matching) recipe on Georgia, shared by every depth.
  local common=(data.dataset_type=georgia data.georgia_split_strategy=paper_strict
                model.num_classes=7 model.dropout=0.1 model.embedding_dim=512
                training.early_stopping_patience=999 training.num_epochs=20)
  # Depth x seed grid, mirroring scripts/sweep_results_matching_georgia.py: nb=2 and
  # nb=4 at 3 seeds each (the cheap depth comparison) and nb=6 at the full seed
  # set (the headline depth). The default SEEDS list is the 10 reported seeds;
  # its first three are the 3-seed family.
  local all_seeds=("${SEEDS[@]}")
  SEEDS=("${all_seeds[@]:0:3}"); train_seeds "$base/nb2" "${common[@]}" model.num_blocks=2
  SEEDS=("${all_seeds[@]:0:3}"); train_seeds "$base/nb4" "${common[@]}" model.num_blocks=4
  SEEDS=("${all_seeds[@]}");     train_seeds "$base/nb6" "${common[@]}" model.num_blocks=6
  SEEDS=("${all_seeds[@]}")
  # Headline = the nb=6 ensemble + bootstrap CI.
  ensemble_and_ci "$base/nb6" "${GEORGIA_CLASSES[@]}"
}

# The report's auxiliary tables, produced by the sweep harnesses. Unlike the
# headline targets above, these read their data paths from config/config.yaml
# (the EXTRA= override applies only to the headline run_training calls), so set
# data.data_dir / data.georgia_dir there first if your data is not at the
# defaults. Each sweep is resumable (a finished run's result.json is skipped).
repro_ablations () {
  echo "=== Ablation tables + as-described baseline (the report's auxiliary results) ==="
  # sweep_main phases A-D: single-axis search, fusion ablation (paper Table 5), the
  # as-described PTB-XL baseline 0.8861 (phase C), and the Georgia paper-recipe
  # per-depth / Table 7 (phase D).
  $PY scripts/sweep_main.py --gpus "$GPUS_RAW"
  # N_FFT ablation (paper Table 2).
  $PY scripts/sweep_nfft_ablation.py --gpus "$GPUS_RAW"
  # mask-ratio / masking-probability ablations (paper Tables 3 and 4).
  $PY scripts/sweep_mask_ablation.py --gpus "$GPUS_RAW"
  # Corrected-recipe leave-one-out ablation (the report's robustness table).
  $PY scripts/sweep_leave_one_out.py --gpus "$GPUS_RAW"
}

case "${1:-}" in
  ptbxl)     repro_ptbxl ;;
  combined)  repro_combined ;;
  georgia)   repro_georgia ;;
  ablations) repro_ablations ;;
  all)       repro_ptbxl; repro_combined; repro_georgia; repro_ablations ;;
  *) echo "usage: bash scripts/reproduce.sh {ptbxl|combined|georgia|ablations|all}"; exit 2 ;;
esac

echo "ALL DONE. Compare against the README 'Reproducing the reported numbers' table"
echo "(regenerated values land within the published bootstrap CI, not bit-exact)."
