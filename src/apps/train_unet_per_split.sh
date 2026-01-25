#!/usr/bin/env bash
set -euo pipefail

# Train a MONAI U-Net for multiple predefined targets using Hydra defaults.
# Universal: supports PMRI (iterate canonical splits) and MNMv2 (iterate canonical scenarios).
# Note: train_unet.py will auto-load an existing checkpoint if one matches dataset+split (+dropout).
#
# Environment overrides (optional):
#   DATASET       pmri | mnmv2 (default pmri)
#   CUDA_DEVICE   GPU id (default 0)
#   DROPOUT       Override unet dropout (default keeps config value)
#   BATCH_SIZE    Override data.batch_size (default keeps config value)
#   SPLITS        PMRI: space-separated list of predefined split keys
#                 MNMv2: space-separated list of canonical split tags
#   LIMIT_BATCHES Override trainer.limit_train_batches (e.g. 100 for smoke test)
#   MAX_EPOCHS    Override trainer.max_epochs
#   EXTRA         Additional arbitrary Hydra overrides as a single string
#
# Examples:
#   # PMRI
#   CUDA_DEVICE=1 DATASET=pmri DROPOUT=0.2 BATCH_SIZE=16 SPLITS="promise12-i2cvb noerc-to-erc" ./train_unet_per_split.sh
#   # MNMv2
#   CUDA_DEVICE=0 DATASET=mnmv2 SPLITS="scanner-symphonytim phase-exclusive-scanners-all" ./train_unet_per_split.sh

DATASET=${DATASET:-pmri}
CUDA_DEVICE=${CUDA_DEVICE:-6}
DROPOUT=${DROPOUT:-}
BATCH_SIZE=${BATCH_SIZE:-}
LIMIT_BATCHES=${LIMIT_BATCHES:-}
MAX_EPOCHS=${MAX_EPOCHS:-}
EXTRA=${EXTRA:-}

case "${DATASET}" in
  pmri)
    # Predefined split keys
    DEFAULT_SPLITS=(
      "promise12"
      # "promise12-i2cvb"
      # "threet-to-onepointfivet"
    )
    if [ -z "${SPLITS:-}" ]; then
      TARGETS=("${DEFAULT_SPLITS[@]}")
    else
      # shellcheck disable=SC2206
      TARGETS=(${SPLITS})
    fi
    echo "Dataset=PMRI | splits: ${TARGETS[*]}"
    for T in "${TARGETS[@]}"; do
      echo "==============================="
      echo "Training U-Net | dataset=pmri | split=${T}"
      echo "==============================="
      OVERRIDES=(
        "+data=pmri"
        "+unet=monai_unet"
        "+trainer=unet_trainer"
        "++data.split=${T}"
        "++unet.out_channels=1"  # pmri is binary (foreground vs background)
      )
      if [ -n "${DROPOUT}" ]; then OVERRIDES+=("++unet.dropout=${DROPOUT}"); fi
      if [ -n "${BATCH_SIZE}" ]; then OVERRIDES+=("++data.batch_size=${BATCH_SIZE}"); fi
      if [ -n "${LIMIT_BATCHES}" ]; then OVERRIDES+=("++trainer.limit_train_batches=${LIMIT_BATCHES}"); fi
      if [ -n "${MAX_EPOCHS}" ]; then OVERRIDES+=("++trainer.max_epochs=${MAX_EPOCHS}"); fi
      if [ -n "${EXTRA}" ]; then EXTRA_ARR=(${EXTRA}); OVERRIDES+=("${EXTRA_ARR[@]}"); fi
      echo "Hydra overrides: ${OVERRIDES[*]}"
      CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} \
      python train_unet.py "${OVERRIDES[@]}" || { echo "Split ${T} failed"; exit 1; }
    done
    ;;
  mnmv2)
    # Canonical MNMv2 split tags
    DEFAULT_SPLITS=(
      "scanner-symphonytim"
      "pathology-norm-vs-fall-scanners-all"
    )
    if [ -z "${SPLITS:-}" ]; then
      TARGETS=("${DEFAULT_SPLITS[@]}")
    else
      # shellcheck disable=SC2206
      TARGETS=(${SPLITS})
    fi
    echo "Dataset=MNMv2 | splits: ${TARGETS[*]}"
    for T in "${TARGETS[@]}"; do
      echo "==============================="
      echo "Training U-Net | dataset=mnmv2 | split=${T}"
      echo "==============================="
      OVERRIDES=(
        "+data=mnmv2"
        "+unet=monai_unet"
        "+trainer=unet_trainer"
        "++data.split=${T}"
      )
      if [ -n "${DROPOUT}" ]; then OVERRIDES+=("++unet.dropout=${DROPOUT}"); fi
      if [ -n "${BATCH_SIZE}" ]; then OVERRIDES+=("++data.batch_size=${BATCH_SIZE}"); fi
      if [ -n "${LIMIT_BATCHES}" ]; then OVERRIDES+=("++trainer.limit_train_batches=${LIMIT_BATCHES}"); fi
      if [ -n "${MAX_EPOCHS}" ]; then OVERRIDES+=("++trainer.max_epochs=${MAX_EPOCHS}"); fi
      if [ -n "${EXTRA}" ]; then EXTRA_ARR=(${EXTRA}); OVERRIDES+=("${EXTRA_ARR[@]}"); fi
      echo "Hydra overrides: ${OVERRIDES[*]}"
      CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} \
      python train_unet.py "${OVERRIDES[@]}" || { echo "Scenario ${T} failed"; exit 1; }
    done
    ;;
  *)
    echo "Unknown DATASET='${DATASET}'. Use pmri or mnmv2." >&2
    exit 2
    ;;
esac

echo "All ${DATASET} runs completed successfully."
