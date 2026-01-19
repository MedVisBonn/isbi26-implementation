#!/usr/bin/env bash
set -euo pipefail

# Environment overrides
#   CUDA_DEVICE     GPU id (default 0)
#   BATCH_SIZE      Inference batch size
#   NUM_RUNS        Number of repetitions (different seeds)
#   INCLUDE_TRAIN   True|False to include train split in evaluation
#   DATASETS        Space-separated list of datasets to run (default: "mnmv2 pmri")
#   SPLITS          Space-separated list of canonical split tags to run (overrides defaults per dataset)

CUDA_DEVICE=${CUDA_DEVICE:-4}
BATCH_SIZE=${BATCH_SIZE:-15}
NUM_RUNS=${NUM_RUNS:-5}
INCLUDE_TRAIN=${INCLUDE_TRAIN:-False}
DATASETS=${DATASETS:-"pmri"}

for RUN_ID in $(seq 1 $((NUM_RUNS-1))); do
    echo "Running iteration: $RUN_ID"

    for SCORE_TYPE in dice; do
        echo "Running evaluation for score type: $SCORE_TYPE"

        for DATASET in ${DATASETS}; do
            echo "Running evaluation for dataset: $DATASET"

            # Default splits per dataset (canonical hyphenated)
            if [ "$DATASET" = "mnmv2" ]; then
                DEFAULT_SPLITS=(
                    "scanner-symphonytim"
                    "pathology-norm-vs-fall-scanners-all"
                    # "pathology-norm-vs-fall-scanners-all"
                    # "phase-exclusive-scanners-all"
                )
            elif [ "$DATASET" = "pmri" ]; then
                DEFAULT_SPLITS=(
                    "promise12"
                    # "promise12-i2cvb"
                    "threet-to-onepointfivet"
                    # "promise12-nci-isbi"
                )
            else
                echo "Unknown DATASET '$DATASET'" >&2
                exit 2
            fi

            # Resolve splits to iterate
            if [ -n "${SPLITS:-}" ]; then
                # shellcheck disable=SC2206
                TARGET_SPLITS=(${SPLITS})
            else
                TARGET_SPLITS=("${DEFAULT_SPLITS[@]}")
            fi

            for SPLIT in "${TARGET_SPLITS[@]}"; do
                echo "Evaluating dataset=$DATASET split=$SPLIT score=$SCORE_TYPE run=$RUN_ID"

                CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} \
                python eval_score_agreement.py \
                    +data="${DATASET}" \
                    ++data.split="${SPLIT}" \
                    ++data.non_empty_target=True \
                    ++batch_size="${BATCH_SIZE}" \
                    ++run_id="${RUN_ID}" \
                    ++score_type="${SCORE_TYPE}" \
                    ++include_train="${INCLUDE_TRAIN}"

                echo "Completed dataset=$DATASET split=$SPLIT score=$SCORE_TYPE run=$RUN_ID"
            done
        done
        echo "Completed score type $SCORE_TYPE for iteration $RUN_ID"
    done
    echo "Completed iteration $RUN_ID"
done

echo "All evaluations completed!"