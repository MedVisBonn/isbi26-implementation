#!/usr/bin/env bash
set -euo pipefail

# Environment overrides
#   CUDA_DEVICE     GPU id (default 0)
#   NUM_RUNS        Number of repetitions (different seeds)
#   INCLUDE_TRAIN   True|False to include train split in evaluation
#   DATASETS        Space-separated list of datasets to run (default: "mnmv2 pmri")
#   SPLITS          Space-separated list of canonical split tags to run (overrides defaults per dataset)
#   SWIVEL_MATCH    Module name substring for swivel discovery (default: "adn.N")
#   SWIVEL_INDICES  Space-separated list of indices to select from matched modules (default: "17")
#   SIGMA_ALGORITHM Covariance estimation method: ledoit_wolf, diagonal, default (default: "ledoit_wolf")

CUDA_DEVICE=${CUDA_DEVICE:-5}
NUM_RUNS=${NUM_RUNS:-6}
INCLUDE_TRAIN=${INCLUDE_TRAIN:-False}
DATASETS=${DATASETS:-"mnmv2"}
SWIVEL_MATCH=${SWIVEL_MATCH:-"adn.N"}
SWIVEL_INDICES=${SWIVEL_INDICES:-"17"}
SIGMA_ALGORITHM=${SIGMA_ALGORITHM:-"ledoit_wolf"}

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
                )
            elif [ "$DATASET" = "pmri" ]; then
                DEFAULT_SPLITS=(
                    "promise12"
                    "threet-to-onepointfivet"
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

                # Convert SWIVEL_INDICES to Python list format
                # e.g., "17" -> [17], "17 23" -> [17,23]
                PYTHON_INDICES="[$(echo ${SWIVEL_INDICES} | sed 's/ /,/g')]"

                CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} \
                python eval_mahalanobis.py \
                    +data="${DATASET}" \
                    ++data.split="${SPLIT}" \
                    ++data.non_empty_target=True \
                    ++run_id="${RUN_ID}" \
                    ++score_type="${SCORE_TYPE}" \
                    ++include_train="${INCLUDE_TRAIN}" \
                    ++discover_swivel=True \
                    ++swivel_match="${SWIVEL_MATCH}" \
                    ++swivel_indices="${PYTHON_INDICES}" \
                    ++sigma_algorithm="${SIGMA_ALGORITHM}"

                echo "Completed dataset=$DATASET split=$SPLIT score=$SCORE_TYPE run=$RUN_ID"
            done
        done
        echo "Completed score type $SCORE_TYPE for iteration $RUN_ID"
    done
    echo "Completed iteration $RUN_ID"
done

echo "All evaluations completed!"
