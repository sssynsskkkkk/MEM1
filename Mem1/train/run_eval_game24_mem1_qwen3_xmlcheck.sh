#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VAL_FILE="${VAL_FILE:-/workspace/MEM1/data/game24_mem1/test.parquet}"
export VAL_DATA_NUM="${VAL_DATA_NUM:-200}"
export MAX_TURNS="${MAX_TURNS:-12}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-192}"
export MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-128}"
export REQUIRE_REASONING="${REQUIRE_REASONING:-false}"
export ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
export TRAINER_LOGGER="${TRAINER_LOGGER:-['console','wandb']}"
export WAND_PROJECT="${WAND_PROJECT:-MEM1-game24-xmlcheck}"
export WANDB_MODE="${WANDB_MODE:-online}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY is not set. wandb logging may fail or run offline."
fi

RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}"

MODELS=(
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
)

for MODEL_PATH in "${MODELS[@]}"; do
    case "$MODEL_PATH" in
        "Qwen/Qwen3-1.7B")
            MODEL_TAG="qwen3-1p7b"
            export VAL_BATCH_SIZE="${VAL_BATCH_SIZE_1P7B:-8}"
            export ROLLOUT_GPU_UTIL="${ROLLOUT_GPU_UTIL_1P7B:-0.55}"
            ;;
        "Qwen/Qwen3-4B")
            MODEL_TAG="qwen3-4b"
            export VAL_BATCH_SIZE="${VAL_BATCH_SIZE_4B:-8}"
            export ROLLOUT_GPU_UTIL="${ROLLOUT_GPU_UTIL_4B:-0.55}"
            ;;
        "Qwen/Qwen3-8B")
            MODEL_TAG="qwen3-8b"
            export VAL_BATCH_SIZE="${VAL_BATCH_SIZE_8B:-4}"
            export ROLLOUT_GPU_UTIL="${ROLLOUT_GPU_UTIL_8B:-0.50}"
            ;;
        *)
            echo "Unknown model: $MODEL_PATH" >&2
            exit 1
            ;;
    esac

    export MODEL_PATH
    export EXPERIMENT_NAME="game24-mem1-${MODEL_TAG}-xmlcheck-${RUN_TAG}"
    export EVAL_DUMP_DIR="${SCRIPT_DIR}/eval_dumps/${EXPERIMENT_NAME}"

    echo
    echo "============================================================"
    echo "Starting evaluation for ${MODEL_PATH}"
    echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
    echo "EVAL_DUMP_DIR=${EVAL_DUMP_DIR}"
    echo "VAL_DATA_NUM=${VAL_DATA_NUM}, VAL_BATCH_SIZE=${VAL_BATCH_SIZE}, ROLLOUT_GPU_UTIL=${ROLLOUT_GPU_UTIL}"
    echo "TRAINER_LOGGER=${TRAINER_LOGGER}, WAND_PROJECT=${WAND_PROJECT}, WANDB_MODE=${WANDB_MODE}"
    echo "============================================================"
    echo

    bash "${SCRIPT_DIR}/eval_game24_mem1.sh"
done

echo
echo "All evaluations completed."
