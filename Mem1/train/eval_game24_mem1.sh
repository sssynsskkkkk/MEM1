#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
export VAL_FILE="${VAL_FILE:-$PROJECT_ROOT/data/game24_mem1/test.parquet}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-game24-mem1-eval}"
export EVAL_DUMP_DIR="${EVAL_DUMP_DIR:-$SCRIPT_DIR/eval_dumps/$EXPERIMENT_NAME}"

export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-game24-mem1-eval}"
mkdir -p "$RAY_TMPDIR"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.9}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
export USE_REFERENCE_POLICY="${USE_REFERENCE_POLICY:-true}"
export REQUIRE_REASONING="${REQUIRE_REASONING:-false}"

cd "$SCRIPT_DIR"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files="$VAL_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_data_num=1 \
    data.val_data_num="${VAL_DATA_NUM:-null}" \
    data.train_batch_size=1 \
    data.val_batch_size="${VAL_BATCH_SIZE:-32}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-1024}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-192}" \
    data.max_start_length="${MAX_START_LENGTH:-1024}" \
    data.max_obs_length="${MAX_OBS_LENGTH:-128}" \
    data.shuffle_train_dataloader=False \
    algorithm.adv_estimator="${ADV_ESTIMATOR:-gae}" \
    +algorithm.use_reference_policy="${USE_REFERENCE_POLICY}" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=false \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.rollout.name="$ROLLOUT_NAME" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_UTIL:-0.55}" \
    +actor_rollout_ref.rollout.micro_batch_size="${ROLLOUT_MICRO_BATCH_SIZE:-1}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size="${ROLLOUT_LOGPROB_MB:-2}" \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE:-0.0}" \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.n=1 \
    +actor_rollout_ref.rollout.stop="['</action>']" \
    actor_rollout_ref.ref.log_prob_micro_batch_size="${REF_LOGPROB_MB:-2}" \
    critic.optim.lr=1e-5 \
    critic.model.path="$MODEL_PATH" \
    critic.model.enable_gradient_checkpointing=false \
    critic.model.use_remove_padding=False \
    critic.ppo_micro_batch_size=1 \
    algorithm.kl_ctrl.kl_coef="${KL_COEF:-0.001}" \
    trainer.critic_warmup=0 \
    trainer.logger="${TRAINER_LOGGER:-['console']}" \
    +trainer.val_only=true \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node="${NUM_GPUS:-1}" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.project_name="${WAND_PROJECT:-MEM1}" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    +trainer.eval_dump_dir="$EVAL_DUMP_DIR" \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.default_local_dir="verl_checkpoints/$EXPERIMENT_NAME" \
    max_turns="${MAX_TURNS:-12}" \
    do_search=true \
    +game24.require_reasoning="${REQUIRE_REASONING}" \
    +game24.format_reward="${FORMAT_REWARD:-0.0}" \
    +game24.summary_present_reward="${SUMMARY_PRESENT_REWARD:-0.0}" \
    +game24.valid_action_reward="${VALID_ACTION_REWARD:-0.0}" \
    +game24.invalid_action_penalty="${INVALID_ACTION_PENALTY:-0.0}" \
    +game24.step_penalty="${STEP_PENALTY:-0.0}" \
    +game24.unparsable_output_penalty="${UNPARSABLE_OUTPUT_PENALTY:-0.0}" \
    +game24.value_tolerance="${VALUE_TOLERANCE:-1e-5}" \
    2>&1 | tee "$EXPERIMENT_NAME.log"
