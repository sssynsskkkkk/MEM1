#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
export TRAIN_FILE="${TRAIN_FILE:-$PROJECT_ROOT/data/game24_mem1/train.parquet}"
export VAL_FILE="${VAL_FILE:-$PROJECT_ROOT/data/game24_mem1/test.parquet}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-game24-mem1-ppo}"
export WAND_PROJECT="${WAND_PROJECT:-MEM1}"

export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-game24-mem1}"
mkdir -p "$RAY_TMPDIR"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.9}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
export USE_REFERENCE_POLICY="${USE_REFERENCE_POLICY:-true}"
export REQUIRE_REASONING="${REQUIRE_REASONING:-false}"

cd "$SCRIPT_DIR"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_data_num="${TRAIN_DATA_NUM:-null}" \
    data.val_data_num="${VAL_DATA_NUM:-null}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-32}" \
    data.val_batch_size="${VAL_BATCH_SIZE:-32}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-1024}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-192}" \
    data.max_start_length="${MAX_START_LENGTH:-1024}" \
    data.max_obs_length="${MAX_OBS_LENGTH:-128}" \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator="${ADV_ESTIMATOR:-gae}" \
    +algorithm.use_reference_policy="${USE_REFERENCE_POLICY}" \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-6}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="${ACTOR_WARMUP_RATIO:-0.05}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-16}" \
    actor_rollout_ref.actor.ppo_micro_batch_size="${PPO_MICRO_BATCH_SIZE:-2}" \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD:-false}" \
    actor_rollout_ref.actor.fsdp_config.grad_offload="${ACTOR_GRAD_OFFLOAD:-false}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD:-false}" \
    actor_rollout_ref.rollout.name="$ROLLOUT_NAME" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_UTIL:-0.55}" \
    +actor_rollout_ref.rollout.micro_batch_size="${ROLLOUT_MICRO_BATCH_SIZE:-1}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size="${ROLLOUT_LOGPROB_MB:-2}" \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE:-1.0}" \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P:-0.95}" \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.n=1 \
    +actor_rollout_ref.rollout.stop="['</action>']" \
    actor_rollout_ref.ref.log_prob_micro_batch_size="${REF_LOGPROB_MB:-2}" \
    critic.optim.lr="${CRITIC_LR:-1e-5}" \
    critic.model.path="$BASE_MODEL" \
    critic.model.enable_gradient_checkpointing=true \
    critic.model.use_remove_padding=False \
    critic.ppo_micro_batch_size="${CRITIC_PPO_MICRO_BATCH_SIZE:-2}" \
    critic.model.fsdp_config.param_offload="${CRITIC_PARAM_OFFLOAD:-false}" \
    critic.model.fsdp_config.grad_offload="${CRITIC_GRAD_OFFLOAD:-false}" \
    critic.model.fsdp_config.optimizer_offload="${CRITIC_OPTIMIZER_OFFLOAD:-false}" \
    algorithm.kl_ctrl.kl_coef="${KL_COEF:-0.001}" \
    trainer.critic_warmup=0 \
    trainer.logger="${TRAINER_LOGGER:-['console']}" \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node="${NUM_GPUS:-1}" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ:-50}" \
    trainer.test_freq="${TEST_FREQ:-50}" \
    trainer.project_name="$WAND_PROJECT" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-500}" \
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
