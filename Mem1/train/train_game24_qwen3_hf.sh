#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
export TRAIN_FILE="${TRAIN_FILE:-/workspace/MEM1/data/game24/train.parquet}"
export VAL_FILE="${VAL_FILE:-/workspace/MEM1/data/game24/test.parquet}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-game24-qwen3-hf-ppo}"
export WAND_PROJECT="${WAND_PROJECT:-MEM1}"

export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-game24}"
mkdir -p "$RAY_TMPDIR"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.9}"

cd /workspace/MEM1/Mem1/train

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-64}" \
    data.val_batch_size="${VAL_BATCH_SIZE:-64}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-1024}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-256}" \
    data.max_start_length="${MAX_START_LENGTH:-1024}" \
    data.max_obs_length="${MAX_OBS_LENGTH:-0}" \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-6}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="${ACTOR_WARMUP_RATIO:-0.05}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-16}" \
    actor_rollout_ref.actor.ppo_micro_batch_size="${PPO_MICRO_BATCH_SIZE:-2}" \
    actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD:-false}" \
    actor_rollout_ref.actor.fsdp_config.grad_offload="${ACTOR_GRAD_OFFLOAD:-false}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD:-false}" \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size="${ROLLOUT_LOGPROB_MB:-2}" \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE:-1.0}" \
    actor_rollout_ref.rollout.top_k=0 \
    actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P:-0.95}" \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.n=1 \
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
    algorithm.no_think_rl="${NO_THINK_RL:-false}" \
    trainer.critic_warmup=0 \
    trainer.logger="${TRAINER_LOGGER:-['console']}" \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node="${NUM_GPUS:-1}" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ:-50}" \
    trainer.test_freq="${TEST_FREQ:-50}" \
    trainer.project_name="$WAND_PROJECT" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.total_epochs="${TOTAL_EPOCHS:-3}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-500}" \
    trainer.default_local_dir="verl_checkpoints/$EXPERIMENT_NAME" \
    max_turns=1 \
    do_search=false \
    2>&1 | tee "$EXPERIMENT_NAME.log"
