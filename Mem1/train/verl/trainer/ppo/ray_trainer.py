# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import re
import json
from collections import defaultdict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
import tqdm
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

import re
from rollout.llm_agent.generation_game24 import GenerationConfig as Game24GenerationConfig
from rollout.llm_agent.generation_game24 import LLMGenerationManager as Game24GenerationManager
from rollout.llm_agent.generation_maze import GenerationConfig as MazeGenerationConfig
from rollout.llm_agent.generation_maze import LLMGenerationManager as MazeGenerationManager
from rollout.llm_agent.generation_think import GenerationConfig as ThinkGenerationConfig
from rollout.llm_agent.generation_think import LLMGenerationManager as ThinkGenerationManager

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['info_mask'] if 'info_mask' in data.batch else data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    # import pdb; pdb.set_trace()
    
    token_level_rewards = token_level_scores - beta * torch.clamp(kld, min=0)

    if (token_level_scores < token_level_rewards).sum().item() > 0:
        import pdb; pdb.set_trace()


    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)
    sequence_true_score = sequence_score - batch.batch['batch_rewards']

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # true score
        'critic/true_score/mean':
            torch.mean(sequence_true_score).detach().item(),
        'critic/true_score/max':
            torch.max(sequence_true_score).detach().item(),
        'critic/true_score/min':
            torch.min(sequence_true_score).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # metrics for actions
    if 'turns_stats' in batch.meta_info:
        metrics['env/number_of_actions/mean'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).mean())
        metrics['env/number_of_actions/max'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).max())
        metrics['env/number_of_actions/min'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).min())
    if 'active_mask' in batch.meta_info:
        metrics['env/finish_ratio'] = 1 - float(np.array(batch.meta_info['active_mask'], dtype=np.int16).mean())
    if 'valid_action_stats' in batch.meta_info:
        metrics['env/number_of_valid_action'] = float(np.array(batch.meta_info['valid_action_stats'], dtype=np.int16).mean())
        metrics['env/ratio_of_valid_action'] = float((np.array(batch.meta_info['valid_action_stats'], dtype=np.int16) / np.array(batch.meta_info['turns_stats'], dtype=np.int16)).mean())
    if 'valid_search_stats' in batch.meta_info:
        metrics['env/number_of_valid_search'] = float(np.array(batch.meta_info['valid_search_stats'], dtype=np.int16).mean())
    if 'success_stats' in batch.meta_info:
        metrics['env/success_rate'] = float(np.array(batch.meta_info['success_stats'], dtype=np.float32).mean())
    if 'invalid_action_counts' in batch.meta_info and 'generated_turns_stats' in batch.meta_info:
        invalid_actions = np.array(batch.meta_info['invalid_action_counts'], dtype=np.float32).sum()
        total_turns = np.array(batch.meta_info['generated_turns_stats'], dtype=np.float32).sum()
        metrics['env/invalid_action_rate'] = float(invalid_actions / total_turns) if total_turns > 0 else 0.0
    if 'format_valid_stats' in batch.meta_info and 'generated_turns_stats' in batch.meta_info:
        compliant = np.array(batch.meta_info['format_valid_stats'], dtype=np.float32).sum()
        total_turns = np.array(batch.meta_info['generated_turns_stats'], dtype=np.float32).sum()
        metrics['env/format_compliance_rate'] = float(compliant / total_turns) if total_turns > 0 else 0.0
    if 'summary_length_sums' in batch.meta_info and 'summary_count_stats' in batch.meta_info:
        total_summary_len = np.array(batch.meta_info['summary_length_sums'], dtype=np.float32).sum()
        total_summary_count = np.array(batch.meta_info['summary_count_stats'], dtype=np.float32).sum()
        metrics['env/avg_summary_len'] = float(total_summary_len / total_summary_count) if total_summary_count > 0 else 0.0
    if 'turns_stats' in batch.meta_info:
        metrics['env/avg_steps'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.float32).mean())


    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor', 'rollout']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


GAME24_SOURCES = {'game24', 'gameof24'}
MAZE_SOURCES = {'maze', 'maze_mem1'}
ROLLOUT_METRIC_KEYS = (
    'success_stats',
    'invalid_action_counts',
    'blocked_move_counts',
    'turns_stats',
    'retry_counts',
    'summary_length_sums',
    'summary_count_stats',
    'format_valid_stats',
    'generated_turns_stats',
    'unique_cells_visited_stats',
    'solved_path_ratio_sums',
    'solved_path_ratio_counts',
)


def _extract_first_data_source(batch_like) -> str:
    data_source = None
    if hasattr(batch_like, 'non_tensor_batch'):
        data_source = batch_like.non_tensor_batch.get('data_source', None)
    elif isinstance(batch_like, dict):
        data_source = batch_like.get('data_source', None)

    if isinstance(data_source, np.ndarray):
        if data_source.size == 0:
            return 'unknown'
        return str(data_source[0])
    if isinstance(data_source, (list, tuple)):
        if not data_source:
            return 'unknown'
        return str(data_source[0])
    if data_source is None:
        return 'unknown'
    return str(data_source)


def _is_game24_source(data_source: str) -> bool:
    return str(data_source).lower() in GAME24_SOURCES


def _is_maze_source(data_source: str) -> bool:
    return str(data_source).lower() in MAZE_SOURCES


def _extract_ground_truths(batch_like):
    reward_models = batch_like.non_tensor_batch.get('reward_model', None)
    if reward_models is None:
        raise ValueError('reward_model ground truth is required for Game24 rollout')

    if isinstance(reward_models, np.ndarray):
        return [item['ground_truth'] for item in reward_models.tolist()]
    if isinstance(reward_models, list):
        return [item['ground_truth'] for item in reward_models]
    raise TypeError(f'Unsupported reward_model container: {type(reward_models)}')


def _append_rollout_metrics(store: Dict[str, list], meta_info: Dict):
    for key in ROLLOUT_METRIC_KEYS:
        if key in meta_info:
            store.setdefault(key, []).append(np.asarray(meta_info[key], dtype=np.float32))


def _aggregate_rollout_metrics(metric_dict: Dict[str, float], data_source: str, metric_store: Dict[str, list]):
    if 'success_stats' in metric_store:
        metric_dict[f'val/success_rate/{data_source}'] = float(np.concatenate(metric_store['success_stats']).mean())
    if 'invalid_action_counts' in metric_store and 'generated_turns_stats' in metric_store:
        invalid_actions = np.concatenate(metric_store['invalid_action_counts']).sum()
        total_turns = np.concatenate(metric_store['generated_turns_stats']).sum()
        metric_dict[f'val/invalid_action_rate/{data_source}'] = float(invalid_actions / total_turns) if total_turns > 0 else 0.0
    if 'blocked_move_counts' in metric_store and 'generated_turns_stats' in metric_store:
        blocked_moves = np.concatenate(metric_store['blocked_move_counts']).sum()
        total_turns = np.concatenate(metric_store['generated_turns_stats']).sum()
        blocked_move_rate = float(blocked_moves / total_turns) if total_turns > 0 else 0.0
        metric_dict[f'val/blocked_move_rate/{data_source}'] = blocked_move_rate
        metric_dict[f'val/invalid_move_rate/{data_source}'] = blocked_move_rate
    if 'turns_stats' in metric_store:
        metric_dict[f'val/avg_steps/{data_source}'] = float(np.concatenate(metric_store['turns_stats']).mean())
    if 'summary_length_sums' in metric_store and 'summary_count_stats' in metric_store:
        total_len = np.concatenate(metric_store['summary_length_sums']).sum()
        total_count = np.concatenate(metric_store['summary_count_stats']).sum()
        metric_dict[f'val/avg_summary_len/{data_source}'] = float(total_len / total_count) if total_count > 0 else 0.0
    if 'format_valid_stats' in metric_store and 'generated_turns_stats' in metric_store:
        compliant = np.concatenate(metric_store['format_valid_stats']).sum()
        total_turns = np.concatenate(metric_store['generated_turns_stats']).sum()
        metric_dict[f'val/format_compliance_rate/{data_source}'] = float(compliant / total_turns) if total_turns > 0 else 0.0
    if 'unique_cells_visited_stats' in metric_store:
        metric_dict[f'val/avg_unique_cells_visited/{data_source}'] = float(
            np.concatenate(metric_store['unique_cells_visited_stats']).mean()
        )
    if 'solved_path_ratio_sums' in metric_store and 'solved_path_ratio_counts' in metric_store:
        ratio_sum = np.concatenate(metric_store['solved_path_ratio_sums']).sum()
        ratio_count = np.concatenate(metric_store['solved_path_ratio_counts']).sum()
        metric_dict[f'val/solved_step_ratio/{data_source}'] = float(ratio_sum / ratio_count) if ratio_count > 0 else 0.0


def _decode_token_rows(tokenizer, token_tensor: torch.Tensor) -> list[str]:
    decoded = []
    pad_token_id = tokenizer.pad_token_id
    for row in token_tensor:
        valid = row[row != pad_token_id]
        decoded.append(tokenizer.decode(valid, skip_special_tokens=True))
    return decoded


def _build_eval_dump_rows(tokenizer, batch: DataProto, reward_tensor: torch.Tensor, meta_info: Dict, global_step: int):
    prompt_texts = _decode_token_rows(tokenizer, batch.batch['prompts']) if 'prompts' in batch.batch else []
    trajectory_texts = _decode_token_rows(tokenizer, batch.batch['responses']) if 'responses' in batch.batch else []
    sample_rewards = reward_tensor.sum(-1).detach().cpu().tolist()
    data_sources = batch.non_tensor_batch.get('data_source', ['unknown'] * len(sample_rewards))
    extra_infos = batch.non_tensor_batch.get('extra_info', [None] * len(sample_rewards))

    if isinstance(data_sources, np.ndarray):
        data_sources = data_sources.tolist()
    if isinstance(extra_infos, np.ndarray):
        extra_infos = extra_infos.tolist()

    success_stats = meta_info.get('success_stats', [None] * len(sample_rewards))
    invalid_action_counts = meta_info.get('invalid_action_counts', [None] * len(sample_rewards))
    turns_stats = meta_info.get('turns_stats', [None] * len(sample_rewards))
    retry_counts = meta_info.get('retry_counts', [None] * len(sample_rewards))
    format_valid_stats = meta_info.get('format_valid_stats', [None] * len(sample_rewards))
    generated_turns_stats = meta_info.get('generated_turns_stats', [None] * len(sample_rewards))
    summary_length_sums = meta_info.get('summary_length_sums', [None] * len(sample_rewards))
    summary_count_stats = meta_info.get('summary_count_stats', [None] * len(sample_rewards))
    blocked_move_counts = meta_info.get('blocked_move_counts', [None] * len(sample_rewards))
    unique_cells_visited_stats = meta_info.get('unique_cells_visited_stats', [None] * len(sample_rewards))
    solved_path_ratio_sums = meta_info.get('solved_path_ratio_sums', [None] * len(sample_rewards))
    solved_path_ratio_counts = meta_info.get('solved_path_ratio_counts', [None] * len(sample_rewards))

    rows = []
    for idx, reward in enumerate(sample_rewards):
        summary_count = summary_count_stats[idx] if idx < len(summary_count_stats) else None
        summary_len_sum = summary_length_sums[idx] if idx < len(summary_length_sums) else None
        avg_summary_len = None
        if summary_count not in (None, 0) and summary_len_sum is not None:
            avg_summary_len = float(summary_len_sum) / float(summary_count)
        solved_ratio = None
        solved_ratio_count = solved_path_ratio_counts[idx] if idx < len(solved_path_ratio_counts) else None
        solved_ratio_sum = solved_path_ratio_sums[idx] if idx < len(solved_path_ratio_sums) else None
        if solved_ratio_count not in (None, 0) and solved_ratio_sum is not None:
            solved_ratio = float(solved_ratio_sum) / float(solved_ratio_count)

        rows.append({
            'global_step': global_step,
            'sample_index_in_batch': idx,
            'data_source': str(data_sources[idx]) if idx < len(data_sources) else 'unknown',
            'extra_info': extra_infos[idx] if idx < len(extra_infos) else None,
            'prompt_text': prompt_texts[idx] if idx < len(prompt_texts) else '',
            'trajectory_text': trajectory_texts[idx] if idx < len(trajectory_texts) else '',
            'reward': float(reward),
            'success': int(success_stats[idx]) if idx < len(success_stats) and success_stats[idx] is not None else None,
            'invalid_action_count': int(invalid_action_counts[idx]) if idx < len(invalid_action_counts) and invalid_action_counts[idx] is not None else None,
            'turn_count': int(turns_stats[idx]) if idx < len(turns_stats) and turns_stats[idx] is not None else None,
            'retry_count': int(retry_counts[idx]) if idx < len(retry_counts) and retry_counts[idx] is not None else None,
            'format_valid_turns': int(format_valid_stats[idx]) if idx < len(format_valid_stats) and format_valid_stats[idx] is not None else None,
            'generated_turns': int(generated_turns_stats[idx]) if idx < len(generated_turns_stats) and generated_turns_stats[idx] is not None else None,
            'avg_summary_len': avg_summary_len,
            'blocked_move_count': int(blocked_move_counts[idx]) if idx < len(blocked_move_counts) and blocked_move_counts[idx] is not None else None,
            'unique_cells_visited': int(unique_cells_visited_stats[idx]) if idx < len(unique_cells_visited_stats) and unique_cells_visited_stats[idx] is not None else None,
            'solved_step_ratio': solved_ratio,
        })
    return rows


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:

            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()
        self._init_logger()
    
    def _init_logger(self):
        from verl.utils.tracking import Tracking
        self.logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        if self.config.data.train_data_num is not None:
            if self.config.data.train_data_num > len(self.train_dataset.dataframe):
                print(f"[WARNING] training dataset size is smaller than desired size. Using the dataset as the original size {len(self.train_dataset.dataframe)}")
            else:
                self.train_dataset.dataframe = self.train_dataset.dataframe.sample(self.config.data.train_data_num, random_state=42)
        print(f"filtered training dataset size: {len(self.train_dataset.dataframe)}")

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=self.config.data.shuffle_train_dataloader,
                                           drop_last=True,
                                           collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        if self.config.data.val_data_num is not None:
            if self.config.data.val_data_num > len(self.val_dataset.dataframe):
                print(f"[WARNING] validation dataset size is smaller than desired size. Using the dataset as the original size {len(self.val_dataset.dataframe)}")
            else:
                self.val_dataset.dataframe = self.val_dataset.dataframe.sample(self.config.data.val_data_num, random_state=42)
        print(f"filtered validation dataset size: {len(self.val_dataset.dataframe)}")

        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.config.data.val_batch_size,
                                         shuffle=False,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')
        
        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _build_generation_manager(self, data_source: str, is_validation: bool = False):
        if _is_maze_source(data_source):
            maze_cfg = self.config.get('maze', {})
            gen_config = MazeGenerationConfig(
                max_turns=self.config.max_turns,
                max_start_length=self.config.data.max_start_length,
                max_prompt_length=self.config.data.max_prompt_length,
                max_response_length=self.config.data.max_response_length,
                max_obs_length=self.config.data.max_obs_length,
                num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
                require_reasoning=maze_cfg.get('require_reasoning', False),
                prepend_no_think=self.config.algorithm.no_think_rl,
                format_reward=maze_cfg.get('format_reward', 0.0),
                summary_present_reward=maze_cfg.get('summary_present_reward', 0.0),
                invalid_action_penalty=maze_cfg.get('invalid_action_penalty', 0.0),
                unparsable_output_penalty=maze_cfg.get('unparsable_output_penalty', 0.0),
            )
            return MazeGenerationManager(
                tokenizer=self.tokenizer,
                actor_rollout_wg=self.actor_rollout_wg,
                config=gen_config,
                is_validation=is_validation,
            )

        if _is_game24_source(data_source):
            game24_cfg = self.config.get('game24', {})
            gen_config = Game24GenerationConfig(
                max_turns=self.config.max_turns,
                max_start_length=self.config.data.max_start_length,
                max_prompt_length=self.config.data.max_prompt_length,
                max_response_length=self.config.data.max_response_length,
                max_obs_length=self.config.data.max_obs_length,
                num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
                require_reasoning=game24_cfg.get('require_reasoning', True),
                prepend_no_think=self.config.algorithm.no_think_rl,
                format_reward=game24_cfg.get('format_reward', 0.0),
                summary_present_reward=game24_cfg.get('summary_present_reward', 0.0),
                valid_action_reward=game24_cfg.get('valid_action_reward', 0.0),
                invalid_action_penalty=game24_cfg.get('invalid_action_penalty', -0.25),
                step_penalty=game24_cfg.get('step_penalty', 0.0),
                unparsable_output_penalty=game24_cfg.get('unparsable_output_penalty', -0.05),
                value_tolerance=game24_cfg.get('value_tolerance', 1e-5),
            )
            return Game24GenerationManager(
                tokenizer=self.tokenizer,
                actor_rollout_wg=self.actor_rollout_wg,
                config=gen_config,
                is_validation=is_validation,
            )

        gen_config = ThinkGenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url=self.config.retriever.url,
            topk=self.config.retriever.topk,
        )
        return ThinkGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation=is_validation,
        )

    def _run_generation_loop(self, generation_manager, gen_batch: DataProto, batch_like, timing_raw=None):
        first_input_ids = gen_batch.batch['input_ids'][:, -generation_manager.config.max_start_length:].clone().long()
        if timing_raw is not None:
            generation_manager.timing_raw = timing_raw

        kwargs = {}
        if getattr(generation_manager, 'requires_ground_truths', False):
            kwargs['ground_truths'] = _extract_ground_truths(batch_like)

        return generation_manager.run_llm_loop(
            gen_batch=gen_batch,
            initial_input_ids=first_input_ids,
            **kwargs,
        )

    def _validate(self):
        """
        The training loop of PPO with global metric computation.
        Accumulates metrics across all batches before computing final statistics.
        """
        import torch
        reward_tensor_lst = []
        data_source_lst = []
        generation_managers = {}
        rollout_metric_store = {}
        eval_dump_dir = self.config.trainer.get('eval_dump_dir', None)
        eval_dump_rows = []

        if not self.config.do_search:
            for test_data in self.val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)

                # we only do validation on rule-based rm
                if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                    return {}

                test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }

                # pad to be divisible by dp_size
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                # unpad
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
                print('validation generation end')

                test_batch = test_batch.union(test_output_gen_batch)

                # evaluate using reward_function
                # for certain reward function (e.g. sandbox), the generation can overlap with reward
                reward_tensor = self.val_reward_fn(test_batch)

                reward_tensor_lst.append(reward_tensor)
                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
        else:
            for batch_dict in tqdm.tqdm(self.val_dataloader):
                timing_raw = {}
                test_batch: DataProto = DataProto.from_single_dict(batch_dict)
                # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)
                data_source = _extract_first_data_source(test_batch)
                generation_manager = generation_managers.get(data_source)
                if generation_manager is None:
                    generation_manager = self._build_generation_manager(data_source, is_validation=True)
                    generation_managers[data_source] = generation_manager
                
                test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }
                with _timer('step', timing_raw):
                    with _timer('gen', timing_raw):
                        final_gen_batch_output = self._run_generation_loop(
                            generation_manager,
                            test_gen_batch,
                            test_batch,
                            timing_raw=timing_raw,
                        )
                    
                    test_batch = test_batch.union(final_gen_batch_output)
                    test_batch.batch['batch_rewards'] = torch.tensor(final_gen_batch_output.meta_info['batch_rewards'])

                    
                    for key in test_batch.batch.keys():
                        if key not in ["attention_mask_4d", "batch_rewards"]:
                            test_batch.batch[key] = test_batch.batch[key].long()
                    
                    # evaluate using reward_function
                    # for certain reward function (e.g. sandbox), the generation can overlap with reward
                    reward_tensor = self.val_reward_fn(test_batch)

                    reward_tensor_lst.append(reward_tensor)
                    data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
                    source_store = rollout_metric_store.setdefault(data_source, {})
                    _append_rollout_metrics(source_store, final_gen_batch_output.meta_info)
                    if eval_dump_dir:
                        eval_dump_rows.extend(
                            _build_eval_dump_rows(
                                tokenizer=self.tokenizer,
                                batch=test_batch,
                                reward_tensor=reward_tensor,
                                meta_info=final_gen_batch_output.meta_info,
                                global_step=self.global_steps,
                            )
                        )

        reward_tensor = torch.cat([rw.sum(-1) for rw in reward_tensor_lst], dim=0).cpu()  # (batch_size,)
        # reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)
            if data_source in rollout_metric_store:
                _aggregate_rollout_metrics(metric_dict, data_source, rollout_metric_store[data_source])

        if eval_dump_dir and eval_dump_rows:
            os.makedirs(eval_dump_dir, exist_ok=True)
            dump_path = os.path.join(eval_dump_dir, f'val_step_{self.global_steps}.jsonl')
            with open(dump_path, 'w', encoding='utf-8') as handle:
                for row in eval_dump_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            metric_dict['val/dump_rows'] = len(eval_dump_rows)

        return metric_dict


    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
            
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        logger = self.logger
        self.global_steps = 0
        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()

            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1
        generation_managers = {}

        # start training loop
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                print(f'epoch {epoch}, step {self.global_steps}')
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                ####################
                # original code here

                with _timer('step', timing_raw):
                    if not self.config.do_search:
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                                dtype=object)
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                ####################
                # Below is aLL about agents - the "LLM + forloop"
                ####################
                # with _timer('step', timing_raw):
                    else:
                        data_source = _extract_first_data_source(batch)
                        generation_manager = generation_managers.get(data_source)
                        if generation_manager is None:
                            generation_manager = self._build_generation_manager(data_source, is_validation=False)
                            generation_managers[data_source] = generation_manager

                        with _timer('gen', timing_raw):
                            final_gen_batch_output = self._run_generation_loop(
                                generation_manager,
                                gen_batch,
                                batch,
                                timing_raw=timing_raw,
                            )

                        # final_gen_batch_output.batch.apply(lambda x: x.long(), inplace=True)
                        for key in final_gen_batch_output.batch.keys():
                            if key not in ["attention_mask_4d", "batch_rewards"]:
                                final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()

                        with torch.no_grad():
                            output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                            final_gen_batch_output = final_gen_batch_output.union(output)

                        # batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                        #                                         dtype=object)
                        batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()
                                            
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(final_gen_batch_output)
                        batch.batch['batch_rewards'] = torch.tensor(final_gen_batch_output.meta_info['batch_rewards'])


                    ####################
                    ####################

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # batch.batch.apply(lambda x, key: x.long() if key != "old_log_probs" else x, inplace=True, key=True)
                    for key in batch.batch.keys():
                        if key not in ['old_log_probs', 'attention_mask_4d', 'batch_rewards']:
                            batch.batch[key] = batch.batch[key].long()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            if self.config.do_search and self.config.actor_rollout_ref.actor.state_masking:
                                batch, metrics = self._create_loss_mask(batch, metrics)
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return
    
    def _create_loss_mask(self, batch, metrics):
        """Create loss mask for state tokens."""
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]
        
        loss_mask = batch.batch['info_mask'][:, -response_length:]
        batch.batch['loss_mask'] = loss_mask

        metrics.update({
            'state_tokens/total': loss_mask.sum().item(),
            'state_tokens/coverage': (loss_mask.sum() / response_mask.sum()).item(),
        })
        
        return batch, metrics
