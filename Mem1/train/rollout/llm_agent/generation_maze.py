from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from verl import DataProto

from rollout.env import MazeEnvManager

from .attn_mask_utils import compose_final_output
from .game24_utils import parse_model_output
from .maze_utils import compose_rollout_input, extract_maze_spec
from .tensor_helper import TensorConfig, TensorHelper


@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    require_reasoning: bool = False
    prepend_no_think: bool = False
    format_reward: float = 0.0
    summary_present_reward: float = 0.0
    invalid_action_penalty: float = 0.0
    unparsable_output_penalty: float = 0.0


class LLMGenerationManager:
    requires_ground_truths = True

    def __init__(self, tokenizer, actor_rollout_wg, config: GenerationConfig, is_validation: bool = False):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        self.tensor_fn = TensorHelper(
            TensorConfig(
                pad_token_id=tokenizer.pad_token_id,
                max_prompt_length=config.max_prompt_length,
                max_obs_length=config.max_obs_length,
                max_start_length=config.max_start_length,
            )
        )

    def _batch_tokenize(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, 0), dtype=torch.long)
        return self.tokenizer(
            texts,
            add_special_tokens=False,
            return_tensors="pt",
            padding="longest",
        )["input_ids"]

    def _postprocess_responses(self, responses: torch.Tensor):
        response_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        trimmed_texts = []
        for text in response_texts:
            if "</action>" in text:
                trimmed_texts.append(text.split("</action>", 1)[0] + "</action>")
            else:
                trimmed_texts.append(text.strip())
        return self._batch_tokenize(trimmed_texts), trimmed_texts

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        if not next_obs or not any(next_obs):
            return torch.empty((len(next_obs), 0), dtype=torch.long)

        next_obs_ids = self.tokenizer(
            next_obs,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            next_obs_ids = next_obs_ids[:, : self.config.max_obs_length]
        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, next_obs_ids: Optional[torch.Tensor] = None):
        if next_obs_ids is not None:
            new_input_ids = self.tensor_fn.concatenate_with_padding(
                [rollings.batch["input_ids"], cur_responses, next_obs_ids]
            )
        else:
            new_input_ids = self.tensor_fn.concatenate_with_padding([rollings.batch["input_ids"], cur_responses])

        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict(
            {
                "input_ids": new_input_ids[:, -max_len:],
                "position_ids": new_position_ids[:, -max_len:],
                "attention_mask": new_attention_mask[:, -max_len:],
            }
        )
        new_rollings.meta_info.update(rollings.meta_info)
        return new_rollings

    def _info_masked_concatenate_with_padding(
        self,
        prompt: torch.Tensor,
        prompt_with_mask: torch.Tensor,
        response: torch.Tensor,
        info: torch.Tensor = None,
        pad_to_left: bool = True,
    ):
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device)
            tensors_with_mask.append(info_mask)

        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)
        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, cur_responses: torch.Tensor, next_obs_ids: torch.Tensor = None):
        if next_obs_ids is not None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                right_side["responses"],
                right_side["responses_with_info_mask"],
                cur_responses,
                next_obs_ids,
                pad_to_left=False,
            )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                right_side["responses"],
                right_side["responses_with_info_mask"],
                cur_responses,
                pad_to_left=False,
            )

        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        return {
            "responses": responses[:, :max_len],
            "responses_with_info_mask": responses_with_info_mask[:, :max_len],
        }

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        batch_size = active_batch.batch["input_ids"].shape[0]
        remainder = batch_size % num_gpus

        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()

        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        padding_size = num_gpus - remainder
        padded_batch = {}
        for key, value in active_batch.batch.items():
            pad_sequence = value[0:1].repeat(padding_size, *[1] * (len(value.shape) - 1))
            padded_batch[key] = torch.cat([value, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        padded_active_batch.meta_info.update(active_batch.meta_info)
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        trimmed_batch = {key: value[:-padding_size] for key, value in padded_output.batch.items()}

        if hasattr(padded_output, "meta_info") and padded_output.meta_info:
            trimmed_meta = {}
            for key, value in padded_output.meta_info.items():
                if isinstance(value, torch.Tensor):
                    trimmed_meta[key] = value[:-padding_size]
                else:
                    trimmed_meta[key] = value
            padded_output.meta_info = trimmed_meta

        padded_output.batch = trimmed_batch
        return padded_output

    def _token_count(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _pad_token_list(self, token_list: List[torch.Tensor]) -> torch.Tensor:
        max_len = max((tokens.shape[0] for tokens in token_list), default=0)
        padded = torch.full(
            (len(token_list), max_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        for idx, tokens in enumerate(token_list):
            if tokens.shape[0] > 0:
                padded[idx, : tokens.shape[0]] = tokens
        return padded

    def _valid_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return token_ids[token_ids != self.tokenizer.pad_token_id]

    def _compose_final_output(self, reconstruction_list: List[Dict], meta_info: Dict):
        final_output = compose_final_output(reconstruction_list, self.tokenizer.pad_token_id)
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        return final_output

    def _maybe_prepend_no_think(self, prompt_text: str) -> str:
        if not self.config.prepend_no_think:
            return prompt_text
        stripped = prompt_text.lstrip()
        if stripped.startswith("/no_think"):
            return prompt_text
        return f"/no_think\n{stripped}"

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor, ground_truths: List[Dict]):
        if ground_truths is None:
            raise ValueError("ground_truths are required for maze rollout")

        batch_size = gen_batch.batch["input_ids"].shape[0]
        original_meta_info = dict(gen_batch.meta_info)
        original_right_side = {
            "responses": initial_input_ids[:, []],
            "responses_with_info_mask": initial_input_ids[:, []],
        }

        specs = [extract_maze_spec(item) for item in ground_truths]
        env_manager = MazeEnvManager()
        pending_observation_blocks = env_manager.reset(range(batch_size), specs)

        base_prompt_texts = []
        prompt_texts_changed = False
        for idx in range(batch_size):
            row = initial_input_ids[idx]
            mask = row != self.tokenizer.pad_token_id
            decoded_prompt = self.tokenizer.decode(row[mask], skip_special_tokens=True)
            adjusted_prompt = self._maybe_prepend_no_think(decoded_prompt)
            if adjusted_prompt != decoded_prompt:
                prompt_texts_changed = True
            base_prompt_texts.append(adjusted_prompt)

        if prompt_texts_changed:
            initial_input_ids = self._batch_tokenize(base_prompt_texts)
            if initial_input_ids.shape[1] > self.config.max_prompt_length:
                initial_input_ids = initial_input_ids[:, -self.config.max_prompt_length :]
            initial_attention_mask = self.tensor_fn.create_attention_mask(initial_input_ids)
            initial_position_ids = self.tensor_fn.create_position_ids(initial_attention_mask)
            gen_batch = DataProto.from_dict(
                {
                    "input_ids": initial_input_ids,
                    "attention_mask": initial_attention_mask,
                    "position_ids": initial_position_ids,
                }
            )
            gen_batch.meta_info.update(original_meta_info)

        active_mask = torch.ones(batch_size, dtype=torch.bool)
        turns_stats = [0 for _ in range(batch_size)]
        valid_action_stats = [0 for _ in range(batch_size)]
        format_valid_stats = [0 for _ in range(batch_size)]
        generated_turns_stats = [0 for _ in range(batch_size)]
        summary_length_sums = [0.0 for _ in range(batch_size)]
        summary_count_stats = [0 for _ in range(batch_size)]
        invalid_action_counts = [0 for _ in range(batch_size)]
        blocked_move_counts = [0 for _ in range(batch_size)]
        success_stats = [0 for _ in range(batch_size)]
        unique_cells_visited_stats = [1 for _ in range(batch_size)]
        solved_path_ratio_sums = [0.0 for _ in range(batch_size)]
        solved_path_ratio_counts = [0 for _ in range(batch_size)]
        batch_rewards = [0.0 for _ in range(batch_size)]
        carried_summary_blocks = ["" for _ in range(batch_size)]

        reconstruction_list = [{"q": initial_input_ids[idx].clone()} for idx in range(batch_size)]
        kept_lengths = [0 for _ in range(batch_size)]
        initial_token_lengths = [
            int((gen_batch.batch["input_ids"][idx] != self.tokenizer.pad_token_id).sum().item())
            for idx in range(batch_size)
        ]

        rollings = gen_batch
        meta_info = {}

        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break

            if step > 0:
                full_prompt_texts = []
                for idx in range(batch_size):
                    full_prompt_texts.append(
                        compose_rollout_input(
                            base_prompt_texts[idx],
                            carried_summary_blocks[idx],
                            pending_observation_blocks[idx],
                        )
                    )
                rebuilt_input_ids = self._batch_tokenize(full_prompt_texts)
                if rebuilt_input_ids.shape[1] > self.config.max_prompt_length:
                    rebuilt_input_ids = rebuilt_input_ids[:, -self.config.max_prompt_length :]
                rebuilt_attention_mask = self.tensor_fn.create_attention_mask(rebuilt_input_ids)
                rebuilt_position_ids = self.tensor_fn.create_position_ids(rebuilt_attention_mask)
                rollings = DataProto.from_dict(
                    {
                        "input_ids": rebuilt_input_ids,
                        "attention_mask": rebuilt_attention_mask,
                        "position_ids": rebuilt_position_ids,
                    }
                )
                rollings.meta_info.update(gen_batch.meta_info)

            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=["input_ids", "attention_mask", "position_ids"],
            )

            rollings_active = DataProto.from_dict({key: value[active_mask] for key, value in rollings.batch.items()})
            rollings_active.meta_info.update(gen_batch.meta_info)

            active_kept_lengths = [kept_lengths[idx] for idx in range(batch_size) if active_mask[idx]]
            active_initial_token_lengths = [initial_token_lengths[idx] for idx in range(batch_size) if active_mask[idx]]

            padding_mask = self.tensor_fn.create_attention_mask(rollings_active.batch["input_ids"])
            padding_mask = self.tensor_fn.mask_using_kept_lengths(
                padding_mask,
                active_kept_lengths,
                active_initial_token_lengths,
            )

            input_ids = rollings_active.batch["input_ids"]
            input_ids = torch.where(padding_mask == 0, self.tokenizer.pad_token_id, input_ids)
            input_ids, _ = self.tensor_fn.convert_pad_structure(input_ids, pad_to_left=True)
            rollings_active.batch["input_ids"] = input_ids
            rollings_active.batch["attention_mask"] = self.tensor_fn.create_attention_mask(input_ids)
            rollings_active.batch["position_ids"] = self.tensor_fn.create_position_ids(rollings_active.batch["attention_mask"])

            gen_output = self._generate_with_gpu_padding(rollings_active)
            meta_info.update(gen_output.meta_info)

            active_response_ids, active_response_texts = self._postprocess_responses(gen_output.batch["responses"])
            responses_ids, responses_texts = self.tensor_fn._example_level_pad(
                active_response_ids,
                active_response_texts,
                active_mask,
            )

            dones = [True for _ in range(batch_size)]
            current_step_rewards = [0.0 for _ in range(batch_size)]
            next_kept_lengths = [0 for _ in range(batch_size)]
            carry_tokens = [responses_ids[idx][:0] for idx in range(batch_size)]
            next_observation_blocks = [pending_observation_blocks[idx] for idx in range(batch_size)]

            for idx in range(batch_size):
                if not active_mask[idx]:
                    continue

                turns_stats[idx] += 1
                generated_turns_stats[idx] += 1
                raw_response_ids = self._valid_tokens(responses_ids[idx])
                parsed = parse_model_output(responses_texts[idx], require_reasoning=self.config.require_reasoning)

                if parsed.format_ok:
                    format_valid_stats[idx] += 1
                    current_step_rewards[idx] += self.config.format_reward
                    summary_length_sums[idx] += float(self._token_count(parsed.summary_text))
                    summary_count_stats[idx] += 1
                    if parsed.summary_text:
                        current_step_rewards[idx] += self.config.summary_present_reward

                    reasoning_action_text = f"{parsed.reasoning_block}{parsed.action_block}"
                    reasoning_action_ids = self._batch_tokenize([reasoning_action_text]).squeeze(0)
                    summary_ids = self._batch_tokenize([parsed.summary_block]).squeeze(0)
                    reconstruction_list[idx][f"t{step}"] = reasoning_action_ids
                    reconstruction_list[idx][f"r{step}"] = summary_ids
                    carried_summary_blocks[idx] = parsed.summary_block

                    observations, rewards, done_values, infos = env_manager.step([parsed.action_text], [idx])
                    reward = rewards[0]
                    done = done_values[0]
                    info = infos[0]
                    next_observation_blocks[idx] = observations[0]
                    current_step_rewards[idx] += reward

                    if info["action_is_valid"]:
                        valid_action_stats[idx] += 1
                    else:
                        invalid_action_counts[idx] += 1
                        current_step_rewards[idx] += self.config.invalid_action_penalty

                    if info["blocked_move"]:
                        blocked_move_counts[idx] += 1
                    if info["goal_reached"]:
                        success_stats[idx] = 1
                    unique_cells_visited_stats[idx] = int(info["unique_cells_visited"])
                    if info["path_efficiency_ratio"] is not None:
                        solved_path_ratio_sums[idx] = float(info["path_efficiency_ratio"])
                        solved_path_ratio_counts[idx] = 1

                    dones[idx] = done
                    carry_tokens[idx] = summary_ids
                    next_kept_lengths[idx] = summary_ids.shape[0]
                else:
                    reconstruction_list[idx][f"t{step}"] = raw_response_ids
                    reconstruction_list[idx][f"r{step}"] = raw_response_ids[:0]

                    observations, rewards, done_values, infos = env_manager.step([""], [idx])
                    reward = rewards[0]
                    done = done_values[0]
                    info = infos[0]
                    next_observation_blocks[idx] = observations[0]
                    current_step_rewards[idx] += reward + self.config.unparsable_output_penalty

                    invalid_action_counts[idx] += 1
                    if info["blocked_move"]:
                        blocked_move_counts[idx] += 1
                    if info["goal_reached"]:
                        success_stats[idx] = 1
                    unique_cells_visited_stats[idx] = int(info["unique_cells_visited"])
                    if info["path_efficiency_ratio"] is not None:
                        solved_path_ratio_sums[idx] = float(info["path_efficiency_ratio"])
                        solved_path_ratio_counts[idx] = 1

                    dones[idx] = done
                    next_kept_lengths[idx] = 0

            batch_rewards = [old + cur for old, cur in zip(batch_rewards, current_step_rewards)]
            pending_observation_blocks = next_observation_blocks
            next_obs_texts = [block if not dones[idx] else "" for idx, block in enumerate(pending_observation_blocks)]
            next_obs_ids = self._process_next_obs(next_obs_texts)
            carry_ids = self._pad_token_list(carry_tokens)

            for idx in range(batch_size):
                if next_obs_ids.shape[1] > 0:
                    reconstruction_list[idx][f"i{step}"] = next_obs_ids[idx]
                if not dones[idx] and next_obs_ids.shape[1] > 0:
                    next_kept_lengths[idx] += int((next_obs_ids[idx] != self.tokenizer.pad_token_id).sum().item())
                kept_lengths[idx] = next_kept_lengths[idx]

            active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            rollings = self._update_rolling_state(rollings, carry_ids, next_obs_ids)
            original_right_side = self._update_right_side(original_right_side, responses_ids, next_obs_ids)

        for item in reconstruction_list:
            item["num_rounds"] = max(
                len([key for key in item.keys() if key.startswith("r")]),
                len([key for key in item.keys() if key.startswith("t")]),
            )

        meta_info["turns_stats"] = turns_stats
        meta_info["valid_action_stats"] = valid_action_stats
        meta_info["format_valid_stats"] = format_valid_stats
        meta_info["generated_turns_stats"] = generated_turns_stats
        meta_info["summary_length_sums"] = summary_length_sums
        meta_info["summary_count_stats"] = summary_count_stats
        meta_info["invalid_action_counts"] = invalid_action_counts
        meta_info["blocked_move_counts"] = blocked_move_counts
        meta_info["success_stats"] = success_stats
        meta_info["unique_cells_visited_stats"] = unique_cells_visited_stats
        meta_info["solved_path_ratio_sums"] = solved_path_ratio_sums
        meta_info["solved_path_ratio_counts"] = solved_path_ratio_counts
        meta_info["retry_counts"] = [0 for _ in range(batch_size)]
        meta_info["batch_rewards"] = batch_rewards
        meta_info["active_mask"] = active_mask.tolist()

        return self._compose_final_output(reconstruction_list, meta_info)
