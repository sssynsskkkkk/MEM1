import torch
from dataclasses import dataclass
from typing import Dict, List, Optional

from verl import DataProto

from .attn_mask_utils import compose_final_output
from .game24_utils import apply_action_to_state, extract_puzzle_spec, format_fraction, format_state, parse_model_output
from .tensor_helper import TensorConfig, TensorHelper


@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    require_reasoning: bool = True
    prepend_no_think: bool = False
    format_reward: float = 0.0
    summary_present_reward: float = 0.0
    valid_action_reward: float = 0.0
    invalid_action_penalty: float = 0.0
    step_penalty: float = 0.0
    unparsable_output_penalty: float = 0.0
    value_tolerance: float = 1e-5


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

    def _build_state_block(
        self,
        state,
        target,
        steps_used: int,
        max_steps: int,
        attempt_index: int,
        previous_result: str,
    ) -> str:
        turns_left = max(max_steps - steps_used, 0)
        lines = [
            "<observation>",
            f"attempt_index: {attempt_index + 1}",
            f"remaining_numbers: [{format_state(state)}]",
            f"target: {format_fraction(target)}",
            f"turns_left_in_current_attempt: {turns_left}",
        ]
        if previous_result:
            lines.append(f"previous_result: {previous_result}")
        lines.append("</observation>")
        return "\n".join(lines)

    def _compose_rollout_input(self, base_prompt: str, summary_block: str, observation_block: str) -> str:
        pieces = [base_prompt.strip()]
        if summary_block:
            pieces.append(summary_block.strip())
        if observation_block:
            pieces.append(observation_block.strip())
        return "\n\n".join(piece for piece in pieces if piece)

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor, ground_truths: List[Dict]):
        if ground_truths is None:
            raise ValueError("ground_truths are required for Game24 rollout")

        original_right_side = {
            "responses": initial_input_ids[:, []],
            "responses_with_info_mask": initial_input_ids[:, []],
        }

        batch_size = gen_batch.batch["input_ids"].shape[0]
        original_meta_info = dict(gen_batch.meta_info)
        base_prompt_texts = []
        prompt_texts_changed = False
        for i in range(batch_size):
            row = initial_input_ids[i]
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

        specs = [extract_puzzle_spec(item) for item in ground_truths]
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        turns_stats = [0 for _ in range(batch_size)]
        valid_action_stats = [0 for _ in range(batch_size)]
        format_valid_stats = [0 for _ in range(batch_size)]
        generated_turns_stats = [0 for _ in range(batch_size)]
        summary_length_sums = [0.0 for _ in range(batch_size)]
        summary_count_stats = [0 for _ in range(batch_size)]
        invalid_action_counts = [0 for _ in range(batch_size)]
        retry_counts = [0 for _ in range(batch_size)]
        success_stats = [0 for _ in range(batch_size)]
        batch_rewards = [0.0 for _ in range(batch_size)]
        initial_states = [list(spec["numbers"]) for spec in specs]
        current_states = [list(spec["numbers"]) for spec in specs]
        current_targets = [spec["target"] for spec in specs]
        current_problem_max_steps = [spec["max_steps"] for spec in specs]
        steps_in_current_attempt = [0 for _ in range(batch_size)]
        carried_summary_blocks = ["" for _ in range(batch_size)]
        pending_observation_blocks = [
            self._build_state_block(
                current_states[i],
                current_targets[i],
                steps_in_current_attempt[i],
                current_problem_max_steps[i],
                retry_counts[i],
                previous_result="attempt_start",
            )
            for i in range(batch_size)
        ]

        reconstruction_list = [{"q": initial_input_ids[i].clone()} for i in range(batch_size)]
        kept_lengths = [0 for _ in range(batch_size)]
        initial_token_lengths = [
            int((gen_batch.batch["input_ids"][i] != self.tokenizer.pad_token_id).sum().item()) for i in range(batch_size)
        ]

        rollings = gen_batch
        meta_info = {}
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break

            if step > 0:
                full_prompt_texts = []
                for i in range(batch_size):
                    full_prompt_texts.append(
                        self._compose_rollout_input(
                            base_prompt_texts[i],
                            carried_summary_blocks[i],
                            pending_observation_blocks[i],
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

            active_kept_lengths = [kept_lengths[i] for i in range(batch_size) if active_mask[i]]
            active_initial_token_lengths = [initial_token_lengths[i] for i in range(batch_size) if active_mask[i]]

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

            next_obs_texts = ["" for _ in range(batch_size)]
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
                current_step_rewards[idx] += self.config.step_penalty

                parsed = parse_model_output(responses_texts[idx], require_reasoning=self.config.require_reasoning)
                raw_response_ids = self._valid_tokens(responses_ids[idx])

                if parsed.format_ok:
                    format_valid_stats[idx] += 1
                    current_step_rewards[idx] += self.config.format_reward
                    summary_length_sums[idx] += float(self._token_count(parsed.summary_text))
                    summary_count_stats[idx] += 1
                    if parsed.summary_text:
                        current_step_rewards[idx] += self.config.summary_present_reward

                    action_outcome = apply_action_to_state(
                        current_states[idx],
                        parsed.action_text,
                        tolerance=self.config.value_tolerance,
                    )
                    reasoning_action_text = f"{parsed.reasoning_block}{parsed.action_block}"
                    reasoning_action_ids = self._batch_tokenize([reasoning_action_text]).squeeze(0)
                    summary_ids = self._batch_tokenize([parsed.summary_block]).squeeze(0)
                    reconstruction_list[idx][f"t{step}"] = reasoning_action_ids
                    reconstruction_list[idx][f"r{step}"] = summary_ids
                    carried_summary_blocks[idx] = parsed.summary_block

                    if action_outcome.is_valid:
                        valid_action_stats[idx] += 1
                        current_step_rewards[idx] += self.config.valid_action_reward
                        current_states[idx] = action_outcome.next_state
                        steps_in_current_attempt[idx] += 1

                        previous_result = "continue"
                        reset_attempt = False
                        if len(current_states[idx]) == 1:
                            if abs(float(current_states[idx][0] - current_targets[idx])) <= self.config.value_tolerance:
                                success_stats[idx] = 1
                                previous_result = "solved"
                                dones[idx] = True
                            else:
                                previous_result = f"retry_wrong_final_value:{format_fraction(current_states[idx][0])}"
                                reset_attempt = True
                        elif steps_in_current_attempt[idx] >= current_problem_max_steps[idx]:
                            previous_result = "retry_step_limit"
                            reset_attempt = True

                        if success_stats[idx] == 1:
                            pass
                        elif reset_attempt:
                            retry_counts[idx] += 1
                            current_states[idx] = list(initial_states[idx])
                            steps_in_current_attempt[idx] = 0
                            dones[idx] = False
                            next_observation_blocks[idx] = self._build_state_block(
                                current_states[idx],
                                current_targets[idx],
                                steps_in_current_attempt[idx],
                                current_problem_max_steps[idx],
                                retry_counts[idx],
                                previous_result=previous_result,
                            )
                            carry_tokens[idx] = summary_ids
                            next_kept_lengths[idx] = summary_ids.shape[0]
                        else:
                            dones[idx] = False
                            next_observation_blocks[idx] = self._build_state_block(
                                current_states[idx],
                                current_targets[idx],
                                steps_in_current_attempt[idx],
                                current_problem_max_steps[idx],
                                retry_counts[idx],
                                previous_result=previous_result,
                            )
                            carry_tokens[idx] = summary_ids
                            next_kept_lengths[idx] = summary_ids.shape[0]
                    else:
                        invalid_action_counts[idx] += 1
                        retry_counts[idx] += 1
                        current_step_rewards[idx] += self.config.invalid_action_penalty
                        current_states[idx] = list(initial_states[idx])
                        steps_in_current_attempt[idx] = 0
                        dones[idx] = False
                        next_observation_blocks[idx] = self._build_state_block(
                            current_states[idx],
                            current_targets[idx],
                            steps_in_current_attempt[idx],
                            current_problem_max_steps[idx],
                            retry_counts[idx],
                            previous_result=f"retry_invalid_action:{action_outcome.error}",
                        )
                        carry_tokens[idx] = summary_ids
                        next_kept_lengths[idx] = summary_ids.shape[0]
                else:
                    reconstruction_list[idx][f"t{step}"] = raw_response_ids
                    reconstruction_list[idx][f"r{step}"] = raw_response_ids[:0]
                    retry_counts[idx] += 1
                    current_step_rewards[idx] += self.config.unparsable_output_penalty
                    current_states[idx] = list(initial_states[idx])
                    steps_in_current_attempt[idx] = 0
                    dones[idx] = False
                    next_observation_blocks[idx] = self._build_state_block(
                        current_states[idx],
                        current_targets[idx],
                        steps_in_current_attempt[idx],
                        current_problem_max_steps[idx],
                        retry_counts[idx],
                        previous_result="retry_unparsable_output",
                    )
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
        meta_info["retry_counts"] = retry_counts
        meta_info["success_stats"] = success_stats
        meta_info["batch_rewards"] = batch_rewards
        meta_info["active_mask"] = active_mask.tolist()

        return self._compose_final_output(reconstruction_list, meta_info)
