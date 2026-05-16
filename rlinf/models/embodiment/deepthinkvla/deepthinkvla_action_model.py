# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RLinf policy wrapper for DeepThinkVLA checkpoints."""

from __future__ import annotations

import logging
import warnings
from functools import partial
from typing import Any, Optional
import sys

import numpy as np
import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType

logger = logging.getLogger(__name__)


class DeepThinkVLAForRLActionPrediction(nn.Module, BasePolicy):
    def __init__(
        self,
        deepthinkvla_model: nn.Module,
        processor: Any,
        action_dim: int,
        num_action_chunks: int,
        add_value_head: bool = True,
        unnorm_key: Optional[str] = None,
        unnormalize_action_fn: Optional[Any] = None,
    ):
        super().__init__()
        self.deepthinkvla_model = deepthinkvla_model
        self.processor = processor
        self.action_dim = int(action_dim)
        self.num_action_chunks = int(num_action_chunks)
        self.unnorm_key = unnorm_key
        self.unnormalize_action_fn = unnormalize_action_fn
        
        policy_param_dtype = next(
            (p.dtype for p in deepthinkvla_model.parameters() if p.is_floating_point()),
            torch.float32,
        )

        self.value_head: Optional[nn.Module] = None
        if add_value_head:
            hidden_size = deepthinkvla_model.config.text_config.hidden_size
            self.value_head = nn.Linear(hidden_size, 1).to(dtype=policy_param_dtype)

    @property
    def _no_split_modules(self) -> list[str]:
        return [
            "GemmaDecoderLayer",
            "SiglipVisionEmbeddings",
            "GemmaRMSNorm",
            "GemmaRotaryEmbedding",
        ]

    def forward(
        self,
        forward_type: ForwardType = ForwardType.DEFAULT,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor | None]:
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"Unsupported forward_type: {forward_type}")

    def default_forward(
        self,
        forward_inputs: Optional[dict[str, torch.Tensor]] = None,
        compute_logprobs: bool = False,
        compute_entropy: bool = False,
        compute_values: bool = False,
        use_cache: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        
        input_cot_ids = forward_inputs["input_cot_ids"]
        pixel_values = forward_inputs["pixel_values"]
        attention_mask = forward_inputs["attention_mask"]

        if pixel_values.ndim == 5:
            bsz, num_images = pixel_values.shape[:2]
            pixel_values = pixel_values.view(bsz * num_images, *pixel_values.shape[2:])

        # This will call forward of DeepThinkVLA with cot_length
        # Since RLinf provides the full trajectory, we need to extract prompt+cot and action parts.
        # But wait! deepthinkvla forward signature has `cot_length` argument for training...
        # However, for PPO/GRPO, the logprobs are typically computed during `calculate_logprobs`
        # in the rollout or training phase.
        
        # We need to compute logprobs of the generated sequence
        # DeepThinkVLA handles this differently. Let's do a simple forward for now.
        
        result = {}
        if compute_logprobs:
            from verl.utils.torch_functional import logprobs_from_logits
            
            pad_id = self.deepthinkvla_model.pad_token_id
            input_cot_ids = input_cot_ids.to(self.deepthinkvla_model.device)
            
            # SINGLE FORWARD PASS
            logits_all, action_start_idx = self.deepthinkvla_model.prompt_cot_predict_action(
                input_cot_ids=input_cot_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
            )
            
            bsz = logits_all.shape[0]
            
            # 1. Action Logprobs
            action_tokens = forward_inputs.get("action_tokens")
            if action_tokens is not None:
                action_tokens = action_tokens.to(logits_all.device)
                start_indices = action_start_idx.unsqueeze(1)
                position_offsets = torch.arange(self.num_action_chunks * self.action_dim, device=logits_all.device).unsqueeze(0)
                seq_indices = start_indices + position_offsets
                
                action_logits = logits_all[
                    torch.arange(logits_all.shape[0], device=logits_all.device).unsqueeze(-1),
                    seq_indices,
                    self.deepthinkvla_model.config.action_token_begin_idx:self.deepthinkvla_model.config.action_token_end_idx + 1
                ]
                
                action_responses = action_tokens - self.deepthinkvla_model.config.action_token_begin_idx
                action_logp = torch.nn.functional.log_softmax(action_logits, dim=-1)
                action_logprobs = torch.gather(action_logp, 2, action_responses.unsqueeze(2)).squeeze(2)
                
                action_seq_logprobs = action_logprobs.sum(dim=-1)
            else:
                action_seq_logprobs = torch.zeros(bsz, device=logits_all.device, dtype=torch.float32)
                
            # 2. CoT Logprobs
            sorted_indices = torch.argsort(((input_cot_ids.ne(pad_id))).int(), dim=1, descending=True, stable=True)
            sorted_input_cot_ids = torch.gather(input_cot_ids, 1, sorted_indices).to(logits_all.device)
            
            cot_logits = logits_all[:, :sorted_input_cot_ids.shape[1]]
            shift_logits = cot_logits[..., :-1, :].contiguous()
            shift_labels = sorted_input_cot_ids[..., 1:].contiguous()
            
            cot_all_logprobs = logprobs_from_logits(shift_logits, shift_labels, inplace_backward=False)
            cot_all_logprobs = torch.cat([torch.zeros_like(cot_all_logprobs[:, :1]), cot_all_logprobs], dim=1)
            
            cot_seq_logprobs = torch.zeros(bsz, device=logits_all.device, dtype=torch.float32)
            prompt_lens = forward_inputs.get("prompt_lens", None)
            
            if prompt_lens is not None:
                for i in range(bsz):
                    # Use the provided prompt_lens directly. In right-padded sequence, 
                    # the CoT tokens start exactly at prompt_lens[i].
                    start = prompt_lens[i]
                    end = action_start_idx[i] + 1
                    cot_seq_logprobs[i] = cot_all_logprobs[i, start:end].sum()
            else:
                for i in range(bsz):
                    end = action_start_idx[i] + 1
                    cot_seq_logprobs[i] = cot_all_logprobs[i, :end].sum()
                
            seq_logprobs = cot_seq_logprobs + action_seq_logprobs
                
            dummy_logprobs = torch.zeros(bsz, self.num_action_chunks, self.action_dim, device=logits_all.device, dtype=torch.float32)
            dummy_logprobs[:, 0, 0] = seq_logprobs
            result["logprobs"] = dummy_logprobs
        if compute_entropy:
            result["entropy"] = None
        if compute_values and self.value_head is not None:
            # We don't have outputs.hidden_states from prompt_cot_predict_action easily,
            # but since GRPO doesn't use the value head anyway, we can just return None.
            result["values"] = None

        return result

    def _find_action_start_idx(self, input_cot_ids: torch.Tensor) -> torch.Tensor:
        """Find the first action token position in each sequence without a forward pass.

        Scans ``input_cot_ids`` for tokens in the action-token range
        ``[action_token_begin_idx, action_token_end_idx]`` and returns the
        index of the first match per sample.

        Args:
            input_cot_ids: Token IDs ``[bsz, seq_len]``.

        Returns:
            ``action_start_idx`` tensor of shape ``[bsz]`` with the position of
            the first action token in each sample. If no action token is found,
            defaults to ``seq_len - 1``.
        """
        begin_id = self.deepthinkvla_model.config.action_token_begin_idx
        end_id = self.deepthinkvla_model.config.action_token_end_idx
        # Boolean mask: True where token is an action token
        is_action = (input_cot_ids >= begin_id) & (input_cot_ids <= end_id)
        # argmax on a bool tensor returns the index of the first True.
        # If no True exists, argmax returns 0 — we detect that case and
        # fall back to seq_len - 1.
        first_action_pos = is_action.int().argmax(dim=-1)  # [bsz]
        # Fix samples where no action token was found (argmax returned 0 but
        # position 0 is not actually an action token).
        no_action_mask = ~is_action.any(dim=-1)
        first_action_pos[no_action_mask] = input_cot_ids.shape[1] - 1
        return first_action_pos

    def compute_masked_cot_action_logprobs(
        self,
        forward_inputs: dict[str, torch.Tensor],
        skip_masking: bool = False,
    ) -> torch.Tensor:
        """Compute action log-probabilities with CoT tokens masked out of attention.

        Runs a forward pass where the attention mask is zeroed for CoT token
        positions so the model cannot attend to its own reasoning. The resulting
        action log-probabilities can be compared to the normal ones to measure
        how much the CoT causally influences actions.

        When ``skip_masking=True``, no attention masking is applied — this is
        used for the wrong-CoT path where CoT tokens have already been shifted
        at the batch level and we just need a plain forward pass.

        Args:
            forward_inputs: Dict with ``input_cot_ids``, ``pixel_values``,
                ``attention_mask``, ``prompt_lens``, and ``action_tokens``.
            skip_masking: If True, skip zeroing out CoT attention positions.

        Returns:
            Scalar action-sequence log-probability per sample ``[bsz]``.
        """
        input_cot_ids = forward_inputs["input_cot_ids"].to(self.deepthinkvla_model.device)
        pixel_values = forward_inputs["pixel_values"]
        attention_mask = forward_inputs["attention_mask"].clone().to(self.deepthinkvla_model.device)
        action_tokens = forward_inputs.get("action_tokens")
        prompt_lens = forward_inputs.get("prompt_lens")

        if pixel_values.ndim == 5:
            bsz, num_images = pixel_values.shape[:2]
            pixel_values = pixel_values.view(bsz * num_images, *pixel_values.shape[2:])

        bsz = input_cot_ids.shape[0]

        # Find action start positions by scanning token IDs — no forward pass needed.
        action_start_idx = self._find_action_start_idx(input_cot_ids)

        if not skip_masking:
            # Zero out CoT positions in the attention mask
            if prompt_lens is not None:
                prompt_lens_dev = prompt_lens.to(input_cot_ids.device)
                for i in range(bsz):
                    cot_start = int(prompt_lens_dev[i].item())
                    cot_end = int(action_start_idx[i].item())
                    attention_mask[i, cot_start:cot_end] = 0

        # Forward (with masked or unmodified attention depending on skip_masking)
        with torch.no_grad():
            logits_out, action_start_idx_out = self.deepthinkvla_model.prompt_cot_predict_action(
                input_cot_ids=input_cot_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
            )

        # Extract action logprobs
        if action_tokens is not None:
            action_tokens_dev = action_tokens.to(logits_out.device)
            start_indices = action_start_idx_out.unsqueeze(1)
            position_offsets = torch.arange(
                self.num_action_chunks * self.action_dim, device=logits_out.device
            ).unsqueeze(0)
            seq_indices = start_indices + position_offsets

            action_logits = logits_out[
                torch.arange(bsz, device=logits_out.device).unsqueeze(-1),
                seq_indices,
                self.deepthinkvla_model.config.action_token_begin_idx:self.deepthinkvla_model.config.action_token_end_idx + 1,
            ]

            action_responses = action_tokens_dev - self.deepthinkvla_model.config.action_token_begin_idx
            action_logp = torch.nn.functional.log_softmax(action_logits, dim=-1)
            action_logprobs = torch.gather(action_logp, 2, action_responses.unsqueeze(2)).squeeze(2)
            result_logprobs = action_logprobs.sum(dim=-1)
        else:
            result_logprobs = torch.zeros(bsz, device=logits_out.device, dtype=torch.float32)

        del logits_out
        torch.cuda.empty_cache()

        return result_logprobs

    def prepare_wrong_cot_inputs(
        self,
        forward_inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Circularly shift CoT tokens across the batch for wrong-CoT contrastive reward.

        This performs the CoT token shift at the full-batch level so the result
        can be micro-batched for forward passes without losing the cross-batch
        contrastive property.

        Args:
            forward_inputs: Dict with ``input_cot_ids``, ``pixel_values``,
                ``attention_mask``, ``prompt_lens``, and ``action_tokens``.

        Returns:
            A new forward_inputs dict with ``input_cot_ids`` containing
            shifted CoT tokens. All other fields are passed through unchanged.
        """
        input_cot_ids = forward_inputs["input_cot_ids"].clone().to(self.deepthinkvla_model.device)
        prompt_lens = forward_inputs.get("prompt_lens")

        bsz = input_cot_ids.shape[0]
        if bsz <= 1:
            # Cannot shift with a single sample; return unmodified
            return dict(forward_inputs)

        pad_id = self.deepthinkvla_model.pad_token_id
        # Find CoT boundaries from token IDs — no forward pass needed.
        action_start_idx = self._find_action_start_idx(input_cot_ids)

        # Shift CoT tokens circularly across the batch
        if prompt_lens is not None:
            prompt_lens_dev = prompt_lens.to(input_cot_ids.device)
            max_cot_len = 0
            cot_slices = []
            for i in range(bsz):
                start = int(prompt_lens_dev[i].item())
                end = int(action_start_idx[i].item())
                cot_slices.append((start, end))
                max_cot_len = max(max_cot_len, end - start)

            if max_cot_len > 0:
                cot_tokens = torch.full(
                    (bsz, max_cot_len), pad_id,
                    device=input_cot_ids.device, dtype=input_cot_ids.dtype,
                )
                for i, (s, e) in enumerate(cot_slices):
                    length = e - s
                    if length > 0:
                        cot_tokens[i, :length] = input_cot_ids[i, s:e]

                cot_tokens_shifted = torch.roll(cot_tokens, shifts=1, dims=0)

                for i, (s, e) in enumerate(cot_slices):
                    length = e - s
                    if length > 0:
                        input_cot_ids[i, s:e] = cot_tokens_shifted[i, :length]

        # Return new dict with shifted input_cot_ids, everything else unchanged
        shifted = dict(forward_inputs)
        shifted["input_cot_ids"] = input_cot_ids
        return shifted

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        calculate_logprobs: bool = True,
        calculate_values: bool = True,
        return_obs: bool = True,
        mode: str = "train",
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        
        # Prepare inputs from env_obs
        from experiments.deepthinkvla_utils import prepare_image_for_vla, THINK_PREFIX
        
        main_images = env_obs.get("main_images")
        wrist_images = env_obs.get("wrist_images")
        task_descriptions = env_obs.get("task_descriptions")
        
        if isinstance(main_images, torch.Tensor):
            main_images = main_images.detach().cpu().numpy()
        if wrist_images is not None and isinstance(wrist_images, torch.Tensor):
            wrist_images = wrist_images.detach().cpu().numpy()
            
        bsz = main_images.shape[0]
        batch_images = []
        batch_texts = []
        
        for i in range(bsz):
            img_list = [prepare_image_for_vla(main_images[i])]
            if wrist_images is not None:
                img_list.append(prepare_image_for_vla(wrist_images[i]))
            
            task_label = str(task_descriptions[i]) if task_descriptions is not None else ""
            prompt = self.processor.tokenizer.additional_special_tokens[0] * len(img_list) + THINK_PREFIX + f"Task: {task_label.lower()};"
            
            batch_images.extend(img_list)
            batch_texts.append(prompt)
            
        inputs = self.processor(text=batch_texts, images=batch_images, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]
        attention_mask = inputs.get("attention_mask")
        
        # PaliGemma expects pixel_values to be 4D: [total_images_in_batch, C, H, W]
        # The AutoProcessor already returns it as 4D. No reshaping needed!
        sampling_params = kwargs.get("sampling_params", {})
        length_params = kwargs.get("length_params", {})
        
        do_sample = sampling_params.get("do_sample", mode == "train")
        temperature = sampling_params.get("temperature_train" if mode == "train" else "temperature_eval", 1.0)
        
        from transformers import GenerationConfig
        generation_config = GenerationConfig(
            max_new_tokens=length_params.get("max_new_token", 300),
            max_length=length_params.get("max_length", 1024),
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=self.deepthinkvla_model.pad_token_id,
            eos_token_id=self.deepthinkvla_model.config.eos_token_id,
        )
        
        with torch.no_grad():
            normalized_actions, predicted_action_token_ids, return_input_cot_ids, return_attention_mask = self.deepthinkvla_model.generate_action_verl(
                input_ids=input_ids.to(self.deepthinkvla_model.device),
                pixel_values=pixel_values.to(self.deepthinkvla_model.device),
                attention_mask=attention_mask.to(self.deepthinkvla_model.device) if attention_mask is not None else None,
                do_sample=do_sample,
                temperature=temperature,
                generation_config=generation_config,
            )
            
            prompt_len = input_ids.shape[1]
            if calculate_logprobs:
                pad_id = self.deepthinkvla_model.pad_token_id
                
                # SINGLE FORWARD PASS
                logits_all, action_start_idx = self.deepthinkvla_model.prompt_cot_predict_action(
                    input_cot_ids=return_input_cot_ids,
                    pixel_values=pixel_values.to(self.deepthinkvla_model.device),
                    attention_mask=return_attention_mask,
                )
                
                # 1. Action Logprobs
                start_indices = action_start_idx.unsqueeze(1)
                position_offsets = torch.arange(self.num_action_chunks * self.action_dim, device=logits_all.device).unsqueeze(0)
                seq_indices = start_indices + position_offsets
                
                action_logits = logits_all[
                    torch.arange(logits_all.shape[0], device=logits_all.device).unsqueeze(-1),
                    seq_indices,
                    self.deepthinkvla_model.config.action_token_begin_idx:self.deepthinkvla_model.config.action_token_end_idx + 1
                ]
                
                action_responses = predicted_action_token_ids.to(action_logits.device) - self.deepthinkvla_model.config.action_token_begin_idx
                action_logp = torch.nn.functional.log_softmax(action_logits, dim=-1)
                action_logprobs = torch.gather(action_logp, 2, action_responses.unsqueeze(2)).squeeze(2)
                
                action_seq_logprobs = action_logprobs.sum(dim=-1)
                
                # 2. CoT Logprobs
                sorted_indices = torch.argsort(((return_input_cot_ids.ne(pad_id))).int(), dim=1, descending=True, stable=True)
                sorted_input_cot_ids = torch.gather(return_input_cot_ids, 1, sorted_indices).to(logits_all.device)
                
                cot_logits = logits_all[:, :sorted_input_cot_ids.shape[1]]
                shift_logits = cot_logits[..., :-1, :].contiguous()
                shift_labels = sorted_input_cot_ids[..., 1:].contiguous()
                
                from verl.utils.torch_functional import logprobs_from_logits
                cot_all_logprobs = logprobs_from_logits(shift_logits, shift_labels, inplace_backward=False)
                cot_all_logprobs = torch.cat([torch.zeros_like(cot_all_logprobs[:, :1]), cot_all_logprobs], dim=1)
                
                cot_seq_logprobs = torch.zeros(bsz, device=logits_all.device, dtype=torch.float32)
                actual_prompt_lens = input_ids.ne(pad_id).sum(dim=1)
                
                for i in range(bsz):
                    start = actual_prompt_lens[i]
                    end = action_start_idx[i] + 1
                    cot_seq_logprobs[i] = cot_all_logprobs[i, start:end].sum()
                
                seq_logprobs = cot_seq_logprobs + action_seq_logprobs
                    
                prev_logprobs = torch.zeros(bsz, self.num_action_chunks, self.action_dim, dtype=torch.float32)
                prev_logprobs[:, 0, 0] = seq_logprobs.cpu()
            else:
                prev_logprobs = None
            
        if self.unnormalize_action_fn is not None:
            unnormalized = self.unnormalize_action_fn(torch.from_numpy(normalized_actions))
            env_chunk_actions = unnormalized.numpy()
        else:
            env_chunk_actions = normalized_actions
            
        env_chunk_actions = env_chunk_actions.reshape(bsz, self.num_action_chunks, self.action_dim)

        seq_len = return_input_cot_ids.shape[1]
        if not hasattr(self, "_target_seq_len"):
            max_new = kwargs.get("max_new_tokens")
            if max_new is None:
                max_new = 256
            self._target_seq_len = int((max(1024, seq_len + max_new) + 63) // 64 * 64)
            
        target_len = self._target_seq_len
        if seq_len < target_len:
            pad_len = target_len - seq_len
            pad_id = self.processor.tokenizer.pad_token_id or 0
            
            pad_tensor = torch.full((bsz, pad_len), pad_id, device=return_input_cot_ids.device, dtype=return_input_cot_ids.dtype)
            return_input_cot_ids = torch.cat([return_input_cot_ids, pad_tensor], dim=1)
            
            if return_attention_mask is not None:
                pad_mask = torch.zeros((bsz, pad_len), device=return_attention_mask.device, dtype=return_attention_mask.dtype)
                return_attention_mask = torch.cat([return_attention_mask, pad_mask], dim=1)
        elif seq_len > target_len:
            # Force truncate if it somehow exceeds to prevent stacking crash
            return_input_cot_ids = return_input_cot_ids[:, :target_len]
            if return_attention_mask is not None:
                return_attention_mask = return_attention_mask[:, :target_len]
                
        num_images_per_sample = pixel_values.shape[0] // bsz
        forward_inputs = {
            "input_cot_ids": return_input_cot_ids.cpu(),
            "attention_mask": return_attention_mask.cpu() if return_attention_mask is not None else torch.ones_like(return_input_cot_ids).cpu(),
            "pixel_values": pixel_values.cpu().view(bsz, num_images_per_sample, *pixel_values.shape[1:]),
            "prompt_lens": torch.full((bsz,), prompt_len, dtype=torch.int64).cpu(),
            "action": torch.from_numpy(env_chunk_actions).view(bsz, -1),
            "action_tokens": predicted_action_token_ids.cpu()
        }
        
        result = {
            "prev_logprobs": prev_logprobs,
            "prev_values": torch.zeros(bsz, dtype=torch.float32) if calculate_values else None,
            "forward_inputs": forward_inputs,
        }
        
        return env_chunk_actions, result

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}
        elif "use_reentrant" not in gradient_checkpointing_kwargs:
            gradient_checkpointing_kwargs["use_reentrant"] = False
        self.deepthinkvla_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.deepthinkvla_model.gradient_checkpointing_disable()

