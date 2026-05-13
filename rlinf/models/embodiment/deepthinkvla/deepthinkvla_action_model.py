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
        
        outputs = self.deepthinkvla_model(
            input_ids=input_cot_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=input_cot_ids.clone()
        )
        
        result = {}
        if compute_logprobs:
            from verl.utils.torch_functional import logprobs_from_logits
            
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_cot_ids[..., 1:].contiguous()
            
            logprobs = logprobs_from_logits(shift_logits, shift_labels, inplace_backward=False)
            logprobs = torch.cat([torch.zeros_like(logprobs[:, :1]), logprobs], dim=1)
            
            bsz = logprobs.shape[0]
            seq_logprobs = torch.zeros(bsz, device=logprobs.device, dtype=torch.float32)
            prompt_lens = forward_inputs.get("prompt_lens", None)
            
            if prompt_lens is not None:
                for i in range(bsz):
                    seq_logprobs[i] = logprobs[i, prompt_lens[i]:].sum()
            else:
                seq_logprobs = logprobs.sum(dim=-1)
                
            dummy_logprobs = torch.zeros(bsz, self.num_action_chunks, self.action_dim, device=logprobs.device, dtype=torch.float32)
            dummy_logprobs[:, 0, 0] = seq_logprobs
            result["logprobs"] = dummy_logprobs
        if compute_entropy:
            result["entropy"] = None
        if compute_values and self.value_head is not None:
            hidden_states = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else None
            result["values"] = self.value_head(hidden_states).squeeze(-1) if hidden_states is not None else None

        return result

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
        do_sample = kwargs.get("do_sample", mode == "train")
        temperature = kwargs.get("temperature", 1.0)
        
        with torch.no_grad():
            normalized_actions, predicted_action_token_ids, return_input_cot_ids, return_attention_mask = self.deepthinkvla_model.generate_action_verl(
                input_ids=input_ids.to(self.deepthinkvla_model.device),
                pixel_values=pixel_values.to(self.deepthinkvla_model.device),
                attention_mask=attention_mask.to(self.deepthinkvla_model.device) if attention_mask is not None else None,
                do_sample=do_sample,
                temperature=temperature,
            )
            
            prompt_len = input_ids.shape[1]
            if calculate_logprobs:
                outputs = self.deepthinkvla_model(
                    input_ids=return_input_cot_ids,
                    pixel_values=pixel_values.to(self.deepthinkvla_model.device),
                    attention_mask=return_attention_mask,
                    labels=return_input_cot_ids.clone()
                )
                from verl.utils.torch_functional import logprobs_from_logits
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = return_input_cot_ids[..., 1:].contiguous()
                logprobs = logprobs_from_logits(shift_logits, shift_labels, inplace_backward=False)
                logprobs = torch.cat([torch.zeros_like(logprobs[:, :1]), logprobs], dim=1)
                
                seq_logprobs = torch.zeros(bsz, device=logprobs.device, dtype=torch.float32)
                for i in range(bsz):
                    seq_logprobs[i] = logprobs[i, prompt_len:].sum()
                    
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
            "action": torch.from_numpy(env_chunk_actions).view(bsz, -1)
        }
        
        result = {
            "prev_logprobs": prev_logprobs,
            "prev_values": torch.zeros(bsz, dtype=torch.float32) if calculate_values else None,
            "forward_inputs": forward_inputs,
        }
        
        return env_chunk_actions, result

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.deepthinkvla_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.deepthinkvla_model.gradient_checkpointing_disable()

