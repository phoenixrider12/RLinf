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
        action_dim: int,
        num_action_chunks: int,
        add_value_head: bool = True,
        unnorm_key: Optional[str] = None,
        unnormalize_action_fn: Optional[Any] = None,
    ):
        super().__init__()
        self.deepthinkvla_model = deepthinkvla_model
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
            result["logprobs"] = None # To be implemented
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
        # Assuming env_obs has pixel_values and input_ids
        pixel_values = env_obs.get("pixel_values", env_obs.get("image"))
        input_ids = env_obs.get("input_ids", env_obs.get("prompt"))
        attention_mask = env_obs.get("attention_mask")
        
        if isinstance(pixel_values, list):
            pixel_values = torch.cat(pixel_values, dim=0)
        
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)
            
        bsz = pixel_values.shape[0]

        # In case we don't have input_ids
        if input_ids is None:
            # Need a fallback, but for now we assume they are provided
            pass
            
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
            
        if self.unnormalize_action_fn is not None:
            unnormalized = self.unnormalize_action_fn(torch.from_numpy(normalized_actions))
            env_chunk_actions = unnormalized.numpy()
        else:
            env_chunk_actions = normalized_actions
            
        env_chunk_actions = env_chunk_actions.reshape(bsz, self.num_action_chunks, self.action_dim)

        
        forward_inputs = {
            "input_cot_ids": return_input_cot_ids.cpu(),
            "attention_mask": return_attention_mask.cpu(),
            "pixel_values": pixel_values.cpu(),
            "action": torch.from_numpy(env_chunk_actions).view(bsz, -1)
        }
        
        result = {
            "prev_logprobs": None, # Should be calculated
            "prev_values": None,
            "forward_inputs": forward_inputs,
        }
        
        return env_chunk_actions, result

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.deepthinkvla_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.deepthinkvla_model.gradient_checkpointing_disable()

