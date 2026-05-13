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

"""DeepThinkVLA embodied policy wrapper for RLinf."""

from __future__ import annotations

import os
import sys
import torch
from omegaconf import DictConfig

from rlinf.utils.logging import get_logger

from .deepthinkvla_action_model import DeepThinkVLAForRLActionPrediction

def get_model(
    cfg: DictConfig,
    torch_dtype: torch.dtype | None = None,
) -> DeepThinkVLAForRLActionPrediction:
    logger = get_logger()
    model_path = getattr(cfg, "model_path", None)
    if model_path is None:
        raise ValueError(
            "DeepThinkVLA requires 'actor.model.model_path'."
        )

    # Append path to user's deepthinkvla implementation
    sys.path.append("/data/aryaman/DeepThinkVLA_RL/src")
    
    try:
        from sft.modeling_deepthinkvla import DeepThinkVLA
        from dt_datasets.normalize import Unnormalize_Action
        from sft.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_MASK
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Could not import DeepThinkVLA from sft.modeling_deepthinkvla. "
            "Please ensure /data/aryaman/DeepThinkVLA_RL/src is available."
        ) from e

    logger.info(f"Loading DeepThinkVLA checkpoint: {model_path}")

    # Load unnormalization stats if available
    unnormalize_action_fn = None
    import json
    import numpy as np
    from huggingface_hub import hf_hub_download
    
    try:
        if os.path.isdir(model_path):
            stats_path = os.path.join(model_path, "norm_stats.json")
        else:
            stats_path = hf_hub_download(repo_id=model_path, filename="norm_stats.json")
            
        if os.path.isfile(stats_path):
            with open(stats_path, "r") as f:
                norm_stats = json.load(f)
            for key in norm_stats["action"].keys():
                norm_stats["action"][key] = np.array(norm_stats["action"][key], dtype=np.float64)
            unnormalize_action_fn = Unnormalize_Action(
                normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
                stats=norm_stats["action"],
                action_mask=ACTION_MASK,
            )
            logger.info("Successfully loaded action unnormalization statistics.")
    except Exception as e:
        logger.warning(f"Could not load norm_stats.json from {model_path}. Actions will NOT be unnormalized! ({e})")

    # Use AutoModel or DeepThinkVLA.from_pretrained to load
    # For now we use the DeepThinkVLA class from their source
    deepthinkvla_model = DeepThinkVLA.from_pretrained(
        model_path,
        torch_dtype=torch_dtype if torch_dtype is not None else torch.float32,
    )

    if torch_dtype is not None:
        deepthinkvla_model = deepthinkvla_model.to(dtype=torch_dtype)

    return DeepThinkVLAForRLActionPrediction(
        deepthinkvla_model=deepthinkvla_model,
        action_dim=cfg.action_dim,
        num_action_chunks=cfg.num_action_chunks,
        add_value_head=getattr(cfg, "add_value_head", True),
        unnorm_key=getattr(cfg, "unnorm_key", None),
        unnormalize_action_fn=unnormalize_action_fn,
    )

__all__ = ["DeepThinkVLAForRLActionPrediction", "get_model"]
