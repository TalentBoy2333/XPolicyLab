"""Inference configuration for single-frame RoboDojo DM0.5 checkpoints."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from opendm.constants.robot import ActionMode
from opendm.exp.dm05_exp import DM05DataConfig as _DM05DataConfig
from opendm.exp.dm05_exp import DM05Exp as _DM05Exp
from opendm.exp.dm05_exp import DM05InferenceConfig as _DM05InferenceConfig
from opendm.exp.dm05_exp import DM05ModelConfig as _DM05ModelConfig


ROBODOJO_IMAGE_KEYS = ["images_1", "images_2", "images_3"]
ROBODOJO_ACTION_DIM = 14


@dataclass
class DM05ModelConfig(_DM05ModelConfig):
    """Attention defaults that work without the optional flash-attn package."""

    llm_attn_implementation: str = field(default="eager")
    vision_attn_implementation: str = field(
        default_factory=lambda: os.environ.get("OPENDM_VISION_ATTN", "sdpa")
    )
    action_attn_implementation: str = field(default="sdpa")


@dataclass
class DM05DataConfig(_DM05DataConfig):
    """Single-frame checkpoint actions are deltas with absolute grippers."""

    action_mode: ActionMode = field(default=ActionMode.RELATIVE)
    is_history: bool = field(default=False)


@dataclass
class DM05InferenceConfig(_DM05InferenceConfig):
    output_action_dim: int = field(default=ROBODOJO_ACTION_DIM)
    image_keys: list[str] = field(default_factory=lambda: list(ROBODOJO_IMAGE_KEYS))


@dataclass
class DM05Exp(_DM05Exp):
    use_lora: bool | None = field(default=False)
    model_config: DM05ModelConfig = field(default_factory=DM05ModelConfig)
    data_config: DM05DataConfig = field(default_factory=DM05DataConfig)
    inference_config: DM05InferenceConfig = field(default_factory=DM05InferenceConfig)
