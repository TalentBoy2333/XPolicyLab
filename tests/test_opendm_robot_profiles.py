from collections import deque
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from XPolicyLab.policy.OpenDM import model as opendm_model
from XPolicyLab.policy.OpenDM.robot_profiles import (
    ROBOT_PROFILES,
    get_repo_robot_action_dim_info,
)

if str(opendm_model._UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(opendm_model._UPSTREAM_ROOT))
from opendm.data.transforms import _format_task_prompt


EXPECTED_ROBOT_TYPES = {
    "arx_x5": "Dual ARX5",
    "piper": "Aloha",
    "piper_x": "Aloha Piper X",
}


def test_opendm_supports_all_three_robodojo_robot_profiles():
    assert {
        env_cfg_type: profile.robot_type
        for env_cfg_type, profile in ROBOT_PROFILES.items()
    } == EXPECTED_ROBOT_TYPES


@pytest.mark.parametrize(
    ("env_cfg_type", "robot_type"),
    sorted(EXPECTED_ROBOT_TYPES.items()),
)
def test_model_selects_robot_metadata_without_changing_mm_runtime(
    monkeypatch, tmp_path, env_cfg_type, robot_type
):
    monkeypatch.setattr(
        opendm_model,
        "_resolve_model_assets",
        lambda _: (tmp_path, tmp_path / "norm_stats.json"),
    )
    monkeypatch.setattr(opendm_model.Model, "_initialize_runtime", lambda *args: None)

    model = opendm_model.Model(
        {
            "env_cfg_type": env_cfg_type,
            "action_type": "joint",
            "history_enabled": True,
        }
    )

    assert model._robot_type == robot_type
    assert model.action_dim == 14
    assert model.history_slots == 20
    assert model.history_action_interval == 25
    assert model.supports_compact_infer is False


def test_default_and_single_frame_deploy_profiles_are_isolated():
    policy_dir = Path(__file__).resolve().parents[1] / "policy" / "OpenDM"
    default = yaml.safe_load((policy_dir / "deploy.yml").read_text())
    single_frame = yaml.safe_load(
        (policy_dir / "deploy_single_frame.yml").read_text()
    )
    piper_bf16 = yaml.safe_load(
        (policy_dir / "deploy_real_piper_single_frame_bf16.yml").read_text()
    )

    assert default["experiment_path"] == "scripts/robodojo_dm05_history.py"
    assert default["model_action_mode"] == "absolute"
    assert default["history_enabled"] is True
    assert default["action_steps"] == 25

    assert single_frame["experiment_path"] == "scripts/robodojo_dm05.py"
    assert single_frame["model_action_mode"] == "relative"
    assert single_frame["history_enabled"] is False
    assert single_frame["action_chunk_size"] == 50
    assert single_frame["action_steps"] == 25
    assert single_frame["model_max_length"] == 1024
    assert single_frame["precision_mode"] == "mixed"
    assert single_frame["bf16"] is False
    assert single_frame["enable_bf16_compute"] is True
    assert single_frame["force_fp32_action_path"] is True

    checkpoint = (
        "/mlp_vepfs/share/zrt/projects/dev/opendm/user_checkpoints/"
        "dm05_robodojo_real_piper/checkpoint-80000"
    )
    assert piper_bf16["model_path"] == checkpoint
    assert piper_bf16["norm_stats_path"] == f"{checkpoint}/norm_stats.json"
    assert piper_bf16["env_cfg_type"] == "piper"
    assert piper_bf16["experiment_path"] == "scripts/robodojo_dm05.py"
    assert piper_bf16["model_action_mode"] == "relative"
    assert piper_bf16["history_enabled"] is False
    assert piper_bf16["model_action_dim"] == 14
    assert piper_bf16["model_state_dim"] == 14
    assert piper_bf16["action_chunk_size"] == 50
    assert piper_bf16["action_steps"] == 25
    assert piper_bf16["diffusion_noise_seed"] == 0
    assert piper_bf16["prompt_format"] == "reference"
    assert piper_bf16["norm_clip_to_bounds"] is True
    assert piper_bf16["precision_mode"] == "bf16"
    assert piper_bf16["bf16"] is True
    assert piper_bf16["enable_bf16_compute"] is False
    assert piper_bf16["force_fp32_action_path"] is False
    assert piper_bf16["diffusion_integration_dtype"] == "model"

    assert single_frame["model_action_dim"] is None
    assert single_frame["model_state_dim"] is None


def test_legacy_precision_knobs_keep_the_existing_mixed_path():
    resolved = opendm_model._resolve_inference_precision(
        {
            "bf16": False,
            "enable_bf16_compute": True,
            "force_fp32_action_path": True,
            "diffusion_integration_dtype": "model",
        },
        history_enabled=True,
    )

    assert resolved == {
        "precision_mode": "legacy",
        "bf16": False,
        "enable_bf16_compute": True,
        "force_fp32_action_path": True,
        "diffusion_integration_dtype": "model",
    }

    custom_single_frame_default = opendm_model._resolve_inference_precision(
        {},
        history_enabled=False,
        default_force_fp32_action_path=True,
    )
    assert custom_single_frame_default["force_fp32_action_path"] is True


@pytest.mark.parametrize(
    ("precision_mode", "expected"),
    [
        (
            "bf16",
            {
                "bf16": True,
                "enable_bf16_compute": False,
                "force_fp32_action_path": False,
                "diffusion_integration_dtype": "model",
            },
        ),
        (
            "mixed",
            {
                "bf16": False,
                "enable_bf16_compute": True,
                "force_fp32_action_path": True,
                "diffusion_integration_dtype": "model",
            },
        ),
    ],
)
def test_named_precision_modes_are_internally_consistent(
    precision_mode,
    expected,
):
    resolved = opendm_model._resolve_inference_precision(
        {"precision_mode": precision_mode},
        history_enabled=True,
    )

    assert resolved == {"precision_mode": precision_mode, **expected}


def test_named_precision_mode_rejects_conflicting_low_level_knob():
    with pytest.raises(
        ValueError,
        match="precision_mode='bf16' requires enable_bf16_compute=False",
    ):
        opendm_model._resolve_inference_precision(
            {
                "precision_mode": "bf16",
                "enable_bf16_compute": True,
            },
            history_enabled=False,
        )


def test_reference_prompt_format_matches_the_original_single_frame_path():
    assert _format_task_prompt("pick the cube", "reference") == "Task: pick the cube."
    assert _format_task_prompt("pick the cube.", "reference") == "Task: pick the cube.."
    assert _format_task_prompt("pick the cube.", "refined") == "Task: pick the cube."


@pytest.mark.parametrize(
    ("model_action_mode", "expected_absolute_conversion"),
    [("absolute", False), ("relative", True)],
)
def test_runtime_matches_checkpoint_action_and_history_contract(
    monkeypatch,
    tmp_path,
    model_action_mode,
    expected_absolute_conversion,
):
    class FakeModelConfig:
        force_fp32_action_path = False

        def build_model(self, use_lora):
            assert use_lora is False
            return object()

    class FakeInferenceConfig:
        def _initialize(self, **kwargs):
            self.initialize_kwargs = kwargs

    experiment = SimpleNamespace(
        model_config=FakeModelConfig(),
        data_config=SimpleNamespace(is_history=True),
        inference_config=FakeInferenceConfig(),
    )
    monkeypatch.setattr(
        opendm_model,
        "_load_experiment",
        lambda _: (experiment, tmp_path / "experiment.py"),
    )
    monkeypatch.setattr(
        opendm_model,
        "_normalization_dimensions",
        lambda _: (14, 22),
    )

    model = opendm_model.Model.__new__(opendm_model.Model)
    model.model_cfg = {
        "bf16": False,
        "force_fp32_action_path": True,
        "model_max_length": 1536,
    }
    model.action_chunk_size = 50
    model.action_dim = 14
    model.add_state = True
    model.history_enabled = True
    model.history_tokens_per_slot = 16
    model.model_action_mode = model_action_mode
    model.robot_action_dim_info = {"arm_dim": [6, 6], "ee_dim": [1, 1]}

    model._initialize_runtime(tmp_path, tmp_path / "norm_stats.json")

    assert model.model_action_dim == 14
    assert model.model_state_dim == 22
    assert experiment.inference_config.output_action_dim == 14
    assert experiment.inference_config.output_action_mask == [True] * 14
    assert experiment.inference_config.initialize_kwargs["is_history"] is True
    assert (
        experiment.inference_config.initialize_kwargs["use_absolute_action"]
        is expected_absolute_conversion
    )


@pytest.mark.parametrize(
    ("model_action_mode", "expected_absolute_conversion"),
    [("absolute", False), ("relative", True)],
)
def test_runtime_matches_single_frame_checkpoint_contract(
    monkeypatch,
    tmp_path,
    model_action_mode,
    expected_absolute_conversion,
):
    class FakeModelConfig:
        force_fp32_action_path = False

        def build_model(self, use_lora):
            assert use_lora is False
            return object()

    class FakeInferenceConfig:
        def _initialize(self, **kwargs):
            self.initialize_kwargs = kwargs

    experiment = SimpleNamespace(
        model_config=FakeModelConfig(),
        data_config=SimpleNamespace(is_history=False),
        inference_config=FakeInferenceConfig(),
    )
    monkeypatch.setattr(
        opendm_model,
        "_load_experiment",
        lambda _: (experiment, tmp_path / "experiment.py"),
    )
    monkeypatch.setattr(
        opendm_model,
        "_normalization_dimensions",
        lambda _: (22, 22),
    )

    model = opendm_model.Model.__new__(opendm_model.Model)
    model.model_cfg = {
        "precision_mode": "bf16",
        "bf16": True,
        "enable_bf16_compute": False,
        "force_fp32_action_path": False,
        "diffusion_integration_dtype": "model",
        "norm_clip_to_bounds": True,
        "prompt_format": "reference",
        "model_max_length": 1024,
    }
    model.action_chunk_size = 50
    model.action_dim = 14
    model.add_state = True
    model.history_enabled = False
    model.history_tokens_per_slot = 16
    model.model_action_mode = model_action_mode
    model.robot_action_dim_info = {"arm_dim": [6, 6], "ee_dim": [1, 1]}

    model._initialize_runtime(tmp_path, tmp_path / "norm_stats.json")

    assert model.model_action_dim == 22
    assert model.model_state_dim == 22
    assert experiment.inference_config.output_action_dim == 22
    assert experiment.inference_config.output_action_mask == [True] * 14 + [False] * 8
    assert experiment.model_config.bf16 is True
    assert experiment.model_config.force_fp32_action_path is False
    assert experiment.model_config.llm_attn_implementation == "eager"
    assert experiment.inference_config.enable_bf16_compute is False
    assert experiment.inference_config.diffusion_integration_dtype == "model"
    assert experiment.inference_config.norm_clip_to_bounds is True
    assert experiment.inference_config.prompt_format == "reference"
    assert model.precision_mode == "bf16"
    assert experiment.inference_config.initialize_kwargs["is_history"] is False
    assert experiment.inference_config.initialize_kwargs["model_max_length"] == 1024
    assert (
        experiment.inference_config.initialize_kwargs["use_absolute_action"]
        is expected_absolute_conversion
    )


def test_runtime_rejects_experiment_history_mismatch(monkeypatch, tmp_path):
    experiment = SimpleNamespace(data_config=SimpleNamespace(is_history=False))
    monkeypatch.setattr(
        opendm_model,
        "_load_experiment",
        lambda _: (experiment, tmp_path / "single_frame.py"),
    )
    monkeypatch.setattr(
        opendm_model,
        "_normalization_dimensions",
        lambda _: (14, 22),
    )

    model = opendm_model.Model.__new__(opendm_model.Model)
    model.model_cfg = {}
    model.action_dim = 14
    model.history_enabled = True

    with pytest.raises(ValueError, match="history_enabled must match"):
        model._initialize_runtime(tmp_path, tmp_path / "norm_stats.json")


def test_existing_mm_history_sampling_is_unchanged():
    model = opendm_model.Model.__new__(opendm_model.Model)
    model.history_enabled = True
    model.history_image_key = "images_1"
    model.history_action_interval = 25
    model.history_slots = 20
    model.history_fps = 1.0
    model.runtime_fps = 25.0
    model._history_buffer_maxlen = 21
    model._history_by_env = {}
    model._history_sequence_by_env = {}
    observation = {
        "vision": {
            "cam_head": {
                "color": np.zeros((4, 4, 3), dtype=np.uint8),
            }
        }
    }

    for _ in range(26):
        model._append_history_observation(0, observation)

    retained = model._history_by_env[0]
    assert isinstance(retained, deque)
    assert [(sequence, actions) for sequence, actions, _ in retained] == [
        (1, 0),
        (26, 25),
    ]
    selected = model._history_images_for(0)
    assert len(selected) == 20
    assert sum(image is not None for image in selected) == 1


def test_history_adapter_is_call_only_and_scoped_calls_remain_valid():
    assert opendm_model.Model.supports_compact_infer is False
    assert not hasattr(opendm_model.Model, "infer_with_scope")
    assert "explicit CALL update_obs" in opendm_model.Model.compact_infer_error

    model = opendm_model.Model.__new__(opendm_model.Model)
    model.history_enabled = True
    model.history_image_key = "images_1"
    model.history_action_interval = 25
    model.history_slots = 20
    model.history_fps = 1.0
    model.runtime_fps = 25.0
    model._history_buffer_maxlen = 21
    model._observations = {}
    model._history_by_env = {}
    model._history_sequence_by_env = {}
    model._latest_env_idx_list = [0]
    model._latest_env_idx_by_evaluation = {}

    observation = {
        "evaluation_id": "evaluation-a",
        "env_idx": 0,
        "vision": {
            "cam_head": {
                "color": np.zeros((4, 4, 3), dtype=np.uint8),
            }
        },
    }

    model.update_obs(observation)
    state_key = ("evaluation-a", 0)
    assert model._observations[state_key] is observation
    assert model._history_sequence_by_env[state_key] == 1
    assert [
        completed_actions
        for _, completed_actions, _ in model._history_by_env[state_key]
    ] == [0]
    selected = model._history_images_for(0, "evaluation-a")
    assert sum(image is not None for image in selected) == 0
    assert model._latest_env_idx_by_evaluation == {"evaluation-a": [0]}


def test_model_allows_relative_action_mode_for_single_frame(monkeypatch, tmp_path):
    monkeypatch.setattr(
        opendm_model,
        "_resolve_model_assets",
        lambda _: (tmp_path, tmp_path / "norm_stats.json"),
    )
    monkeypatch.setattr(opendm_model.Model, "_initialize_runtime", lambda *args: None)

    model = opendm_model.Model(
        {
            "env_cfg_type": "arx_x5",
            "action_type": "joint",
            "history_enabled": False,
            "model_action_mode": "relative",
        }
    )

    assert model.model_action_mode == "relative"
    assert model.supports_compact_infer is True


def test_model_allows_relative_action_mode_for_history(monkeypatch, tmp_path):
    monkeypatch.setattr(
        opendm_model,
        "_resolve_model_assets",
        lambda _: (tmp_path, tmp_path / "norm_stats.json"),
    )
    monkeypatch.setattr(opendm_model.Model, "_initialize_runtime", lambda *args: None)

    model = opendm_model.Model(
        {
            "env_cfg_type": "arx_x5",
            "action_type": "joint",
            "history_enabled": True,
            "model_action_mode": "relative",
        }
    )

    assert model.model_action_mode == "relative"
    assert model.supports_compact_infer is False


def test_model_rejects_unknown_action_mode_before_loading_assets():
    with pytest.raises(ValueError, match="must be 'relative' or 'absolute'"):
        opendm_model.Model(
            {
                "env_cfg_type": "arx_x5",
                "action_type": "joint",
                "history_enabled": True,
                "model_action_mode": "velocity",
            }
        )


def test_launcher_exposes_single_frame_overrides():
    launcher = (
        Path(__file__).resolve().parents[1]
        / "policy"
        / "OpenDM"
        / "setup_eval_policy_server.sh"
    ).read_text()

    assert 'OPENDM_ACTION_MODE:-' in launcher
    assert 'OVERRIDES+=(model_action_mode="${OPENDM_ACTION_MODE}")' in launcher
    assert 'OPENDM_PRECISION_MODE:-' in launcher
    assert 'OVERRIDES+=(precision_mode="${OPENDM_PRECISION_MODE}")' in launcher


def test_model_enables_compact_infer_only_for_single_frame(monkeypatch, tmp_path):
    monkeypatch.setattr(
        opendm_model,
        "_resolve_model_assets",
        lambda _: (tmp_path, tmp_path / "norm_stats.json"),
    )
    monkeypatch.setattr(opendm_model.Model, "_initialize_runtime", lambda *args: None)

    model = opendm_model.Model(
        {
            "env_cfg_type": "arx_x5",
            "action_type": "joint",
            "history_enabled": False,
        }
    )

    assert model.history_enabled is False
    assert model.supports_compact_infer is True
    assert model.action_steps == 50


def test_single_frame_observations_do_not_touch_history_buffers():
    model = opendm_model.Model.__new__(opendm_model.Model)
    model.history_enabled = False
    model._history_by_env = {}
    model._history_sequence_by_env = {}

    observation = {
        "vision": {
            "cam_head": {
                "color": np.zeros((4, 4, 3), dtype=np.uint8),
            }
        }
    }
    model._append_history_observation(0, observation)
    images = model._history_images_for(0)

    assert model._history_by_env == {}
    assert model._history_sequence_by_env == {}
    assert images == []


@pytest.mark.parametrize("model_action_dim", [14, 22])
def test_single_frame_predict_uses_current_images_without_history(
    monkeypatch,
    model_action_dim,
):
    captured = {}
    expected_model_chunk = np.arange(
        50 * model_action_dim,
        dtype=np.float32,
    ).reshape(50, model_action_dim)

    class FakeInference:
        diffusion_steps = 10
        output_action_mask = [True] * 14 + [False] * (model_action_dim - 14)

        def _predict(self, payload):
            captured.update(payload)
            return expected_model_chunk

    monkeypatch.setattr(
        opendm_model,
        "pack_robot_state",
        lambda *args, **kwargs: np.arange(14, dtype=np.float32),
    )
    monkeypatch.setattr(
        opendm_model,
        "unpack_robot_state",
        lambda action, *args, **kwargs: {"joint": np.asarray(action)},
    )

    model = opendm_model.Model.__new__(opendm_model.Model)
    model.history_enabled = False
    model.action_type = "joint"
    model.robot_action_dim_info = {"arm_dim": [6, 6], "ee_dim": [1, 1]}
    model.action_dim = 14
    model.model_action_dim = model_action_dim
    model.model_state_dim = 22
    model.action_steps = 25
    model.default_prompt = "perform the task"
    model.model_cfg = {}
    model._robot_type = "Dual ARX5"
    model._state_desc = []
    model._inference = FakeInference()

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    observation = {
        "vision": {
            "cam_head": {"color": image},
            "cam_left_wrist": {"color": image},
            "cam_right_wrist": {"color": image},
        }
    }
    actions = model._predict(observation)

    assert len(actions) == 25
    np.testing.assert_array_equal(
        actions[0]["joint"],
        expected_model_chunk[0, :14],
    )
    np.testing.assert_array_equal(
        actions[-1]["joint"],
        expected_model_chunk[24, :14],
    )
    assert captured["history_images"] == []
    np.testing.assert_array_equal(
        captured["state"],
        np.pad(
            np.arange(14, dtype=np.float32),
            (0, model_action_dim - 14),
        ),
    )
    np.testing.assert_array_equal(
        captured["meta_data"]["valid_dim_mask"],
        np.array([True] * 14 + [False] * 8),
    )
    assert set(captured) == {
        "images_1",
        "images_2",
        "images_3",
        "history_images",
        "prompt",
        "state",
        "meta_data",
    }


def test_normalization_dimensions_are_detected_from_stats(tmp_path):
    stats_path = tmp_path / "norm_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "norm_stats": {
                    "action": {
                        "q01": [0.0] * 22,
                        "q99": [1.0] * 22,
                        "mean": [0.5] * 22,
                        "std": [0.1] * 22,
                    },
                    "state": {
                        "q01": [0.0] * 22,
                        "q99": [1.0] * 22,
                        "mean": [0.5] * 22,
                        "std": [0.1] * 22,
                    },
                }
            }
        )
    )

    assert opendm_model._normalization_dimensions(stats_path) == (22, 22)


def test_normalization_dimensions_reject_inconsistent_vectors(tmp_path):
    stats_path = tmp_path / "norm_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "norm_stats": {
                    "action": {"q01": [0.0] * 22, "q99": [1.0] * 14},
                    "state": {"q01": [0.0] * 22, "q99": [1.0] * 22},
                }
            }
        )
    )

    with pytest.raises(ValueError, match="vectors disagree on dimension"):
        opendm_model._normalization_dimensions(stats_path)


def test_configured_model_dimension_is_validation_only():
    assert opendm_model._validate_configured_dimension(
        {"model_action_dim": 22},
        "model_action_dim",
        22,
    ) == 22
    with pytest.raises(ValueError, match="does not match norm_stats dimension"):
        opendm_model._validate_configured_dimension(
            {"model_action_dim": 14},
            "model_action_dim",
            22,
        )


def test_model_action_mask_keeps_shared_layout_padding_inactive():
    if str(opendm_model._UPSTREAM_ROOT) not in sys.path:
        sys.path.insert(0, str(opendm_model._UPSTREAM_ROOT))
    from opendm.exp.dm05_exp import _resolve_output_action_mask

    np.testing.assert_array_equal(
        _resolve_output_action_mask(22, [True] * 14 + [False] * 8),
        np.array([True] * 14 + [False] * 8),
    )
    with pytest.raises(ValueError, match="output_action_dim=22 entries"):
        _resolve_output_action_mask(22, [True] * 14)


def test_relative_output_keeps_both_grippers_absolute():
    if str(opendm_model._UPSTREAM_ROOT) not in sys.path:
        sys.path.insert(0, str(opendm_model._UPSTREAM_ROOT))
    from opendm.constants.robot import RobotStateDesc
    from opendm.data.transforms import ActionAbsolute

    state = np.arange(22, dtype=np.float32) + 10.0
    relative_action = np.full((2, 22), 0.5, dtype=np.float32)
    relative_action[:, 6] = 0.2
    relative_action[:, 13] = 0.8
    state_desc = (
        [RobotStateDesc.JOINT] * 6
        + [RobotStateDesc.GRIPPER]
        + [RobotStateDesc.JOINT] * 6
        + [RobotStateDesc.GRIPPER]
    )

    converted = ActionAbsolute()(
        {
            "state": state,
            "action": relative_action.copy(),
            "meta_data": {"state_desc": state_desc},
        }
    )["action"]

    joint_indices = [*range(6), *range(7, 13)]
    np.testing.assert_allclose(
        converted[:, joint_indices],
        np.broadcast_to(state[joint_indices] + 0.5, (2, len(joint_indices))),
    )
    np.testing.assert_allclose(converted[:, 6], 0.2)
    np.testing.assert_allclose(converted[:, 13], 0.8)


def test_model_rejects_unknown_robot_before_loading_repo_dimensions(monkeypatch):
    def fail_if_called(_):
        raise AssertionError("dimension lookup should not run for unknown profiles")

    monkeypatch.setattr(
        opendm_model,
        "get_repo_robot_action_dim_info",
        fail_if_called,
    )

    with pytest.raises(ValueError, match="arx_x5, piper, piper_x"):
        opendm_model.Model(
            {
                "env_cfg_type": "unknown_robot",
                "action_type": "joint",
            }
        )


@pytest.mark.parametrize("env_cfg_type", sorted(EXPECTED_ROBOT_TYPES))
def test_opendm_robot_profiles_use_repo_robot_dimensions(env_cfg_type):
    repo_root = Path(__file__).resolve().parents[1]
    robot_info = json.loads(
        (repo_root / "utils" / "robot" / "_robot_info.json").read_text()
    )

    assert robot_info[env_cfg_type] == {
        "arm_dim": [6, 6],
        "ee_dim": [1, 1],
    }
    assert get_repo_robot_action_dim_info(env_cfg_type) == robot_info[env_cfg_type]
