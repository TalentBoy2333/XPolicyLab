"""Robot metadata supported by the OpenDM XPolicyLab adapter."""

from dataclasses import dataclass
import json
from pathlib import Path


_ROBOT_INFO_PATH = (
    Path(__file__).resolve().parents[2] / "utils" / "robot" / "_robot_info.json"
)


@dataclass(frozen=True)
class RobotProfile:
    """OpenDM prompt metadata for one XPolicyLab environment config."""

    robot_type: str


ROBOT_PROFILES = {
    "arx_x5": RobotProfile(robot_type="Dual ARX5"),
    "piper": RobotProfile(robot_type="Aloha"),
    "piper_x": RobotProfile(robot_type="Aloha Piper X"),
}


def get_repo_robot_action_dim_info(env_cfg_type: str) -> dict[str, list[int]]:
    """Read action dimensions directly from this checkout's robot registry."""

    with _ROBOT_INFO_PATH.open("r", encoding="utf-8") as robot_info_file:
        robot_info = json.load(robot_info_file)
    try:
        dimensions = robot_info[env_cfg_type]
    except KeyError as exc:
        supported = ", ".join(sorted(robot_info))
        raise KeyError(
            f"env_cfg_type {env_cfg_type!r} is absent from {_ROBOT_INFO_PATH}; "
            f"available values: {supported}"
        ) from exc
    return {
        "arm_dim": list(dimensions["arm_dim"]),
        "ee_dim": list(dimensions["ee_dim"]),
    }
