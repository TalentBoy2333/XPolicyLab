# OpenDM

**Contributor:** XPolicyLab Team | **Paper:** DM0.5 technical report | **arXiv:** TBD | **Original code:** [Dexmal OpenDM](https://gitlab.dexmal.com/robotics/opendm)

This eval-only adapter runs RoboDojo DM0.5 single-frame and memory checkpoints with the XPolicyLab WebSocket policy server. The existing `mm-robodojo` history/absolute-action simulation profile remains the default and is unchanged. Single-frame checkpoints for ARX5 (`arx_x5`), Piper (`piper`), and Piper X (`piper_x`) use the isolated `deploy_single_frame.yml` profile. The pinned upstream OpenDM source is vendored under `opendm/`; see [opendm/UPSTREAM.md](opendm/UPSTREAM.md).

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

The default environment uses Python 3.10, PyTorch 2.11.0, torchvision 0.26.0, and CUDA 12.8 wheels:

```bash
cd XPolicyLab/policy/OpenDM
bash install.sh
conda activate opendm
```

PyTorch SDPA is used by default, so `flash-attn` is not required. Set `OPENDM_INSTALL_FLASH_ATTN=1` before running `install.sh` only if you explicitly switch the vision attention implementation.

## Data Processing

Not included in this eval-only release. RoboDojo observations are consumed online through XPolicyLab; no dataset conversion is needed for evaluation.

## Training

Training is not included in this eval-only release.

## Model Assets

| Item | Value |
| --- | --- |
| Model name | `mm-robodojo` |
| Version | `v1` |
| Repository | [TalentBoy/mm-robodojo](https://huggingface.co/TalentBoy/mm-robodojo) |

Download `v1` from Hugging Face into the local checkpoint directory:

```bash
huggingface-cli download TalentBoy/mm-robodojo \
  --revision v1 \
  --local-dir ./checkpoints/mm-robodojo
```

The adapter expects the complete Hugging Face-style directory at:

```text
policy/OpenDM/checkpoints/mm-robodojo/
```

It must include model/config, processor/tokenizer files, and the matching `norm_stats.json`. The adapter also accepts a checkpoint directory as `ckpt_name` or through `MODEL_PATH`.

Runtime profiles:

| Profile | Deploy config | `env_cfg_type` | Action mode | History |
| --- | --- | --- | --- | --- |
| `mm-robodojo` simulation | `deploy.yml` | `arx_x5` | absolute | 20 slots at 1 Hz |
| Single-frame/non-memory | `deploy_single_frame.yml` | `arx_x5`, `piper`, `piper_x` | relative checkpoint -> absolute targets | disabled |
| Piper checkpoint-80000 BF16 parity | `deploy_real_piper_single_frame_bf16.yml` | `piper` | relative checkpoint -> absolute targets | disabled |

Shared model contract:

| Item | Value |
| --- | --- |
| Benchmark | RoboDojo simulation or real-robot client |
| Robots | Dual ARX5 (`arx_x5`), Aloha/Piper (`piper`), Aloha Piper X (`piper_x`) |
| Cameras | head, left wrist, right wrist RGB |
| State/action | 14-D: left arm 6, left gripper 1, right arm 6, right gripper 1 |
| Action semantics | absolute joint targets |

The vendored OpenDM code remains under Apache-2.0. The `mm-robodojo` weights are distributed under the repository's MM-RoboDojo Community License, together with the upstream DM05/Gemma terms identified in its `NOTICE` and `THIRD_PARTY_NOTICES.md` files.

## Evaluation

Run one RoboDojo simulator task from this policy directory:

```bash
EVAL_ENV_TYPE=sim bash eval.sh \
  RoboDojo <task_name> mm-robodojo \
  arx_x5 joint 0 <policy_gpu_id> <env_gpu_id> opendm <robodojo_env>
```

For example, using policy GPU 0 and simulator GPU 1:

```bash
EVAL_ENV_TYPE=sim bash eval.sh \
  RoboDojo stack_bowls mm-robodojo \
  arx_x5 joint 0 0 1 opendm RoboDojo
```

The policy server can also be started independently with `setup_eval_policy_server.sh`; use the standard split-machine workflow linked above.

### Single-frame deployment

Select `deploy_single_frame.yml` and point `ckpt_name` at the complete
single-frame checkpoint directory. The directory must include its matching
`norm_stats.json`. The profile defaults to `model_action_mode: relative`: arm
deltas are converted back to absolute joint targets using the current state,
while gripper targets remain absolute. Set `OPENDM_ACTION_MODE=absolute` only
for a checkpoint that was actually trained with absolute actions.

```bash
OPENDM_DEPLOY_CONFIG=deploy_single_frame.yml \
OPENDM_ACTION_MODE=relative \
bash setup_eval_policy_server.sh \
  RoboDojo <task_name> <single-frame-checkpoint-path> \
  <arx_x5|piper|piper_x> joint 0 <policy_gpu_id> opendm <port> 0.0.0.0
```

The compatibility profile generates the checkpoint's full 50-step horizon but
returns the first 25 actions before the client replans. Override
`OPENDM_ACTION_STEPS` only when a client intentionally uses a different
replanning cadence. Single-frame models retain the established compact
`INFER` protocol; they may also use explicit `CALL update_obs -> CALL
get_action`.

For the 14-D relative Piper checkpoint at
`/mlp_vepfs/share/zrt/projects/dev/opendm/user_checkpoints/dm05_robodojo_real_piper/checkpoint-80000`,
use the checkpoint-specific `deploy_real_piper_single_frame_bf16.yml`. It pins
the checkpoint and normalization paths, Piper metadata, the 14-D model space,
and `diffusion_noise_seed: 0` for repeatable parity checks:

```bash
OPENDM_DEPLOY_CONFIG=deploy_real_piper_single_frame_bf16.yml \
bash setup_eval_policy_server.sh \
  RoboDojo <task_name> \
  /mlp_vepfs/share/zrt/projects/dev/opendm/user_checkpoints/dm05_robodojo_real_piper/checkpoint-80000 \
  piper joint 0 <policy_gpu_id> opendm <port> 0.0.0.0
```

The generic `deploy_single_frame.yml` profile pins `precision_mode: mixed`:
FP32 model weights, BF16 autocast, and a forced FP32 action expert. The
checkpoint-specific named `precision_mode: bf16` casts the complete model to BF16, keeps the
diffusion state in model dtype, disables the redundant BF16 autocast wrapper,
and leaves the action expert in BF16. This matches the established
single-frame reference path. Other existing profiles without `precision_mode`
keep their independent precision knobs unchanged. A named mode rejects
conflicting low-level dtype settings instead of
silently running a different numerical path. The launcher can override the
selection with `OPENDM_PRECISION_MODE`.

The same checkpoint profile also sets `prompt_format: reference` and
`norm_clip_to_bounds: true` so prompt tokens and out-of-quantile mock states
match the reference server. Other zrt profiles retain the refined prompt
format and unclipped normalization defaults.

Each OpenDM policy-server instance currently supports one active RoboDojo
evaluation client. Although observation and history state is keyed by
`(evaluation_id, env_idx)`, RoboDojo environment reset still invokes a
server-wide policy reset. Do not share one policy-server instance between
concurrent evaluation clients; start a separate instance for each evaluator.

## Configuration

The runtime settings required by `mm-robodojo` v1 remain fixed in [deploy.yml](deploy.yml). Single-frame deployments use [deploy_single_frame.yml](deploy_single_frame.yml), which is kept separate so their settings cannot alter the simulation defaults. `MODEL_PATH` and `NORM_STATS_PATH` may be used to select matching local model assets.

The adapter derives model-space dimensions from the selected
`norm_stats.json`; `model_action_dim` and `model_state_dim` are optional
validation-only YAML fields. Current single-frame checkpoints may use 14-D or
22-D action and state statistics. OpenDM generates,
denormalizes, and (for relative checkpoints) converts actions to absolute
targets in the checkpoint's own action space, then returns only the 14-D robot
wire layout (left 6+1, right 6+1). No client protocol change is required.
For relative checkpoints only the twelve arm-joint dimensions receive
`state + delta`; the left and right gripper dimensions (indices 6 and 13) stay
absolute. In the 22-D shared model layout, dimensions 14–21 remain masked and
are never treated as active robot actions.

Action dimensions for the three OpenDM robot profiles come directly from this
checkout's `utils/robot/_robot_info.json`.

## Limitations

- Real-robot transport and safety controls are outside the policy adapter.
- `arx_x5`, `piper`, and `piper_x` are supported with `action_type=joint` only.
- All three clients must retain the 14-D dual-arm wire layout (left 6+1, right 6+1); checkpoint model space may be wider and is detected from its normalization statistics.
- The three real-robot checkpoints and hardware clients are not bundled here; end-to-end hardware validation is still required.
- Batch API calls are processed serially by the model adapter.
- One active evaluation client is supported per policy-server instance; concurrent evaluators require separate server instances.
- The checkpoint checksum, GPU-memory requirement, and benchmark score are pending.
- Request logging, cloud deployment, and distributed evaluation are intentionally outside the public policy adapter.
