"""Command-line entrypoint for SmolVLA MiniWorld evaluation.

This script mirrors the planned CLI documented in README.MD and TODO.MD.
It currently prepares runtime configuration, validates paths, and initializes
logging for a MiniWorld rollout.
"""
from __future__ import annotations

import argparse
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import gymnasium as gym
import miniworld
import numpy as np
import torch
from lerobot.processor.core import TransitionKey
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


@dataclass
class Normalizer:
    """Simple tensor normalizer/unnormalizer using mean and std statistics."""

    mean: torch.Tensor
    std: torch.Tensor

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std

    def unnormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.std + self.mean

    def to(self, device: torch.device) -> Normalizer:
        return Normalizer(self.mean.to(device), self.std.to(device))


def _load_stats_section(section: dict) -> dict[str, Normalizer]:
    normalizers: dict[str, Normalizer] = {}
    for key, values in section.items():
        if not isinstance(values, dict) or "mean" not in values or "std" not in values:
            logging.warning("Skipping malformed stats entry for %s", key)
            continue
        mean = torch.tensor(values["mean"], dtype=torch.float32)
        std = torch.tensor(values["std"], dtype=torch.float32).clamp(min=1e-6)
        normalizers[key] = Normalizer(mean=mean, std=std)
    return normalizers


def load_dataset_statistics_from_meta(stats: dict) -> tuple[dict[str, Normalizer], dict[str, Normalizer]]:
    input_section = stats.get("inputs") or {}
    output_section = stats.get("outputs") or {}

    input_normalizers = _load_stats_section(input_section)
    output_normalizers = _load_stats_section(output_section)

    logging.info(
        "Loaded dataset statistics from LeRobot meta (inputs=%s, outputs=%s)",
        sorted(input_normalizers.keys()),
        sorted(output_normalizers.keys()),
    )
    return input_normalizers, output_normalizers


class ManualVideoRecorder:
    """Lazy video writer that mirrors RecordVideo when unavailable."""

    def __init__(self, video_path: Path, fps: float) -> None:
        self.video_path = video_path
        self.fps = fps
        self.writer: cv2.VideoWriter | None = None

    def write(self, frame_rgb: torch.Tensor | "np.ndarray", overlay_text: str | None = None) -> None:  # type: ignore[name-defined]
        if isinstance(frame_rgb, torch.Tensor):
            frame_rgb = frame_rgb.detach().cpu().numpy()

        frame_bgr = frame_rgb
        if overlay_text:
            frame_bgr = frame_bgr.copy()
            cv2.putText(
                frame_bgr,
                overlay_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)

        if self.writer is None:
            height, width, _ = frame_bgr.shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(str(self.video_path), fourcc, self.fps, (width, height))
        self.writer.write(frame_bgr)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def get_env_fps(env: gym.Env, default: float = 30.0) -> float:
    metadata = getattr(env, "metadata", {}) or {}
    fps = metadata.get("render_fps")
    try:
        return float(fps) if fps else default
    except (TypeError, ValueError):
        return default


def setup_recording(env: gym.Env, outdir: Path) -> tuple[gym.Env, ManualVideoRecorder | None]:
    """Configure video recording via RecordVideo when possible or manual writer fallback."""

    video_dir = outdir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    render_modes = getattr(env, "metadata", {}).get("render_modes", []) or []
    if "rgb_array" in render_modes:
        try:
            from gymnasium.wrappers import RecordVideo

            logging.info("Enabling RecordVideo. Videos will be saved under %s", video_dir)
            return RecordVideo(env, video_folder=str(video_dir), name_prefix="episode"), None
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("RecordVideo unavailable or failed (%s). Falling back to manual writer.", exc)
    else:
        logging.info("RecordVideo unsupported by env metadata; using manual capture.")

    video_path = video_dir / "episode.mp4"
    fps = get_env_fps(env)
    logging.info("Manual video writer configured at %s (fps=%s)", video_path, fps)
    return env, ManualVideoRecorder(video_path, fps)


def existing_path(path_str: str) -> Path:
    """Argparse helper to require an existing file or directory path."""

    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Policy path does not exist: {path}")
    return path


def dir_path(path_str: str) -> Path:
    """Argparse helper for output directory creation."""
    path = Path(path_str).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise argparse.ArgumentTypeError(f"Output path exists and is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_mapping(pairs: Optional[list[str]]) -> dict[str, str]:
    """Parse CLI KEY=VALUE pairs into a dictionary."""

    mapping: dict[str, str] = {}
    if not pairs:
        return mapping

    for pair in pairs:
        if "=" not in pair:
            raise argparse.ArgumentTypeError("Normalization mapping entries must be KEY=VALUE pairs")
        key, value = pair.split("=", maxsplit=1)
        if not key or not value:
            raise argparse.ArgumentTypeError("Normalization mapping entries must include both key and value")
        mapping[key] = value
    return mapping


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Closed-loop evaluation of a SmolVLA policy in MiniWorld",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--env-name",
        required=True,
        help="MiniWorld environment ID to load (e.g., MiniWorld-FourRooms-v0)",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Render the environment in fullscreen mode",
    )
    parser.add_argument(
        "--mouse-sensitivity",
        type=float,
        default=0.005,
        help="Mouse sensitivity for camera control",
    )
    parser.add_argument(
        "--hide-hud",
        action="store_true",
        help="Hide on-screen HUD elements",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Task description to condition the policy (e.g., 'Find the red cube')",
    )
    parser.add_argument(
        "--policy",
        required=True,
        type=existing_path,
        help="Path to the trained SmolVLA policy checkpoint directory or file",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset repo ID or local path used for normalization",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        type=dir_path,
        help="Directory to store evaluation outputs (videos, logs)",
    )
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=60,
        help="Maximum allowed episode duration in seconds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed applied to Python, NumPy, Torch, and the environment",
    )
    parser.add_argument(
        "--input-features",
        nargs="+",
        default=None,
        help="Optional override for policy input feature names (defaults to checkpoint config)",
    )
    parser.add_argument(
        "--output-features",
        nargs="+",
        default=None,
        help="Optional override for policy output feature names (defaults to checkpoint config)",
    )
    parser.add_argument(
        "--normalization-mapping",
        nargs="*",
        default=None,
        metavar="KEY=VALUE",
        help="Mapping from policy feature names to dataset statistic keys (e.g., image=rgb)",
    )

    return parser.parse_args(args=args)


def setup_logging(outdir: Path) -> None:
    """Initialize root logger writing to console and file in the output directory."""
    log_path = outdir / "eval.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        ],
        force=True,
    )
    logging.info("Logging initialized. Writing to %s", log_path)


def seed_everything(seed: int) -> None:
    logging.info("Seeding Python, NumPy, Torch with seed=%s", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    setup_logging(args.outdir)

    logging.info("Resolved miniworld module path: %s", Path(miniworld.__file__).resolve())
    seed_everything(args.seed)

    logging.info("Starting MiniWorld eval")
    logging.info("Environment: %s", args.env_name)
    logging.info("Fullscreen: %s", args.fullscreen)
    logging.info("Mouse sensitivity: %s", args.mouse_sensitivity)
    logging.info("Hide HUD: %s", args.hide_hud)
    logging.info("Task: %s", args.task)
    logging.info("Policy: %s", args.policy)
    logging.info("Dataset: %s", args.dataset)
    logging.info("Output dir: %s", args.outdir)
    logging.info("Max seconds: %s", args.max_seconds)
    logging.info("Seed: %s", args.seed)
    logging.info("Input features override: %s", args.input_features)
    logging.info("Output features override: %s", args.output_features)
    logging.info("Normalization mapping override: %s", args.normalization_mapping)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    ds = LeRobotDataset(repo_id=args.dataset, force_cache_sync=True)
    dataset_root = next(
        (
            Path(value)
            for value in (
                getattr(ds, attr, None)
                for attr in ("data_dir", "repo_path", "cache_dir", "root", "dataset_dir")
            )
            if value
        ),
        None,
    )
    if dataset_root:
        logging.info("LeRobot dataset cached at: %s", dataset_root)
    else:
        logging.info("LeRobot dataset cache path not available on dataset object")

    input_normalizers, output_normalizers = load_dataset_statistics_from_meta(ds.meta.stats)
    if input_normalizers:
        logging.info("Input normalizers available for: %s", ", ".join(sorted(input_normalizers)))
    if output_normalizers:
        logging.info("Output unnormalizers available for: %s", ", ".join(sorted(output_normalizers)))
    logging.info("Dataset statistics loaded from dataset metadata")

    policy = SmolVLAPolicy.from_pretrained(args.policy)
    policy.to(device)
    policy.eval()

    policy_config = getattr(policy, "config", {}) or {}

    def get_config_value(cfg, key, default=None):
        if hasattr(cfg, key):
            value = getattr(cfg, key)
        elif isinstance(cfg, dict):
            value = cfg.get(key, default)
        else:
            value = default
        return value if value is not None else default

    normalization_mapping = parse_mapping(args.normalization_mapping)
    if not normalization_mapping:
        normalization_mapping = get_config_value(policy_config, "normalization_mapping", {})

    policy_input_features = get_config_value(policy_config, "input_features", {})
    policy_output_features = get_config_value(policy_config, "output_features", {})

    if args.input_features:
        if isinstance(policy_input_features, dict):
            input_features = {k: v for k, v in policy_input_features.items() if k in args.input_features}
        else:
            input_features = {k: None for k in args.input_features}
    else:
        input_features = policy_input_features

    if args.output_features:
        if isinstance(policy_output_features, dict):
            output_features = {k: v for k, v in policy_output_features.items() if k in args.output_features}
        else:
            output_features = {k: None for k in args.output_features}
    else:
        output_features = policy_output_features

    combined_features = {}
    if isinstance(input_features, dict):
        combined_features.update(input_features)
    if isinstance(output_features, dict):
        combined_features.update(output_features)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=str(args.policy),
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "normalizer_processor": {
                "stats": ds.meta.stats,
                "features": combined_features or None,
                "norm_map": normalization_mapping or None,
            },
            "tokenizer_processor": {
                "task_key": "task",
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": ds.meta.stats,
                "features": output_features or None,
                "norm_map": normalization_mapping or None,
            }
        },
    )

    logging.info("Policy input features: %s", input_features)
    logging.info("Policy output features: %s", output_features)
    logging.info("Using normalization mapping: %s", normalization_mapping)

    def move_to_device(batch: dict | torch.Tensor | list | tuple):
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        if isinstance(batch, dict):
            return {k: move_to_device(v) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            return type(batch)(move_to_device(v) for v in batch)
        return batch

    def summarize_action(value):
        """Make action outputs logging-friendly while preserving numeric detail."""

        def to_serializable(item):
            if isinstance(item, torch.Tensor):
                item = item.detach().cpu().numpy()
            if isinstance(item, np.ndarray):
                return item.tolist()
            if isinstance(item, (list, tuple)):
                return [to_serializable(x) for x in item]
            if isinstance(item, dict):
                return {k: to_serializable(v) for k, v in item.items()}
            try:
                return float(item)
            except (TypeError, ValueError):
                return item

        return to_serializable(value)

    def build_policy_transition(observation, render_frame):
        def to_image_tensor(frame):
            if isinstance(frame, torch.Tensor):
                tensor = frame
            else:
                tensor = torch.from_numpy(np.asarray(frame))
            if tensor.ndim == 3:  # HWC -> CHW
                tensor = tensor.permute(2, 0, 1)
            if tensor.ndim == 3:  # add batch
                tensor = tensor.unsqueeze(0)
            tensor = tensor.float()
            if tensor.max().item() > 1.0:
                tensor = tensor / 255.0
            return tensor

        def to_tensor(value):
            if isinstance(value, torch.Tensor):
                tensor = value
            else:
                tensor = torch.tensor(value)
            tensor = tensor.float()
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            return tensor

        feature_names = list(input_features.keys()) if isinstance(input_features, dict) else list(input_features)
        nested_obs: dict[str, object] = {}
        flat_obs: dict[str, object] = {}

        state_source = observation.get("state") if isinstance(observation, dict) else None
        render_tensor = to_image_tensor(render_frame)

        requires_state = any("state" in feature for feature in feature_names)
        state_tensor: torch.Tensor | None = None

        if requires_state:
            if state_source is not None:
                state_tensor = to_tensor(state_source)
            elif env is not None:
                state_components: list[float] = []
                try:
                    unwrapped = env.unwrapped
                    agent = getattr(unwrapped, "agent", unwrapped)
                    pos = getattr(agent, "pos", None)
                    direction = getattr(agent, "dir", getattr(agent, "direction", None))
                    pitch = getattr(agent, "pitch", None)

                    if pos is not None:
                        state_components.extend(np.asarray(pos, dtype=np.float32).flatten().tolist())
                    if direction is not None:
                        state_components.append(float(direction))
                    if pitch is not None:
                        state_components.append(float(pitch))

                    if state_components:
                        state_tensor = torch.tensor([state_components], dtype=torch.float32)
                        logging.info("Synthesized observation.state from env agent attributes (pos/dir/pitch)")
                except Exception as exc:  # pragma: no cover - best effort debug info
                    logging.debug("Unable to extract env agent state: %s", exc)

            if state_tensor is None:
                state_tensor = torch.zeros((1, 5), dtype=torch.float32)
                logging.info("Observation missing state; using zero state tensor shaped %s", tuple(state_tensor.shape))

        for feature in feature_names:
            nested_key = feature.split("observation.", 1)[-1] if feature.startswith("observation.") else feature
            value = None
            if isinstance(observation, dict):
                value = observation.get(feature)
                if value is None and feature.startswith("observation."):
                    value = observation.get(nested_key)

            if value is None and any(token in feature for token in ("image", "rgb", "pixels")):
                value = render_tensor
            if value is None and "state" in feature and state_tensor is not None:
                value = state_tensor

            if value is None:
                continue

            flat_obs[feature] = value
            nested_obs[nested_key] = value

        if state_tensor is not None:
            nested_obs.setdefault("state", state_tensor)
            flat_obs.setdefault("observation.state", state_tensor)

        if not nested_obs:
            default_key = "image" if "image" in feature_names else feature_names[0] if feature_names else "image"
            nested_default_key = default_key.split("observation.", 1)[-1] if default_key.startswith("observation.") else default_key
            nested_obs[nested_default_key] = render_tensor
            flat_obs[default_key if default_key.startswith("observation.") else f"observation.{default_key}"] = render_tensor

        transition = {
            "observation": nested_obs,
            TransitionKey.OBSERVATION: nested_obs,
            "task": args.task,
            TransitionKey.COMPLEMENTARY_DATA: {"task": args.task},
        }
        transition.update(flat_obs)
        return transition

    env = None
    manual_recorder: ManualVideoRecorder | None = None

    obs_width = 512
    obs_height = 512

    logging.info(
        "Configuring MiniWorld render at %sx%s (HUD %s, mouse sensitivity=%s)",
        obs_width,
        obs_height,
        "hidden" if args.hide_hud else "visible",
        args.mouse_sensitivity,
    )

    try:
        env = gym.make(
            args.env_name,
            render_mode="rgb_array",
            show_hud=not args.hide_hud,
            obs_width=obs_width,
            obs_height=obs_height,
        )

        if hasattr(env, "mouse_sensitivity"):
            setattr(env, "mouse_sensitivity", args.mouse_sensitivity)

        env.action_space.seed(args.seed)
        if hasattr(env, "observation_space"):
            env.observation_space.seed(args.seed)

        env, manual_recorder = setup_recording(env, args.outdir)

        obs, info = env.reset(seed=args.seed)
        render = env.render()
        logging.info("Environment reset: observation keys=%s", list(obs.keys()) if isinstance(obs, dict) else type(obs))

        if manual_recorder:
            manual_recorder.write(render, overlay_text="reset")

        done = False
        truncated = False
        step_idx = 0
        max_steps = 1000
        start_time = time.time()
        termination_reason = "time_limit"

        with torch.no_grad():
            while not (done or truncated):
                if time.time() - start_time > args.max_seconds:
                    termination_reason = "wall_clock_timeout"
                    logging.info("Stopping due to wall-clock timeout after %s seconds", args.max_seconds)
                    break
                if step_idx >= max_steps:
                    termination_reason = "step_limit"
                    logging.info("Stopping due to max step limit %s", max_steps)
                    break

                transition = build_policy_transition(obs, render)
                processed_inputs = preprocessor(transition)
                processed_inputs = move_to_device(processed_inputs)

                action_output = policy.select_action(processed_inputs)
                logging.info("Model raw action output: %s", summarize_action(action_output))

                action_post = postprocessor(action_output)
                logging.info("Postprocessed action: %s", summarize_action(action_post))
                action_value = (
                    action_post["action"] if isinstance(action_post, dict) and "action" in action_post else action_post
                )

                if isinstance(action_value, torch.Tensor):
                    action = action_value.squeeze().detach().cpu().numpy()
                    if action.shape == ():
                        action = int(action.item())
                else:
                    action = action_value

                obs, reward, done, truncated, info = env.step(action)
                render = env.render()
                action_text = f"action: {action}"
                if manual_recorder:
                    manual_recorder.write(render, overlay_text=action_text)
                logging.info("Step %s: action=%s reward=%s done=%s truncated=%s", step_idx, action, reward, done, truncated)
                step_idx += 1

        if done:
            termination_reason = "terminated"
        elif truncated and termination_reason == "time_limit":
            termination_reason = "environment_truncated"

        logging.info("Episode finished after %s steps. Reason: %s", step_idx, termination_reason)
    finally:
        if manual_recorder:
            manual_recorder.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
