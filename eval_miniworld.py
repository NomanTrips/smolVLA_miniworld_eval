"""Command-line entrypoint for SmolVLA MiniWorld evaluation.

This script mirrors the planned CLI documented in README.MD and TODO.MD.
It currently prepares runtime configuration, validates paths, and initializes
logging for a MiniWorld rollout.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import gymnasium as gym
import torch


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


def existing_file(path_str: str) -> Path:
    """Argparse helper to require an existing file path."""
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Policy path does not exist or is not a file: {path}")
    return path


def existing_path(path_str: str) -> Path:
    """Argparse helper to require an existing path (file or directory)."""
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Dataset path does not exist: {path}")
    return path


def dir_path(path_str: str) -> Path:
    """Argparse helper for output directory creation."""
    path = Path(path_str).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise argparse.ArgumentTypeError(f"Output path exists and is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


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
        type=existing_file,
        help="Path to the trained SmolVLA policy checkpoint",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=existing_path,
        help="Path to the dataset or Hugging Face repo used for normalization",
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
    )
    logging.info("Logging initialized. Writing to %s", log_path)


def main() -> None:
    args = parse_args()
    setup_logging(args.outdir)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    model = torch.load(args.policy, map_location=device)
    model.to(device)
    model.eval()

    def prepare_batch(image, state: Optional[torch.Tensor] = None) -> dict:
        """Construct a model-ready batch containing image, optional state, and task text."""

        if isinstance(image, torch.Tensor):
            image_tensor = image
        else:
            image_tensor = torch.from_numpy(image)
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.permute(2, 0, 1)
        image_tensor = image_tensor.float().unsqueeze(0).to(device)

        batch: dict[str, torch.Tensor | str] = {"image": image_tensor, "task": args.task}
        if state is not None:
            batch["state"] = state.to(device)
        return batch

    env = None
    manual_recorder: ManualVideoRecorder | None = None

    try:
        env = gym.make(
            args.env_name,
            render_mode="rgb_array",
            fullscreen=args.fullscreen,
            mouse_sensitivity=args.mouse_sensitivity,
            hide_hud=args.hide_hud,
        )

        env, manual_recorder = setup_recording(env, args.outdir)

        obs, info = env.reset()
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

                state_tensor = None
                if isinstance(obs, dict):
                    image = obs.get("image", render)
                    if "state" in obs:
                        state_tensor = torch.tensor(obs["state"])
                else:
                    image = render

                batch = prepare_batch(image, state_tensor)
                action_tensor = model(batch)
                if isinstance(action_tensor, dict) and "action" in action_tensor:
                    action_tensor = action_tensor["action"]
                if isinstance(action_tensor, torch.Tensor):
                    action = action_tensor.squeeze().detach().cpu().numpy()
                    if action.shape == ():
                        action = int(action.item())
                else:
                    action = action_tensor

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
