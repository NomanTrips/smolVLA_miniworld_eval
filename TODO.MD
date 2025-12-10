# TODO

## MiniWorld closed-loop eval

- [ ] Create `eval_miniworld.py` (main entrypoint).
- [ ] Support CLI args similar to data-collection config, e.g.:
  - `--env-name MiniWorld-FourRooms-v0`
  - `--fullscreen`
  - `--mouse-sensitivity 0.005`
  - `--hide-hud`
  - `--task "Find the red cube"`

## Core rollout loop

- [ ] Implement rollout loop:
  1. Start environment (Gym/MiniWorld).
  2. Get first frame (from observation or `env.render()`).
  3. Build model batch (image + optional state + task text).
  4. Run policy inference to get action.
  5. Apply action to env via `env.step(action)`.
  6. Repeat until:
     - goal reached (terminated), OR
     - time limit hit (default: 60 seconds), OR
     - max steps reached (backup limiter)

## Recording / outputs

- [ ] Save a video of the episode (env frames with the model’s actions applied).
  - Prefer Gymnasium `RecordVideo` wrapper if compatible.
  - Otherwise fall back to manual `cv2.VideoWriter`.

## Correctness + reproducibility

- [ ] Ensure eval uses the same dataset stats / normalization mapping as training.
- [ ] Print the resolved `miniworld.__file__` path at startup to verify we’re using the local fork.
- [ ] Add seeding support for deterministic runs.
