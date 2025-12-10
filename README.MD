# smolVLA MiniWorld Eval

Simple closed-loop evaluation for a trained **SmolVLA policy** inside **MiniWorld**.

The eval:
- spins up a MiniWorld environment
- runs a closed-loop control rollout (frame -> model -> action -> env)
- stops when the goal is reached or a time limit is hit
- saves a video of the episode

## Setup

Clone custom for of Miniworld:

https://github.com/NomanTrips/Miniworld.git

```bash
pip uninstall -y miniworld
pip install -e /path/to/custom/miniworld-fork
```

Install LeRobot editable (recommended while iterating):

```bash
pip install -e /path/to/lerobot
```

## Run (planned CLI)

Example target CLI:

```bash
python eval_miniworld.py \
  --env-name MiniWorld-FourRooms-v0 \
  --fullscreen \
  --mouse-sensitivity 0.005 \
  --hide-hud \
  --task "Find the red cube" \
  --policy /path/to/smolvla/checkpoint \
  --dataset /path/to/dataset-or-hf-repo \
  --outdir outputs/eval_runs \
  --max-seconds 60
```

## What this repo is for

This repo is intentionally small:
- one eval entrypoint script
- minimal glue code to build SmolVLA input batches
- straightforward video recording of the rollout
