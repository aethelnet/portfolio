#!/usr/bin/env python3
"""
Standalone Read-Only Trajectory Exporter for Portfolio Showcase.
Runs lp85 Level 0 with SovereignARC3Agent and exports frames + actions to JSON.
Zero modifications to kaggle_arc.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "kaggle_arc"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Fix if inside apps/portfolio/scripts:
if not (REPO_ROOT / "kaggle_arc").exists():
    REPO_ROOT = REPO_ROOT.parent
    sys.path.insert(0, str(REPO_ROOT / "kaggle_arc"))

os.environ['OPERATION_MODE'] = 'offline'
from arc_agi import Arcade, OperationMode
from my_agent import SovereignARC3Agent

def export():
    env_dir = str(REPO_ROOT / "kaggle_arc/environment_files")
    game_id = "lp85-305b61c3"
    arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=env_dir)

    print(f"[*] Starting simulation for {game_id}...")
    env = arcade.make(game_id)
    agent = SovereignARC3Agent(game_id=game_id)
    obs = env.observation_space

    trajectory = {
        "game_id": "lp85",
        "title": "Dual-Key Affordance & Causal Click Sequencing",
        "grid_size": [64, 64],
        "steps": []
    }

    initial_grid = np.array(obs.frame).tolist()
    frames = []

    for step_idx in range(40):
        act = agent.choose_action(frames, obs)
        data = getattr(act, 'action_data', None)
        if hasattr(data, 'model_dump'):
            data = data.model_dump()
        elif not isinstance(data, dict):
            data = {'x': 32, 'y': 32}

        # Record action info before step
        action_name = act.name if hasattr(act, 'name') else f"ACTION_{act.value}"
        action_desc = f"{action_name}"
        if action_name == 'ACTION6' or act.value == 6:
            action_desc = f"CLICK ({data.get('x', 0)}, {data.get('y', 0)})"

        prev_frame = np.array(obs.frame)
        obs = env.step(act, data=data)
        frames.append(obs)
        curr_frame = np.array(obs.frame)

        # Compute pixel delta
        prev_2d = np.array(prev_frame).squeeze()
        curr_2d = np.array(curr_frame).squeeze()
        if prev_2d.ndim > 2:
            prev_2d = prev_2d[:, :, 0]
        if curr_2d.ndim > 2:
            curr_2d = curr_2d[:, :, 0]

        # Check for level transition or resolution change
        if prev_2d.shape != curr_2d.shape:
            print(f"🎉 Level transition detected! Shape changed from {prev_2d.shape} to {curr_2d.shape}")
            trajectory["steps"].append({
                "step": step_idx,
                "action": action_desc,
                "action_type": action_name,
                "action_data": data,
                "delta_f": int(prev_2d.size),
                "diffs": [],
                "levels_completed": obs.levels_completed,
                "state": obs.state.name,
                "transition": True
            })
            print(f"✅ Success! Level 0 completed in {step_idx + 1} steps. Status: {obs.state.name} | Lvls: {obs.levels_completed}")
            break

        diff_mask = (prev_2d != curr_2d)
        delta_count = int(np.sum(diff_mask))
        changed_pixels = []
        if delta_count > 0:
            ys, xs = np.where(diff_mask)
            for y, x in zip(ys, xs):
                val = curr_2d[y, x]
                changed_pixels.append([int(x), int(y), int(val.item() if hasattr(val, 'item') else val)])

        trajectory["steps"].append({
            "step": step_idx,
            "action": action_desc,
            "action_type": action_name,
            "action_data": data,
            "delta_f": delta_count,
            "diffs": changed_pixels if step_idx > 0 else [],
            "levels_completed": obs.levels_completed,
            "state": obs.state.name
        })

        print(f"   Step {step_idx:02d}: {action_desc} -> Delta F={delta_count:2d} | Lvls={obs.levels_completed}")

        if obs.levels_completed > 0 or obs.state.name in ('WIN', 'GAME_OVER'):
            print(f"✅ Success! Level completed in {step_idx + 1} steps. Status: {obs.state.name}")
            break

    trajectory["initial_frame"] = initial_grid
    trajectory["total_steps"] = len(trajectory["steps"])

    out_path = REPO_ROOT / "apps/portfolio/src/data/arc3_lp85.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(trajectory, f, separators=(',', ':'))

    print(f"🎉 Trajectory saved to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    export()
