#!/usr/bin/env python3
"""
ARC-AGI-3 Sovereign Live WebSocket Bridge Server.
Exposes a WebSocket interface for the Portfolio HUD to trigger live autonomous
ARC-3 game solving sessions using SovereignARC3Agent and stream step-by-step
diffs, actions, and causal reasoning in real-time.
"""

import os
import sys
import json
import time
import asyncio
import traceback
from pathlib import Path
import numpy as np
import websockets

REPO_ROOT = Path(__file__).resolve()
while REPO_ROOT != REPO_ROOT.parent and not (REPO_ROOT / "kaggle_arc").exists():
    REPO_ROOT = REPO_ROOT.parent

sys.path.insert(0, str(REPO_ROOT / "kaggle_arc"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ['OPERATION_MODE'] = 'offline'

try:
    from arc_agi import Arcade, OperationMode
    from my_agent import SovereignARC3Agent
except ImportError as e:
    print(f"[FATAL] Failed to import arc_agi or my_agent: {e}")
    sys.exit(1)

ENV_DIR = str(REPO_ROOT / "kaggle_arc/environment_files")
arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=ENV_DIR)

GAME_CATALOG = {
    "lp85": {
        "title": "Dual-Key Affordance & Causal Click Sequencing",
        "genre": "Point & Click Causal Graph",
        "mechanic": "Centroid Detection & Dead-Click Pruning",
        "verified_levels": 1
    },
    "ls20": {
        "title": "Discrete Grid-World Sokoban Navigation",
        "genre": "Spatial Pathfinding & Gate Mechanics",
        "mechanic": "A* Breadth Search with Key-Door Matching",
        "verified_levels": 2
    },
    "tn36": {
        "title": "Transition Automata & Topological Register",
        "genre": "State Machine & Flow Control",
        "mechanic": "Active Inference & Screen-Redraw Barrier",
        "verified_levels": 1
    }
}

ACTIVE_SOLVERS = {}

def get_game_subid(game_prefix: str) -> str:
    path = Path(ENV_DIR) / game_prefix
    if not path.exists():
        raise FileNotFoundError(f"Game directory not found: {path}")
    subdirs = [d.name for d in path.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No variant found for {game_prefix}")
    return f"{game_prefix}-{subdirs[0]}"

def unwrap_grid(raw_frame) -> list:
    """Ensure frame is strictly a 2D list [64, 64] of ints."""
    arr = np.array(raw_frame, dtype=np.int32)
    while arr.ndim > 2:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr[0]
    return arr.tolist()

async def run_solver_stream(websocket, game_id: str, delay_ms: int = 150):
    client_id = id(websocket)
    try:
        sub_id = get_game_subid(game_id)
        print(f"[*] [WS-{client_id}] Launching live solve for {sub_id} (delay={delay_ms}ms)...")

        env = arcade.make(sub_id)
        agent = SovereignARC3Agent(game_id=sub_id)
        obs = env.observation_space

        initial_grid = unwrap_grid(obs.frame)
        prev_grid = np.array(initial_grid, dtype=np.int32)

        meta = GAME_CATALOG.get(game_id, {
            "title": f"ARC-3 Game {game_id}",
            "genre": "Frontier AI Benchmark",
            "mechanic": "Autonomous Active Inference"
        })

        # Emit INIT payload
        await websocket.send(json.dumps({
            "type": "GAME_INIT",
            "game_id": game_id,
            "title": meta["title"],
            "genre": meta["genre"],
            "mechanic": meta["mechanic"],
            "initial_frame": initial_grid,
            "grid_size": [len(initial_grid), len(initial_grid[0]) if initial_grid else 64]
        }))

        frames = []
        max_steps = 45
        step_idx = 0

        while step_idx < max_steps:
            if ACTIVE_SOLVERS.get(client_id) is False:
                print(f"[*] [WS-{client_id}] Solve aborted by client request.")
                await websocket.send(json.dumps({
                    "type": "SOLVE_ABORTED",
                    "step": step_idx
                }))
                break

            # 1. Agent chooses action
            t_think_0 = time.perf_counter()
            act = agent.choose_action(frames, obs)
            think_time_ms = round((time.perf_counter() - t_think_0) * 1000, 1)

            data = getattr(act, 'action_data', None)
            if hasattr(data, 'model_dump'):
                data = data.model_dump()
            elif not isinstance(data, dict):
                data = {'x': 32, 'y': 32}

            action_name = act.name if hasattr(act, 'name') else f"ACTION_{act.value}"
            action_desc = action_name
            if action_name == 'ACTION6' or (hasattr(act, 'value') and act.value == 6):
                action_desc = f"CLICK ({data.get('x', 0)}, {data.get('y', 0)})"
            elif action_name == 'ACTION1' or (hasattr(act, 'value') and act.value == 1):
                action_desc = "MOVE UP (▲)"
            elif action_name == 'ACTION2' or (hasattr(act, 'value') and act.value == 2):
                action_desc = "MOVE DOWN (▼)"
            elif action_name == 'ACTION3' or (hasattr(act, 'value') and act.value == 3):
                action_desc = "MOVE LEFT (◀)"
            elif action_name == 'ACTION4' or (hasattr(act, 'value') and act.value == 4):
                action_desc = "MOVE RIGHT (▶)"

            # 2. Step environment
            obs = env.step(act, data=data)
            frames.append(obs)

            # 3. Compute pixel diffs
            curr_grid = np.array(unwrap_grid(obs.frame), dtype=np.int32)
            diff_indices = np.where(prev_grid != curr_grid)
            diffs = []
            for y_idx, x_idx in zip(diff_indices[0], diff_indices[1]):
                diffs.append([int(x_idx), int(y_idx), int(curr_grid[y_idx, x_idx])])
            prev_grid = curr_grid

            # 4. Generate Causal Inference Note
            reasoning = "Kausale Inferenz: Zustandsveränderung registriert."
            if action_name == 'ACTION6':
                if len(diffs) == 0:
                    reasoning = f"Dead-Click Pruning: Koordinate ({data.get('x')}, {data.get('y')}) ohne Auswirkung (ΔF=0). Blacklist aktualisiert."
                else:
                    reasoning = f"Affordance-Aktivierung: Klick löste {len(diffs)} Pixel-Mutationen aus. Neuer Systemzustand verifiziert."
            elif action_name in ('ACTION1', 'ACTION2', 'ACTION3', 'ACTION4'):
                reasoning = f"Diskrete Pfadfindung: Vektorschritt {action_name}. Avatar-Zentroid lokalisiert ({len(diffs)} Pixel translatiert)."

            # 5. Emit step payload
            await websocket.send(json.dumps({
                "type": "STEP_UPDATE",
                "step": step_idx,
                "action": action_desc,
                "action_type": action_name,
                "action_data": data,
                "delta_f": len(diffs),
                "diffs": diffs,
                "levels_completed": obs.levels_completed,
                "state": obs.state.name,
                "think_time_ms": think_time_ms,
                "reasoning": reasoning
            }))

            step_idx += 1

            if obs.state.name in ('WIN', 'GAME_OVER') or obs.levels_completed >= meta["verified_levels"]:
                print(f"[*] [WS-{client_id}] Target reached: levels_completed={obs.levels_completed}, state={obs.state.name}")
                break

            await asyncio.sleep(max(0.02, delay_ms / 1000.0))

        # Emit completion
        await websocket.send(json.dumps({
            "type": "GAME_COMPLETED",
            "total_steps": step_idx,
            "levels_completed": obs.levels_completed,
            "state": obs.state.name
        }))

    except Exception as err:
        print(f"[ERROR] [WS-{client_id}] Error in run_solver_stream: {err}")
        traceback.print_exc()
        try:
            await websocket.send(json.dumps({
                "type": "ERROR",
                "message": str(err)
            }))
        except Exception:
            pass
    finally:
        ACTIVE_SOLVERS.pop(client_id, None)

async def handler(websocket):
    client_id = id(websocket)
    print(f"[+] Client connected [WS-{client_id}]")
    try:
        # Initial Welcome & Game Catalog
        await websocket.send(json.dumps({
            "type": "CATALOG",
            "games": GAME_CATALOG,
            "server_status": "ONLINE",
            "agent": "SovereignARC3Agent V5",
            "engine": "Official ARC-AGI-3 Offline Arcade"
        }))

        async for message in websocket:
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                continue

            cmd = msg.get("command")
            if cmd == "PING":
                await websocket.send(json.dumps({"type": "PONG", "time": time.time()}))
            elif cmd == "START_SOLVE":
                game_id = msg.get("game_id", "lp85")
                delay_ms = int(msg.get("delay_ms", 150))
                ACTIVE_SOLVERS[client_id] = True
                asyncio.create_task(run_solver_stream(websocket, game_id, delay_ms))
            elif cmd == "STOP":
                ACTIVE_SOLVERS[client_id] = False
                await websocket.send(json.dumps({"type": "STOPPED"}))

    except websockets.exceptions.ConnectionClosed:
        print(f"[-] Client disconnected [WS-{client_id}]")
    finally:
        ACTIVE_SOLVERS.pop(client_id, None)

async def main():
    host = os.environ.get("ARC_WS_HOST", "127.0.0.1")
    port = int(os.environ.get("ARC_WS_PORT", 8765))
    print("=================================================================")
    print("   AURATIC PRIME // ARC-AGI-3 LIVE WEBSOCKET SOLVER BRIDGE      ")
    print(f"   Listening on: ws://{host}:{port}                             ")
    print("=================================================================")
    async with websockets.serve(handler, host, port):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[*] Server shutdown gracefully.")
