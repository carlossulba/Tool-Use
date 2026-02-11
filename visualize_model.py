import torch
import argparse
import time
import random
import numpy as np

from tensordict.nn import set_composite_lp_aggregate
set_composite_lp_aggregate(False).set()

from model.env import ToolDesignPlacementEnv
from model.model_impl import (
    DesignerHead, PlacementHead, SceneEncoder, ToolEncoder, make_torchrl_actor
)

import scenes.levels as levels

LEVEL_MAP = {
    "floor": levels.TWO_BALLS_ON_FLOOR,
    "bump": levels.TWO_BALLS_ON_FLOOR_WITH_TINY_BUMP,
    "wall": levels.TWO_BALLS_ON_FLOOR_WITH_CENTER_WALL,
    "gap": levels.TWO_BALLS_GAP,
    "cliff": levels.TWO_BALLS_CLIFF,
    "valley": levels.TWO_BALLS_VALLEY
}

def run_multi_visualization(checkpoint_path, level_key, count=5, delay=12, device_str="cpu"):
    device = torch.device(device_str)
    selected_level = LEVEL_MAP.get(level_key)
    
    # 1. Setup Environment
    env = ToolDesignPlacementEnv(
        specs=[selected_level],
        device=device,
        max_segments=6,
    )

    # 2. Reconstruct Model Architecture
    scene_encoder = SceneEncoder(hidden_dim=128, out_dim=128, num_layers=3).to(device)
    tool_encoder = ToolEncoder(hidden_size=32).to(device)
    designer_head = DesignerHead(scene_dim=128, tool_dim=32, hidden_dim=256).to(device)
    placement_head = PlacementHead(scene_dim=128, tool_dim=32, hidden_dim=256).to(device)
    actor = make_torchrl_actor(scene_encoder, tool_encoder, designer_head, placement_head).to(device)

    # 3. Load Weights
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    print(f"Starting Video Sequence: {count} episodes on level '{level_key}'")
    
    for i in range(count):
        print(f"\n--- Episode {i+1}/{count} ---")
        
        # --- DEFINITIVE RANDOMIZATION FIX ---
        # seed = int(time.time() * 1000) % (2**31 - 1)
        # env.set_seed(seed)
        # random.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # -------------------------------------

        td = env.reset()
        
        from tensordict.nn import InteractionType, set_interaction_type
        with torch.no_grad(), set_interaction_type(InteractionType.DETERMINISTIC):
            done = False
            while not done:
                td = actor(td)
                td = env.step(td)
                done = td["next"]["done"].item()
                
                from torchrl.envs.utils import step_mdp
                td = step_mdp(td)

        print(f"Design sent to UI. Watching simulation for {delay} seconds...")
        # We wait here so the UI has time to finish the "Automatic" simulation
        # before we overwrite the file with a new scene.
        time.sleep(delay)

    print("\nVideo sequence finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--level", type=str, default="floor", choices=LEVEL_MAP.keys())
    parser.add_argument("--count", type=int, default=5, help="Number of scenes to generate")
    parser.add_argument("--delay", type=int, default=12, help="Seconds to wait between scenes")
    args = parser.parse_args()
    
    run_multi_visualization(args.ckpt, args.level, args.count, args.delay)