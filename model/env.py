import math
import random
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tensordict import TensorDict
from tensordict import NonTensorData

from torchrl.envs import EnvBase
from torchrl.data import Composite, Bounded, UnboundedContinuous, NonTensor, Categorical

from scenes.models import *
from scenes.sampler import ScenarioSampler

from websockets.sync.client import connect



class WsSimClient:
    WS_URL = "ws://127.0.0.1:3000"

    def __init__(self, ws_url: Optional[str] = None):
        self.ws_url = ws_url or self.WS_URL
        self.ws = connect(self.ws_url)

    def run_once(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        msg = json.dumps(payload, allow_nan=False)
        
        self.ws.send(msg)
        raw = self.ws.recv()

        out = json.loads(raw)
        if isinstance(out, list):
            out = out[0]

        return out
    

@dataclass
class RewardConfig:
    # Core task signal (dense, always-on)
    w_dist: float = 3.0          # weight for normalized distance penalty (always applied)

    # Terminal success/failure shaping
    success_bonus: float = 10.0  # big enough to dominate distance when solved
    fail_penalty: float = 1.0    # mild constant penalty for failing (prevents "stalling")

    # "Solve faster" shaping (only meaningful on success)
    w_speed: float = 2.0         # bonus in [0, w_speed] based on how early solved

    # Complexity penalties (apply ONLY on success so it doesn't push "stop early")
    w_segments: float = 0.20     # penalty per segment
    w_material: float = 0.05     # penalty per total length unit (meter-ish in your world units)

    # Small per-decision cost (always applied) to avoid infinitely long polylines
    w_design_step: float = 0.01  # penalty per design step / segment decision

    # Safety: final reward clipping (highly recommended for PPO stability)
    reward_clip: float = 20.0


def map_minus1_1_to_0_1(x: float) -> float:
    return 0.5 * (x + 1.0)


def finite(x: float, default: float) -> float:
    return x if math.isfinite(x) else default

def clamp(x: float, low: float, high: float) -> float:
    return low if x < low else high if x > high else x


def clean_segments(segments: List[Tuple[float, float]], min_length: float, max_length: float):
    clean_segments: List[Tuple[float, float]] = []

    for ang, ln in segments:
        ang = finite(float(ang), 0.0)
        ln = finite(float(ln), min_length)
        ln = clamp(ln, min_length, max_length)
        clean_segments.append((ang, ln))

    return clean_segments


def serialize_scene_object(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, WallState):
        return {
            "type": 0, # Wall
            "x": obj.x,
            "y": obj.y,
            "width": obj.width,
            "height": obj.height,
            "rotation": obj.rotation,
        }
    if isinstance(obj, BallState):
        return {
            "type": 1, # Ball
            "x": obj.x,
            "y": obj.y,
            "width": obj.radius,
            "height": obj.radius,
            "rotation": 0.0
        }
    raise TypeError(f"Don't know how to serialize object type: {type(obj)}")

def serialize_tool(
    x: float,
    y: float,
    segments: List[Tuple[float, float]],
) -> Dict[str, Any]:
    segs = []
    for angle_rad, length in segments:
        angle_deg = (angle_rad * 180.0 / math.pi) % 360.0
        segs.append({"angle": float(angle_deg), "length": float(length)})

    return {
        "x": x,
        "y": y,
        "segments": segs,
    }


PHASE_DESIGN = 0
PHASE_PLACEMENT = 1


class ToolDesignPlacementEnv(EnvBase):
    def __init__(
        self,
        specs: List[ScenarioSpec],
        *,
        device: str | torch.device = "cpu",
        max_segments: int = 12,
        min_segment_length: float = 0.05,
        max_segment_length: float = 8.0,
        sampling_mode: str = "random", # "random" or "round_robin"
        reward_cfg: Optional[RewardConfig] = None,
        sim_client: Optional[WsSimClient] = None,
        scene_sampler: Optional[ScenarioSampler] = None
    ):
        super().__init__(device=device)

        if len(specs) == 0:
            raise ValueError("specs must be a non-empty list of ScenarioSpec.")

        self.phase = PHASE_DESIGN

        self.scene_specs = specs
        self.sampling_mode = sampling_mode
        self._rng = random.Random()
        self._spec_index = 0

        self.max_segments = int(max_segments)
        self.min_segment_length = float(min_segment_length)
        self.max_segment_length = float(max_segment_length)
        self.reward_cfg = reward_cfg or RewardConfig()

        self.sim_client = sim_client or WsSimClient()
        self.scene_sampler = scene_sampler or ScenarioSampler()

        # internal episode state
        self._scene: Optional[SampledScene] = None
        self._segments: List[Tuple[float, float]] = []
        self._design_steps = 0

        # --- Specs (keep very simple; dynamic NonTensorData is not tightly spec'd) ---
        self.observation_spec = Composite(
            scene=NonTensor(),
            tool_segments=NonTensor(),
            world=UnboundedContinuous(shape=2, device=self.device, dtype=torch.float32),
            phase=Categorical(n=2, shape=torch.Size((1,)), device=self.device, dtype=torch.int64) # design / place
        )
        self.action_spec = Composite(
            stop=Categorical(n=2, shape=torch.Size((1,)), device=self.device, dtype=torch.int64),
            design=Bounded(low=-1.0, high=1.0, shape=3, device=self.device, dtype=torch.float32),
            place=Bounded(low=-1.0, high=1.0, shape=2, device=self.device, dtype=torch.float32),
        )
        self.reward_spec = UnboundedContinuous(shape=1, device=self.device, dtype=torch.float32)
        self.done_spec = Bounded(low=0, high=1, shape=1, device=self.device, dtype=torch.bool)

    # TorchRL calls these
    def _reset(self, tensordict: Optional[TensorDict] = None, **kwargs) -> TensorDict:
        self.phase = PHASE_DESIGN
        self._scene = self.sample_next_scene()
        self._segments = []
        self._design_steps = 0

        world_w = float(self._scene.world.width)
        world_h = float(self._scene.world.height)

        td = TensorDict(
            {
                "phase": torch.tensor([self.phase], dtype=torch.int64, device=self.device),
                "scene": NonTensorData(self._scene),
                "tool_segments": NonTensorData(list(self._segments)),
                "world": torch.tensor([world_w, world_h], dtype=torch.float32, device=self.device),
                "done": torch.tensor([False], dtype=torch.bool, device=self.device),
                "terminated": torch.tensor([False], dtype=torch.bool, device=self.device),
                "truncated": torch.tensor([False], dtype=torch.bool, device=self.device),
            },
            batch_size=[],
            device=self.device,
        )

        return td

    def _step(self, tensordict: TensorDict) -> TensorDict:
        if self._scene is None:
            raise ValueError("Scene not sampled. Needs env reset")

        action = tensordict.get("action")
        if action is None:
            raise KeyError("Expected 'action' key in tensordict.")

        stop = int(action["stop"].item())
        design = action["design"].detach().cpu().flatten().tolist()
        place = action["place"].detach().cpu().flatten().tolist()

        angle_sin = float(design[0])
        angle_cos = float(design[1])
        length_raw = float(design[2])

        placement_x_raw = float(place[0])
        placement_y_raw = float(place[1])

        world_w = float(self._scene.world.width)
        world_h = float(self._scene.world.height)

        done = False
        reward = 0.0
        info: Dict[str, Any] = {}

        # Phase 0: design segments
        if self.phase == 0:
            # forbid stop on first segment
            if len(self._segments) == 0:
                stop = 0

            # if stop OR reached max segments -> transition to placement phase (not done yet)
            if (stop == 1) or (len(self._segments) >= self.max_segments):
                self.phase = 1
                done = False
                reward = 0.0  # could also add a tiny penalty here if you want
            else:
                angle_rad = math.atan2(angle_sin, angle_cos)

                t = map_minus1_1_to_0_1(length_raw)
                length = self.min_segment_length + t * (self.max_segment_length - self.min_segment_length)
                length = max(self.min_segment_length, min(self.max_segment_length, length)) # clamp to be sure

                self._segments.append((angle_rad, length))
                self._design_steps += 1

                reward = -self.reward_cfg.w_design_step
                done = False

        # Phase 1: placement + simulate (terminal)
        else:
            # placement in world coords
            placement_x = map_minus1_1_to_0_1(placement_x_raw) * world_w
            placement_y = map_minus1_1_to_0_1(placement_y_raw) * world_h

            placement_x = finite(placement_x, 0.5 * world_w)
            placement_y = finite(placement_y, 0.5 * world_h)
            placement_x = clamp(placement_x, 0.0, world_w)
            placement_y = clamp(placement_y, 0.0, world_h)

            self._segments = clean_segments(self._segments, self.min_segment_length, self.max_segment_length)

            payload = {
                "scene": [serialize_scene_object(o) for o in self._scene.scene],
                "tool": serialize_tool(placement_x, placement_y, self._segments),
            }

            try:
                result = self.sim_client.run_once(payload)
            except ValueError as e:
                reward = -float(self.reward_cfg.reward_clip)
                done = True
                info = {
                    "completion": 0,
                    "distance": float("inf"),
                    "sim_steps": 0,
                    "num_segments": len(self._segments),
                    "total_length": sum(l for _, l in self._segments),
                    "design_steps": self._design_steps,
                    "error": str(e),
                }

                return TensorDict(
                    {
                        "phase": torch.tensor([self.phase], dtype=torch.int64, device=self.device),
                        "scene": NonTensorData(self._scene),
                        "tool_segments": NonTensorData(list(self._segments)),
                        "world": torch.tensor([world_w, world_h], dtype=torch.float32, device=self.device),
                        "reward": torch.tensor([reward], dtype=torch.float32, device=self.device),
                        "done": torch.tensor([True], dtype=torch.bool, device=self.device),
                        "terminated": torch.tensor([True], dtype=torch.bool, device=self.device),
                        "truncated": torch.tensor([False], dtype=torch.bool, device=self.device),
                        "info": NonTensorData(info),
                    },
                    batch_size=[],
                    device=self.device,
                )

            completion = int(result.get("completion", 0))
            distance = float(result.get("eucledianDistance", 0.0))
            distance = min(distance, max(world_w, world_h) * 2) # clip falling of the "map" to prevent distance > 15000 problem

            sim_steps = int(result.get("steps", 0))
            total_length = sum(l for _, l in self._segments)
            num_segments = len(self._segments)

            reward = self.compute_reward(result, num_segments, total_length, self._design_steps, self.reward_cfg)

            info = {
                "completion": completion,
                "distance": distance,
                "sim_steps": sim_steps,
                "num_segments": num_segments,
                "total_length": total_length,
                "design_steps": self._design_steps,
            }
            done = True

        td_out = TensorDict(
            {
                "phase": torch.tensor([self.phase], dtype=torch.int64, device=self.device),
                "scene": NonTensorData(self._scene),
                "tool_segments": NonTensorData(list(self._segments)),
                "world": torch.tensor([world_w, world_h], dtype=torch.float32, device=self.device),
                "reward": torch.tensor([reward], dtype=torch.float32, device=self.device),
                "done": torch.tensor([done], dtype=torch.bool, device=self.device),
                "terminated": torch.tensor([done], dtype=torch.bool, device=self.device),
                "truncated": torch.tensor([False], dtype=torch.bool, device=self.device),
                "info": NonTensorData(info),
            },
            batch_size=[],
            device=self.device,
        )
        return td_out
    

    def compute_reward(self, sim_out: dict,
                   num_segments: int,
                   sum_segments_length: float,
                   design_steps: int,
                   cfg: RewardConfig) -> float:

        completion = float(sim_out["completion"])
        dist = float(sim_out["eucledianDistance"])
        init = float(sim_out.get("initialDistance", 1.0))
        steps = float(sim_out.get("steps", sim_out.get("maxSteps", 1.0)))
        max_steps = float(sim_out.get("maxSteps", 1.0))

        # --- Dense distance shaping (ALWAYS on) ---
        # normalized distance in [0, ...], usually around 1 when no progress
        init = max(init, 1e-6)
        norm_dist = dist / init
        reward = -cfg.w_dist * norm_dist

        # --- Mild terminal shaping ---
        if completion > 0.5:
            reward += cfg.success_bonus

            # speed bonus in [0, w_speed]
            # if solved instantly -> ~w_speed ; if solved at timeout -> ~0
            frac = 1.0 - min(max(steps / max(max_steps, 1e-6), 0.0), 1.0)
            reward += cfg.w_speed * frac

            # complexity penalties ONLY on success
            reward -= cfg.w_segments * float(num_segments)
            reward -= cfg.w_material * float(sum_segments_length)

        else:
            reward -= cfg.fail_penalty

        # reward -= cfg.w_design_step * float(design_steps)

        # --- PPO stability: clip final reward ---
        if cfg.reward_clip is not None:
            c = float(cfg.reward_clip)
            if reward > c:
                reward = c
            elif reward < -c:
                reward = -c

        return reward
    

    def _set_seed(self, seed: int | None):
        """
        Required by EnvBase (abstract). TorchRL calls this via env.set_seed(...).
        Return the seed actually used.
        """
        if seed is None:
            seed = int(torch.seed() % (2**31 - 1))

        self._seed = int(seed)

        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)

        self._rng = random.Random(self._seed)   

    def sample_next_scene(self) -> SampledScene:
        if self.sampling_mode == "round_robin":
            spec = self.scene_specs[self._spec_index % len(self.scene_specs)]
            self._spec_index += 1
        elif self.sampling_mode == "random":
            spec = self._rng.choice(self.scene_specs)
        else:
            raise ValueError(f"Unknown sampling_mode '{self.sampling_mode}' (only 'random' or 'round_robin').")

        scene = self.scene_sampler.sample(spec)
        return scene