import inspect
import itertools
from typing import Any, Dict, Optional, List, Tuple, Union, cast
from dataclasses import dataclass

import numpy as np
from scenes.models import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import EdgeType
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, ProbabilisticTensorDictModule, ProbabilisticTensorDictSequential
from tensordict.nn.distributions import CompositeDistribution
from torchrl.modules import ValueOperator
from torchrl.modules.distributions import TanhNormal


NODE_TYPE_MAP: dict[type, str] = {
    WallState: "wall",
    BallState: "ball"
}

NODE_FEATURE_FNS = {
    BallState: lambda b: [b.x, b.y, b.radius],
    WallState: lambda w: [w.x, w.y, w.width, w.height, w.rotation]
}

NODE_FALLBACK_DIMS: dict[type, int] = {
    BallState: 3,
    WallState: 5
}


def fully_connected_edge_index(n_src: int, n_dst: int, allow_same_index_edges: bool = False):
    src = torch.arange(n_src)
    dst = torch.arange(n_dst)
    grid = torch.cartesian_prod(src, dst)
    edge_index = grid.t().contiguous()

    if (n_src == n_dst) and (not allow_same_index_edges):
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        
    return edge_index


def build_scene_heterodata(scene: SampledScene) -> HeteroData:
    data = HeteroData()

    objects_by_type: dict[type, list] = {t: [] for t in NODE_TYPE_MAP}
    global_to_local: dict[int, tuple[str, int]] = {}
    for global_idx, obj in enumerate(scene.scene):
        for t in NODE_TYPE_MAP:
            if isinstance(obj, t):
                objects_by_type[t].append(obj)
                node_type = NODE_TYPE_MAP[t]
                local_idx = len(objects_by_type[t]) - 1
                global_to_local[global_idx] = (node_type, local_idx)
                break

    for t, objs in objects_by_type.items():
        if len(objs) > 0:
            data[NODE_TYPE_MAP[t]].x = torch.tensor([NODE_FEATURE_FNS[t](o) for o in objs], dtype=torch.float32)
        else:
            data[NODE_TYPE_MAP[t]].x = torch.zeros((0, NODE_FALLBACK_DIMS[t]), dtype=torch.float32)

    for src_t, dst_t in itertools.product(NODE_TYPE_MAP, NODE_TYPE_MAP):
        src_node_type = NODE_TYPE_MAP[src_t]
        dst_node_type = NODE_TYPE_MAP[dst_t]

        n_src = data[src_node_type].num_nodes
        n_dst = data[dst_node_type].num_nodes

        data[(src_node_type, "in_scene", dst_node_type)].edge_index = fully_connected_edge_index(n_src, n_dst, src_node_type != dst_node_type)


    goal_edges: dict[tuple[str, str], list[list[int]]] = {}
    for a_global_idx, b_global_idx in scene.goal_pairs:
        if (a_global_idx not in global_to_local) or (b_global_idx not in global_to_local):
            continue

        a_type, a_local_idx = global_to_local[a_global_idx]
        b_type, b_local_idx = global_to_local[b_global_idx]

        key = (a_type, b_type)
        if key not in goal_edges:
            goal_edges[key] = [[], []]

        goal_edges[key][0].append(a_local_idx)
        goal_edges[key][1].append(b_local_idx)

    for (src_type, dst_type), (src_idx, dst_idx) in goal_edges.items():
        data[(src_type, "goal", dst_type)].edge_index = torch.tensor(
            [src_idx, dst_idx], dtype=torch.long
        )

    return data


class SceneEncoder(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, num_layers: int, aggr: str = "sum", dropout: float = 0.0):
        super().__init__()

        self.hidden_dim = hidden_dim

        # project feature set for each node type to same size vector
        self.in_proj = nn.ModuleDict({
            nt: nn.Linear(NODE_FALLBACK_DIMS[t], hidden_dim)
            for t, nt in NODE_TYPE_MAP.items()
        })

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict: Dict[EdgeType, MessagePassing] = {
                edge_type: SAGEConv((hidden_dim, hidden_dim), hidden_dim)
                for edge_type in itertools.product(NODE_TYPE_MAP.values(), ["in_scene", "goal"], NODE_TYPE_MAP.values())
            }
            self.convs.append(HeteroConv(conv_dict, aggr=aggr))

        # normalization of aggreagtion, then MLP for NON-LINEARITY + DROPOUT
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data):
        x_dict = {
            nt: self.in_proj[nt](data[nt].x)
            for nt in data.node_types
        }

        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            
            # nonlinearity after each conv
            x_dict = {nt: F.relu(x) for nt, x in x_dict.items()}

        # mean-pool per type then mean across types
        # should work but maybe dense layer like with transformer is better?
        typewise_mean = []
        for x in x_dict.values():
            if x.size(0) > 0:
                typewise_mean.append(x.mean(dim=0))

        # fallback all zeros if scene is empty for whatever reason
        if len(typewise_mean) == 0:
            scene_h = torch.zeros(self.hidden_dim, device=next(self.parameters()).device)
        else:
            scene_h = torch.stack(typewise_mean, dim=0).mean(dim=0)

        scene_embedding = self.out(scene_h)
        return scene_embedding
    

@dataclass
class ToolState:
    segments: List[Tuple[float, float]]


class ToolEncoder(nn.Module):
    def __init__(self, hidden_size: int = 8, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        
        self.lstm = nn.LSTM(
            input_size=3, # input segment features [sin(angle), cos(angle), length]
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(
        self,
        segments: List[Tuple[float, float]],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device

        T = len(segments)
        if T == 0:
            return torch.zeros((1, self.hidden_size), dtype=torch.float32, device=device)

        angles = torch.tensor([a for a, _ in segments], dtype=torch.float32, device=device)
        lengths = torch.tensor([l for _, l in segments], dtype=torch.float32, device=device)

        # use sin/cos for continuity, 1° and 359° are very near directions but big difference in the number
        # sin(1°) and sin(359°) are close to each other like real direction
        x = torch.stack([torch.sin(angles), torch.cos(angles), lengths], dim=-1) # [n_seg, 3]
        x = x.unsqueeze(0) # [1, n_seg, 3]

        _, (h, _) = self.lstm(x) # hidden states - h: [num_layers, 1, hidden]
        return h[-1] # last hidden state - [1, hidden]
    

@dataclass
class DesignerOutput:
    # STOP Bernoulli logits
    stop_logit: torch.Tensor # [B, 1]

    # Angle as direction vector
    angle_vec: torch.Tensor # [B, 2] (unnormalized)
    angle_unit: torch.Tensor # [B, 2] normalized
    angle_radians: torch.Tensor # [B, 1]

    # Length [min_len, max_len]
    length: torch.Tensor # [B, 1]
    length_norm: torch.Tensor # [B, 1] in [0, 1]


class DesignerHead(nn.Module):
    def __init__(
        self,
        scene_dim: int,
        tool_dim: int,
        hidden_dim: int = 128,
        min_segment_length: float = 0.05,
        max_segment_length: float = 8.0,
    ):
        super().__init__()

        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length

        in_dim = scene_dim + tool_dim

        # encode/embed current scene + current tool
        # shared base for all heads
        # dense layer to shared hidden size, then another non-linearity (probably not necessary)
        self.base = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # get STOP (Bernoulli logit)
        self.stop_head = nn.Linear(hidden_dim, 1)

        # get angle vector -> normalized -> then atan2 for angle
        self.angle_head = nn.Linear(hidden_dim, 2)

        # Length raw -> sigmoid -> [min,max]
        # this sigmoid part was proposed by ChatGPT
        self.length_head = nn.Linear(hidden_dim, 1)

    def forward(self, scene_embedding: torch.Tensor, tool_embedding: torch.Tensor) -> DesignerOutput:
        # Make both batched [B, D] if not already
        if scene_embedding.dim() == 1:
            scene_embedding = scene_embedding.unsqueeze(0)
        if tool_embedding.dim() == 1:
            tool_embedding = tool_embedding.unsqueeze(0)

        # If one is [1, D] and the other is [B, D] it will not work, so force same batch size
        if scene_embedding.shape[0] != tool_embedding.shape[0]:
            raise ValueError(
                f"Batch mismatch: scene_embedding batch={scene_embedding.shape[0]} tool_embedding batch={tool_embedding.shape[0]}."
            )

        h = self.base(torch.cat([scene_embedding, tool_embedding], dim=-1))

        stop_logit = self.stop_head(h) # [B,1]
        angle_vec = self.angle_head(h) # [B,2]

        # Normalize to unit direction
        angle_unit = F.normalize(angle_vec, dim=-1, eps=1e-8) # [B,2]

        # angle_unit = [sin, cos] -> angle = atan2(sin, cos)
        angle_radians = torch.atan2(angle_unit[:, 0], angle_unit[:, 1]).unsqueeze(-1) # [B,1]

        # Length bounded to [min,max]
        length_raw = self.length_head(h) # [B,1]
        length_norm = torch.sigmoid(length_raw) # [0,1]
        length = self.min_segment_length + length_norm * (self.max_segment_length - self.min_segment_length)

        return DesignerOutput(
            stop_logit=stop_logit,
            angle_vec=angle_vec,
            angle_unit=angle_unit,
            angle_radians=angle_radians,
            length=length,
            length_norm=length_norm
        )
    

@dataclass
class PlacementOutput:
    x: torch.Tensor # [B, 1]
    y: torch.Tensor # [B, 1]


class PlacementHead(nn.Module):
    def __init__(self, scene_dim: int, tool_dim: int, hidden_dim: int = 128):
        super().__init__()

        in_dim = scene_dim + tool_dim

        # encode/embed current scene + current tool
        # shared base for all heads
        # dense layer to shared hidden size, then another non-linearity (probably not necessary)
        self.base = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # predict normalized coords with sigmoid then scale by world size
        self.position_head = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        scene_embedding: torch.Tensor,
        tool_embedding: torch.Tensor,
        world_width: torch.Tensor | float,
        world_height: torch.Tensor | float,
    ) -> PlacementOutput:
        # Make both batched [B, D]
        if scene_embedding.dim() == 1:
            scene_embedding = scene_embedding.unsqueeze(0)
        if tool_embedding.dim() == 1:
            tool_embedding = tool_embedding.unsqueeze(0)

        # If one is [1, D] and the other is [B, D] it will not work, so force same batch size
        if scene_embedding.shape[0] != tool_embedding.shape[0]:
            raise ValueError(
                f"Batch mismatch: scene_embedding batch={scene_embedding.shape[0]} tool_embedding batch={tool_embedding.shape[0]}."
            )

        h = self.base(torch.cat([scene_embedding, tool_embedding], dim=-1))
        position_raw = self.position_head(h) # [B,2]
        position_norm = torch.sigmoid(position_raw) # [B,2] in [0,1]

        device = position_norm.device
        dtype = position_norm.dtype

        width = world_width if torch.is_tensor(world_width) else torch.tensor(world_width, device=device, dtype=dtype)
        height = world_height if torch.is_tensor(world_height) else torch.tensor(world_height, device=device, dtype=dtype)

        x = (position_norm[:, 0:1] * width) # [B,1]
        y = (position_norm[:, 1:1+1] * height) # [B,1]

        return PlacementOutput(x=x, y=y)
    

@dataclass
class ValueOutput:
    value: torch.Tensor  # [B, 1]


class ValueHead(nn.Module):
    def __init__(self, scene_dim: int, tool_dim: int, hidden_dim: int = 128):
        super().__init__()
        in_dim = scene_dim + tool_dim
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, scene_embedding: torch.Tensor, tool_embedding: torch.Tensor) -> ValueOutput:
        if scene_embedding.dim() == 1:
            scene_embedding = scene_embedding.unsqueeze(0)
        if tool_embedding.dim() == 1:
            tool_embedding = tool_embedding.unsqueeze(0)

        if scene_embedding.shape[0] != tool_embedding.shape[0]:
            raise ValueError(
                f"Batch mismatch: scene_embedding batch={scene_embedding.shape[0]} tool_embedding batch={tool_embedding.shape[0]}."
            )

        x = torch.cat([scene_embedding, tool_embedding], dim=-1)
        v = self.network(x) # [B,1]
        return ValueOutput(value=v)
    

class Critic(nn.Module):
    def __init__(
        self,
        scene_encoder,
        tool_encoder,
        value_head
    ):
        super().__init__()
        self.scene_encoder = scene_encoder
        self.tool_encoder = tool_encoder
        self.value_head = value_head

    def forward(
            self,
            scene: SampledScene | List[SampledScene] | Tuple[SampledScene],
            tool_segments # List[Tuple[float, float]] | List[List[Tuple[float, float]]] - no fixed typed here to prevent big checks just trust in pipeline to give correct data
        ):
        device = next(self.parameters()).device

        # if more than one scene like batches
        # I dont know if batches will work with only one blocking ws-env
        if isinstance(scene, (list, tuple)):
            values = []
            for s, segs in zip(scene, tool_segments):
                values.append(self._forward_one(s, segs, device))

            return torch.stack(values, dim=0) # [B,1]

        return self._forward_one(scene, tool_segments, device).unsqueeze(0) # [1,1]

    def _forward_one(self, scene: SampledScene, tool_segments: List[Tuple[float, float]], device: torch.device) -> torch.Tensor:
        data = build_scene_heterodata(scene).to(str(device))
        scene_emb = self.scene_encoder(data) # [scene_dim]

        tool_emb = self.tool_encoder(tool_segments, device=device) # [1, tool_dim]

        v = self.value_head(scene_emb, tool_emb).value # [1,1]
        return v.squeeze(0)
    

def make_torchrl_critic(
    scene_encoder,
    tool_encoder,
    value_head
):
    critic = Critic(
        scene_encoder=scene_encoder,
        tool_encoder=tool_encoder,
        value_head=value_head
    )

    td_critic = TensorDictModule(
        module=critic,
        in_keys=["scene", "tool_segments"],
        out_keys=["state_value"],
    )

    critic_value_op = ValueOperator(
        module=td_critic,
        in_keys=["scene", "tool_segments"],
    )

    return td_critic, critic_value_op


class Actor(nn.Module):
    def __init__(
        self,
        scene_encoder: SceneEncoder,
        tool_encoder: ToolEncoder,
        designer_head: DesignerHead,
        placement_head: PlacementHead,
        init_log_std: float = 0.5,
    ):
        super().__init__()
        self.scene_encoder = scene_encoder
        self.tool_encoder = tool_encoder
        self.designer_head = designer_head
        self.placement_head = placement_head

        self.log_std = nn.Parameter(torch.full((4,), float(init_log_std)))

    def forward(
        self,
        scene: SampledScene | List[SampledScene] | Tuple[SampledScene],
        tool_segments,
        world: torch.Tensor,
    ):
        device = next(self.parameters()).device

        if isinstance(scene, (list, tuple)):
            locs = []
            for idx, (s, segs) in enumerate(zip(scene, tool_segments)):
                wh = world[idx] if world.dim() == 2 else world
                locs.append(self._forward_one(s, segs, wh, device))
            loc = torch.stack(locs, dim=0) # [B,6]
        else:
            loc = self._forward_one(scene, tool_segments, world, device).unsqueeze(0) # [1,6]

        LOG_STD_MIN, LOG_STD_MAX = -3.0, 1.0  # tighter = more stable early training
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

        scale = log_std.exp().unsqueeze(0).expand_as(loc)

        # hard safety: never allow non-finite / degenerate scales
        scale = torch.nan_to_num(scale, nan=0.1, posinf=1.0, neginf=0.1)
        scale = torch.clamp(scale, 1e-3, 2.0)

        # also ensure loc is finite (in case upstream produced NaN after stacking)
        loc = torch.nan_to_num(loc, nan=0.0, posinf=0.0, neginf=0.0)

        return loc, scale

    def _forward_one(
        self,
        scene: SampledScene,
        tool_segments: List[Tuple[float, float]],
        world: torch.Tensor, # (width,height)
        device: torch.device,
    ) -> torch.Tensor:
        data = build_scene_heterodata(scene).to(str(device))
        scene_emb = self.scene_encoder(data) # [scene_dim]

        tool_emb = self.tool_encoder(tool_segments, device=device) # [1, tool_dim]

        width = float(world[0].item())
        height = float(world[1].item())

        designer_output = self.designer_head(scene_emb, tool_emb)
        placement_output = self.placement_head(scene_emb, tool_emb, width, height)

        # Map head outputs to normalized action space [-1,1]
        stop_prob = torch.sigmoid(designer_output.stop_logit).squeeze(0).squeeze(-1)  # [0,1]
        stop_signal = stop_prob * 2.0 - 1.0  # [-1,1]

        angle_sin = designer_output.angle_unit[:, 0].squeeze(0)
        angle_cos = designer_output.angle_unit[:, 1].squeeze(0)

        seg_length_raw = (designer_output.length_norm.squeeze(0).squeeze(-1) * 2.0) - 1.0 # [0,1] -> [-1,1]

        place_x_raw = (placement_output.x.squeeze(0).squeeze(-1) / max(width, 1e-8)) * 2.0 - 1.0 # [0, width] -> [-1,1]
        place_y_raw = (placement_output.y.squeeze(0).squeeze(-1) / max(height, 1e-8)) * 2.0 - 1.0 # [0, height] -> [-1,1]

        loc = torch.stack([stop_signal, angle_sin, angle_cos, seg_length_raw], dim=0)
        loc = torch.nan_to_num(loc, nan=0.0, posinf=0.999, neginf=-0.999)
        loc = torch.clamp(loc, -0.999, 0.999)

        # safe atanh
        return torch.atanh(loc)
    

def _atanh_safe(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # clamp to avoid inf at atanh(±1)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
    return torch.atanh(x)


class PolicyParamNet(nn.Module):
    def __init__(
        self,
        scene_encoder: SceneEncoder,
        tool_encoder: ToolEncoder,
        designer_head: DesignerHead,
        placement_head: PlacementHead,
        log_std_design_init: float = -1.0,
        log_std_place_init: float = -1.0,
    ):
        super().__init__()
        self.scene_encoder = scene_encoder
        self.tool_encoder = tool_encoder
        self.designer_head = designer_head
        self.placement_head = placement_head

        # learnable log stds (separate!)
        self.log_std_design = nn.Parameter(torch.full((3,), float(log_std_design_init)))
        self.log_std_place = nn.Parameter(torch.full((2,), float(log_std_place_init)))

    def forward(self, scene, tool_segments, world, phase):
        device = world.device

        # TorchRL may batch NonTensorData into Python lists/tuples.
        if isinstance(scene, (list, tuple)):
            params_list = []
            B = len(scene)
            for i in range(B):
                s_i = scene[i]
                segs_i = tool_segments[i]

                # world may be [B,2] or [2]
                w_i = world[i] if (torch.is_tensor(world) and world.dim() == 2) else world
                p_i = phase[i] if (torch.is_tensor(phase) and phase.numel() > 1) else phase

                params_list.append(self._forward_one(s_i, segs_i, w_i, p_i, device))

            # stack into a batch TensorDict
            return torch.stack(params_list, dim=0)

        return self._forward_one(scene, tool_segments, world, phase, device)

    def _forward_one(self, scene, tool_segments, world, phase, device):
        # encode
        data = build_scene_heterodata(scene).to(str(device))
        scene_emb = self.scene_encoder(data)
        tool_emb = self.tool_encoder(tool_segments, device=device)

        width = float(world[..., 0].item())
        height = float(world[..., 1].item())

        d_out = self.designer_head(scene_emb, tool_emb)
        p_out = self.placement_head(scene_emb, tool_emb, width, height)

        # stop logits: shape [2]
        stop_logit = d_out.stop_logit.squeeze()
        stop_logits = torch.stack([torch.zeros_like(stop_logit), stop_logit], dim=-1)  # [2]

        # design loc/scale (pre-tanh loc, scale positive)
        eps = 1e-6
        angle_sin = d_out.angle_unit[:, 0].squeeze(0)
        angle_cos = d_out.angle_unit[:, 1].squeeze(0)
        length_raw = d_out.length_norm.squeeze(0).squeeze(-1) * 2.0 - 1.0
        design_tanh = torch.stack([angle_sin, angle_cos, length_raw], dim=-1)  # [3]
        design_tanh = torch.nan_to_num(design_tanh, nan=0.0, posinf=0.0, neginf=0.0)
        design_tanh = torch.clamp(design_tanh, -1 + eps, 1 - eps)
        design_loc = torch.atanh(design_tanh)  # [3]

        ds = self.log_std_design.clamp(-3.0, 1.0).exp()
        ds = torch.nan_to_num(ds, nan=0.1, posinf=1.0, neginf=0.1).clamp(1e-3, 2.0)  # [3]

        # place loc/scale
        x_raw = (p_out.x.squeeze(0).squeeze(-1) / max(width, 1e-8)) * 2.0 - 1.0
        y_raw = (p_out.y.squeeze(0).squeeze(-1) / max(height, 1e-8)) * 2.0 - 1.0
        place_tanh = torch.stack([x_raw, y_raw], dim=-1)  # [2]
        place_tanh = torch.nan_to_num(place_tanh, nan=0.0, posinf=0.0, neginf=0.0)
        place_tanh = torch.clamp(place_tanh, -1 + eps, 1 - eps)
        place_loc = torch.atanh(place_tanh)  # [2]

        ps = self.log_std_place.clamp(-3.0, 1.0).exp()
        ps = torch.nan_to_num(ps, nan=0.1, posinf=1.0, neginf=0.1).clamp(1e-3, 2.0)  # [2]

        # -------- phase gating (library-safe, no manual logprobs) ----------
        ph = int(phase.item())
        if ph == 0:
            # design active; placement deterministic ~0
            place_loc = torch.zeros_like(place_loc)
            ps = torch.full_like(ps, 1e-3)
        else:
            # placement active; stop forced to 1; design deterministic ~0
            stop_logits = torch.tensor([-20.0, 20.0], device=device)
            design_loc = torch.zeros_like(design_loc)
            ds = torch.full_like(ds, 1e-3)

        # Build params tensordict with nested keys matching distribution_map keys
        params = TensorDict(
            {
                ("stop", "logits"): stop_logits,
                ("design", "loc"): design_loc,
                ("design", "scale"): ds,
                ("place", "loc"): place_loc,
                ("place", "scale"): ps,
            },
            batch_size=[],
            device=device,
        )
        return params

    

def make_torchrl_actor(
    scene_encoder: SceneEncoder,
    tool_encoder: ToolEncoder,
    designer_head: DesignerHead,
    placement_head: PlacementHead,
):
    param_net = PolicyParamNet(scene_encoder, tool_encoder, designer_head, placement_head)

    param_module = TensorDictModule(
        module=param_net,
        in_keys=["scene", "tool_segments", "world", "phase"],
        out_keys=["params"],
    )

    # probabilistic part: read dist params and sample action
    prob_module = ProbabilisticTensorDictModule(
        in_keys=["params"],
        out_keys=[("action", "stop"), ("action", "design"), ("action", "place")],
        distribution_class=CompositeDistribution,
        distribution_kwargs={
            "distribution_map": {
                "stop": Categorical,
                "design": TanhNormal,
                "place": TanhNormal,
            },
            "name_map": {
                "stop": ("action", "stop"),
                "design": ("action", "design"),
                "place": ("action", "place"),
            },
            "extra_kwargs": {
                "design": {"low": -1.0, "high": 1.0},
                "place": {"low": -1.0, "high": 1.0},
            },
        },
        return_log_prob=True,
        # IMPORTANT: composite_lp_aggregate(False) => MUST use log_prob_keys (one per out_key)
        log_prob_keys=[
            ("action", "stop_log_prob"),
            ("action", "design_log_prob"),
            ("action", "place_log_prob"),
        ],

        cache_dist=True,
    )


    # IMPORTANT: use ProbabilisticTensorDictSequential (not TensorDictSequential)
    actor = ProbabilisticTensorDictSequential([param_module, prob_module])
    return actor