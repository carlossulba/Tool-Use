# train_ppo.py
import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from torchrl.collectors import SyncDataCollector
from torchrl.envs.utils import step_mdp
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from model.env import ToolDesignPlacementEnv
from model.model_impl import (
    DesignerHead,
    PlacementHead,
    SceneEncoder,
    ToolEncoder,
    ValueHead,
    make_torchrl_actor,
    make_torchrl_critic,
)

from scenes.levels import TWO_BALLS_ON_FLOOR, TWO_BALLS_ON_FLOOR_WITH_TINY_BUMP, TWO_BALLS_ON_FLOOR_WITH_CENTER_WALL

from tqdm.auto import tqdm

from contextlib import nullcontext
from tensordict.nn import InteractionType, set_interaction_type
from tensordict.nn import set_composite_lp_aggregate

set_composite_lp_aggregate(False).set()


def unique_params(*modules):
    seen = set()
    out = []
    for m in modules:
        for p in m.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
    return out


def _run_name_now() -> str:
    # Example: RUN_20260206_153012
    return "RUN_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def _make_unique_dir(path: Path) -> Path:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=False)
        return path
    # If it already exists, append _1, _2, ...
    for k in range(1, 10_000):
        cand = Path(f"{str(path)}_{k}")
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=False)
            return cand
    raise RuntimeError(f"Could not create a unique run directory under: {path}")


def _atomic_torch_save(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _save_snapshot(
    run_dir: Path,
    run_batch: int,
    global_batch: int,
    actor: torch.nn.Module,
    critic_value_op: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    cfg: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    ckpt: Dict[str, Any] = {
        "schema_version": 1,
        "time_saved": datetime.now().isoformat(timespec="seconds"),
        "run_batch": int(run_batch),
        "global_batch": int(global_batch),
        # Save the FULL actor/critic to include ALL params (e.g., PolicyParamNet log_stds).
        "actor": actor.state_dict(),
        "critic_value_op": critic_value_op.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_rng_state": random.getstate(),
        "cfg": cfg,
        "extra": extra or {},
    }

    out_path = run_dir / f"model_{run_batch}.pt"
    _atomic_torch_save(ckpt, out_path)

    # Convenience pointer to the newest checkpoint
    latest_path = run_dir / "latest.pt"
    try:
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        # Symlink when possible (nice for large checkpoints)
        latest_path.symlink_to(out_path.name)
    except Exception:
        # Fallback: copy the checkpoint contents
        _atomic_torch_save(ckpt, latest_path)

    return out_path


def _load_snapshot(
    ckpt_path: Path,
    device: torch.device,
    actor: torch.nn.Module,
    critic_value_op: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    *,
    load_optimizer: bool,
    strict: bool,
    restore_rng: bool,
) -> Dict[str, Any]:
    ckpt: Dict[str, Any] = torch.load(ckpt_path, map_location="cpu")

    actor.load_state_dict(ckpt["actor"], strict=strict)
    critic_value_op.load_state_dict(ckpt["critic_value_op"], strict=strict)

    if load_optimizer and optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    if restore_rng:
        try:
            torch.set_rng_state(ckpt["torch_rng_state"])
        except Exception:
            pass
        if torch.cuda.is_available() and ckpt.get("cuda_rng_state_all") is not None:
            try:
                torch.cuda.set_rng_state_all(ckpt["cuda_rng_state_all"])
            except Exception:
                pass
        try:
            random.setstate(ckpt["python_rng_state"])
        except Exception:
            pass

    # Ensure modules are on the requested device
    actor.to(device)
    critic_value_op.to(device)

    return ckpt


def evaluate_actor(env, actor, *, episodes: int = 1000, deterministic: bool = True) -> Dict[str, Any]:
    actor.eval()

    successes = 0
    ep_returns: list[float] = []
    segs: list[int] = []
    lengths: list[float] = []

    # "Deterministic" interaction for TorchRL distributions

    ctx = set_interaction_type(InteractionType.DETERMINISTIC) if deterministic else nullcontext()

    # Hard safety cap: your env should finish in <= max_segments + 2 steps
    max_steps = int(getattr(env, "max_segments", 12)) + 5

    with torch.no_grad(), ctx:
        for e in range(episodes):
            print(e)

            td = env.reset()
            ep_ret = 0.0

            for _t in range(max_steps):
                td = actor(td)
                td = env.step(td)

                # IMPORTANT: move ("next", ...) -> root, otherwise you keep acting on the same obs forever
                if "next" in td.keys():
                    td = step_mdp(td, keep_other=False)

                if "reward" in td.keys():
                    ep_ret += float(td["reward"].item())

                if "done" in td.keys() and bool(td["done"].item()):
                    info = td.get("info", None)
                    if info is not None:
                        info_dict = info.data if hasattr(info, "data") else info
                        completion = int(info_dict.get("completion", 0))
                        successes += int(completion == 1)
                        segs.append(int(info_dict.get("num_segments", 0)))
                        lengths.append(float(info_dict.get("total_length", 0.0)))

                    ep_returns.append(ep_ret)
                    break
            else:
                # If we ever hit this, something is wrong — don't infinite loop.
                print(f"[EVAL] WARNING: episode {e} did not terminate after {max_steps} steps")
                ep_returns.append(ep_ret)

    n = float(max(1, episodes))
    return {
        "episodes": episodes,
        "deterministic": deterministic,
        "successes": int(successes),
        "success_rate": float(successes / n),
        "avg_return": float(sum(ep_returns) / max(1, len(ep_returns))),
        "avg_num_segments": float(sum(segs) / max(1, len(segs))),
        "avg_total_length": float(sum(lengths) / max(1, len(lengths))),
    }




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot_root", type=str, default="model_snapshots")
    parser.add_argument("--run_name", type=str, default=None)

    # Start from weights only (fresh optimizer), saves into a NEW run folder
    parser.add_argument("--init_from", type=str, default=None)

    # Resume weights + optimizer + counters, saves into SAME run folder as checkpoint
    parser.add_argument("--resume_from", type=str, default=None)

    parser.add_argument("--no_strict_load", action="store_true", help="Disable strict state_dict loading")

    parser.add_argument(
    "--evaluate",
        action="store_true",
        help="Run deterministic evaluation for 1000 episodes and exit",
    )

    args = parser.parse_args()

    strict_load = not args.no_strict_load

    # ---------- SNAPSHOT DIR ----------
    snapshot_root = Path(args.snapshot_root).expanduser().resolve()

    if args.resume_from:
        ckpt_path = Path(args.resume_from).expanduser().resolve()
        run_dir = ckpt_path.parent
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir_base = snapshot_root / (args.run_name or _run_name_now())
        run_dir = _make_unique_dir(run_dir_base)

    # Save minimal run metadata
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        meta = {
            "run_dir": str(run_dir),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "init_from": args.init_from,
            "resume_from": args.resume_from,
        }
        meta_path.write_text(json.dumps(meta, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- ENV ----------
    env = ToolDesignPlacementEnv(
        specs=[TWO_BALLS_ON_FLOOR_WITH_CENTER_WALL],
        device=device,
        sampling_mode="random",
        max_segments=6,
        min_segment_length=0.05,
        max_segment_length=8.0,
    )

    # ---------- MODEL ----------
    scene_hidden_dim = 128
    scene_out_dim = 128
    tool_hidden_dim = 32

    scene_encoder = SceneEncoder(hidden_dim=scene_hidden_dim, out_dim=scene_out_dim, num_layers=3, dropout=0.10).to(device)
    tool_encoder = ToolEncoder(hidden_size=tool_hidden_dim).to(device)

    designer_head = DesignerHead(scene_dim=scene_out_dim, tool_dim=tool_hidden_dim, hidden_dim=256).to(device)
    placement_head = PlacementHead(scene_dim=scene_out_dim, tool_dim=tool_hidden_dim, hidden_dim=256).to(device)
    value_head = ValueHead(scene_dim=scene_out_dim, tool_dim=tool_hidden_dim, hidden_dim=256).to(device)

    actor = make_torchrl_actor(scene_encoder, tool_encoder, designer_head, placement_head).to(device)
    _, critic_value_op = make_torchrl_critic(scene_encoder, tool_encoder, value_head)
    critic_value_op = critic_value_op.to(device)

    advantage = GAE(
        gamma=0.99,
        lmbda=0.95,
        value_network=critic_value_op,
        deactivate_vmap=True,
    )

    loss_module = ClipPPOLoss(
        actor=actor,
        critic=critic_value_op,
        clip_epsilon=0.2,
        entropy_bonus=True,
        entropy_coeff=0.01,
        critic_coeff=1.0,
    )

    optimizer = torch.optim.Adam(unique_params(actor, critic_value_op), lr=2e-4)

    # ---------- OPTIONAL LOAD ----------
    # Counters for naming checkpoints
    run_batch = 1
    global_batch = 0

    if args.resume_from:
        ckpt_path = Path(args.resume_from).expanduser().resolve()
        ckpt = _load_snapshot(
            ckpt_path,
            device,
            actor,
            critic_value_op,
            optimizer,
            load_optimizer=True,
            strict=strict_load,
            restore_rng=True,
        )
        # continue numbering
        run_batch = int(ckpt.get("run_batch", 0)) + 1
        global_batch = int(ckpt.get("global_batch", ckpt.get("run_batch", 0))) + 1

    elif args.init_from:
        ckpt_path = Path(args.init_from).expanduser().resolve()
        _load_snapshot(
            ckpt_path,
            device,
            actor,
            critic_value_op,
            optimizer=None,  # fresh optimizer
            load_optimizer=False,
            strict=strict_load,
            restore_rng=False,
        )
        run_batch = 1
        global_batch = 0

    if args.evaluate:
        stats = evaluate_actor(env, actor, episodes=1000, deterministic=True)
        print(json.dumps(stats, indent=2))

        # optional: write next to snapshots (safe even if you don't care)
        (run_dir / "eval.json").write_text(json.dumps(stats, indent=2))

        return

    # ---------- COLLECTOR ----------
    collector = SyncDataCollector(
        env,
        actor,
        frames_per_batch=2048 * 4,
        total_frames=2048 * 25 * 20,
        device=device,
    )

    ppo_epochs = 4
    minibatch_size = 2048  # frames_per_batch % mini_batch == 0

    batch_collector_progress = tqdm(
        enumerate(collector, start=global_batch),
        total=collector.total_frames / collector.frames_per_batch,
        desc=f"PPO batches [{run_dir.name}]",
    )

    # Keep a minimal cfg dict in checkpoints (add more if you want)
    cfg: Dict[str, Any] = {
        "frames_per_batch": collector.frames_per_batch,
        "total_frames": collector.total_frames,
        "ppo_epochs": ppo_epochs,
        "minibatch_size": minibatch_size,
        "lr": 1e-4,
        "gamma": 0.99,
        "lambda": 0.95,
    }

    for i, batch in batch_collector_progress:
        # Compute advantage + value targets ONCE for this rollout
        advantage(batch)

        # Flatten [time, env] into one dimension (TorchRL collectors often return 2D batch shapes)
        batch = batch.reshape(-1)

        N = batch.shape[0]
        assert N >= minibatch_size, f"Batch too small: N={N}, minibatch_size={minibatch_size}"

        # PPO multiple epochs over same data
        for epoch in tqdm(range(ppo_epochs), leave=False, desc="Epochs"):
            # shuffle indices each epoch
            idx = torch.randperm(N, device=batch.device)
            shuffled = batch[idx]

            mini_batch_iter = range(0, N, minibatch_size)
            mini_batch_progress = tqdm(mini_batch_iter, leave=False, desc="Mini batches")
            # iterate minibatches
            for start in mini_batch_progress:
                sub = shuffled[start : start + minibatch_size]

                losses = loss_module(sub)

                if (not torch.isfinite(losses["loss_objective"])) or (not torch.isfinite(losses["loss_critic"])):
                    continue

                loss = losses["loss_objective"] + losses["loss_critic"]
                if "loss_entropy" in losses.keys() and torch.isfinite(losses["loss_entropy"]):
                    loss = loss + losses["loss_entropy"]

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unique_params(actor, critic_value_op), 1.0)
                optimizer.step()

        log: Dict[str, Any] = {
            "iter": int(i),
            "run_batch": int(run_batch),
            "loss": float(loss.item()),  # type: ignore
            "loss_pi": float(losses["loss_objective"].item()),  # type: ignore
            "loss_v": float(losses["loss_critic"].item()),  # type: ignore
        }
        if "entropy" in losses.keys():  # type: ignore
            log["entropy"] = float(losses["entropy"].item())  # type: ignore

        print(log)

        # ---------- SAVE AFTER EVERY COLLECTOR BATCH ----------
        saved = _save_snapshot(
            run_dir=run_dir,
            run_batch=run_batch,
            global_batch=i,
            actor=actor,
            critic_value_op=critic_value_op,
            optimizer=optimizer,
            cfg=cfg,
            extra={"log": log},
        )
        batch_collector_progress.set_postfix_str(f"saved={saved.name}")

        run_batch += 1


if __name__ == "__main__":
    main()
