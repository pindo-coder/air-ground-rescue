"""Train a CPU-friendly local policy with BC or aggregated DAgger episodes."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CONTROLLERS = Path(__file__).resolve().parents[1] / "controllers"
for name in ("uav_mapper", "uav_map_rover"):
    sys.path.insert(0, str(CONTROLLERS / name))
from local_policy import MultimodalPolicy, decode_actions, load_policy
from policy_features import ACTION_SCALE, NAV_SIZE, RANGE_SHAPE, RGBD_SHAPE, SCHEMA_VERSION


def scenario_key(condition):
    return tuple(condition[name] for name in ("world_sha256", "goal_id", "map_sha256"))


def load_episode(run):
    data_path = run / "rover/imitation"
    manifest = json.loads((data_path / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if manifest["feature_schema"] != SCHEMA_VERSION or not manifest["samples"]:
        raise ValueError(f"Episode is empty or has an incompatible feature schema: {run}")
    columns = {"rgbd": RGBD_SHAPE, "ranges": RANGE_SHAPE, "state": (NAV_SIZE,),
               "expert_vw": (2,), "expert_mode": ()}
    parts = {name: [] for name in columns}
    sources = Counter()
    applied = 0
    for filename in manifest["shards"]:
        with np.load(data_path / filename, allow_pickle=False) as shard:
            n = len(shard["time_s"])
            for name, shape in columns.items():
                values = shard[name]
                if values.shape != (n, *shape) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid {name} in {run.name}/{filename}")
                dtype = np.int64 if name == "expert_mode" else np.float32
                parts[name].append(values.astype(dtype, copy=False))
            source = shard["action_source"]
            sources.update(source.tolist())
            learner_moved = (source == "learner") & (np.abs(shard["executed_vw"]) > 1e-4).any(axis=1)
            applied += int(learner_moved.sum())
    arrays = tuple(np.concatenate(parts[name], axis=0) for name in columns)
    if len(arrays[-1]) != manifest["samples"] or len(sources) == 0:
        raise ValueError(f"Episode sample count does not match its manifest: {run}")
    if np.any((arrays[-1] < 0) | (arrays[-1] > 2)) or np.any(np.abs(arrays[-2]) > ACTION_SCALE + 1e-5):
        raise ValueError(f"Invalid action labels: {run}")
    if applied != manifest["decisions"].get("learner_nonzero_applied", 0):
        raise ValueError(f"Learner action count does not match the recorded episode: {run}")
    metadata = dict(run=str(run), run_id=summary["run_id"],
                    condition=summary["condition"], samples=manifest["samples"],
                    checkpoint_sha256=summary["local_policy"]["checkpoint_sha256"],
                    policy=manifest["policy"], expert_mix=manifest["expert_mix"],
                    sources=dict(sources), learner_nonzero_applied=applied)
    return arrays, metadata


def assemble_dataset(episodes):
    arrays = [arrays for arrays, _ in episodes]
    return TensorDataset(*(torch.from_numpy(np.concatenate(channel)) for channel in zip(*arrays)))


def train(train_runs, val_runs, output, epochs=30, batch_size=64, lr=0.002, seed=17,
          init=None, threads=2):
    if epochs < 1 or batch_size < 1 or threads < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Epochs, batch size, threads and learning rate must be positive.")
    if not 0 <= seed < 2**32:
        raise ValueError("--seed must be an unsigned 32-bit integer.")
    train_paths = [Path(run).resolve() for run in train_runs]
    val_paths = [Path(run).resolve() for run in val_runs]
    all_paths = train_paths + val_paths
    if not train_paths or not val_paths or len(set(all_paths)) != len(all_paths):
        raise ValueError("Use distinct rollout directories for training and validation, without duplicates.")
    output = Path(output).resolve()
    if output.suffix != ".pt":
        raise ValueError("Save the model to a .pt file.")
    if init is not None and Path(init).resolve() == output:
        raise ValueError("Give the DAgger checkpoint its own path so the input checkpoint is retained.")

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    train_episodes = [load_episode(path) for path in train_paths]
    val_episodes = [load_episode(path) for path in val_paths]
    train_conditions = {scenario_key(meta["condition"]) for _, meta in train_episodes}
    val_conditions = {scenario_key(meta["condition"]) for _, meta in val_episodes}
    if train_conditions & val_conditions:
        raise ValueError("Validation must use a different world/goal/map combination from training.")
    train_meta = [meta for _, meta in train_episodes]
    val_meta = [meta for _, meta in val_episodes]
    run_ids = [meta["run_id"] for meta in train_meta + val_meta]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("A copied episode still counts as the same rollout. Remove duplicate run IDs.")
    init_sha = bc_sha = None
    if init is None:
        if any(meta["policy"] != "rule" for meta in train_meta + val_meta):
            raise ValueError("BC pretraining expects expert-driven training and validation episodes.")
        model, stage = MultimodalPolicy(), "bc"
    else:
        init = Path(init).resolve()
        init_sha = hashlib.sha256(init.read_bytes()).hexdigest()
        model, previous = load_policy(init)
        bc_sha = init_sha if previous["stage"] == "bc" else previous["bc_sha256"]
        prior_training = {
            meta.get("run_id") or meta["run"].replace("\\", "/").rsplit("/", 1)[-1]
            for meta in previous["training_episodes"]
        }
        if not prior_training.issubset({meta["run_id"] for meta in train_meta}):
            raise ValueError("DAgger training must aggregate the previous training episodes.")
        if not any(meta["checkpoint_sha256"] == init_sha and meta["learner_nonzero_applied"] > 0
                   for meta in train_meta):
            raise ValueError("DAgger needs at least one new episode with applied nonzero learner actions from --init.")
        model, stage = model.train(), "dagger"

    training, validation = assemble_dataset(train_episodes), assemble_dataset(val_episodes)
    batches = DataLoader(training, batch_size=batch_size, shuffle=True,
                         generator=torch.Generator().manual_seed(seed))
    val_batches = DataLoader(validation, batch_size=batch_size)
    counts = torch.bincount(training.tensors[-1], minlength=3)
    weights = (counts.max() / counts.clamp(min=1)).sqrt().clamp(max=4)
    weights[counts == 0] = 1.0
    scale = torch.from_numpy(ACTION_SCALE)
    regression = nn.SmoothL1Loss(beta=0.15)
    classification = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def loss_fn(pred, logits, action, modes):
        return regression(pred, action / scale) + 0.35 * classification(logits, modes)

    def validate():
        absolute = torch.zeros(2)
        correct = 0
        loss = 0.0
        model.eval()
        with torch.inference_mode():
            for rgbd, ranges, state, action, modes in val_batches:
                pred, logits = model(rgbd, ranges, state)
                loss += float(loss_fn(pred, logits, action, modes)) * len(action)
                absolute += (decode_actions(pred, logits) - action).abs().sum(dim=0)
                correct += int((logits.argmax(dim=1) == modes).sum())
        mae_v, mae_w = (absolute / len(validation)).tolist()
        return dict(loss=loss / len(validation), mae_v_mps=mae_v, mae_w_rad_s=mae_w,
                    mode_accuracy=correct / len(validation))

    best_loss = math.inf
    best = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for rgbd, ranges, state, action, modes in batches:
            optimizer.zero_grad(set_to_none=True)
            pred, logits = model(rgbd, ranges, state)
            loss = loss_fn(pred, logits, action, modes)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * len(action)
        val = validate()
        history.append(dict(epoch=epoch, train_loss=train_loss / len(training), validation=val))
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            best = dict(epoch=epoch, validation=val,
                        model_state_dict={key: value.detach().clone() for key, value in model.state_dict().items()})
    if best is None:
        raise RuntimeError("Training did not produce a finite validation checkpoint.")

    metadata = dict(feature_schema=SCHEMA_VERSION, stage=stage, seed=seed,
                    init_sha256=init_sha, bc_sha256=bc_sha,
                    parameter_count=sum(p.numel() for p in model.parameters()),
                    training_episodes=train_meta, validation_episodes=val_meta,
                    training_samples=len(training), validation_samples=len(validation),
                    class_counts=counts.tolist(), epochs=epochs, batch_size=batch_size, lr=lr)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".tmp")
    torch.save(dict(best, **metadata), temp)
    temp.replace(output)

    metrics = dict(metadata, best_epoch=best["epoch"], best_validation=best["validation"], history=history)
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dict(checkpoint=str(output), stage=stage, train_samples=len(training),
                          val_samples=len(validation), best_epoch=best["epoch"],
                          validation=best["validation"]), indent=2), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs="+", type=Path, required=True, dest="train_runs")
    parser.add_argument("--val", nargs="+", type=Path, required=True, dest="val_runs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init", type=Path, help="Previous checkpoint for aggregated DAgger training.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=2)
    train(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
