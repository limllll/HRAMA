#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Single-stage joint CARLA-domain adaptation for image and LiDAR encoders."""

from __future__ import absolute_import, division, print_function

import argparse
import collections
import csv
import json
import os
import random
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from macad_gym.agents.multimodal_encoder.dataset import MultimodalPairDataset, load_manifest, split_entries_by_episode
from macad_gym.agents.multimodal_encoder.initialization import load_image_encoder_initialization, load_lidar_encoder_initialization
from macad_gym.agents.multimodal_encoder.models import (
    MultimodalAdaptationModel,
    local_region_distillation,
    masked_image_reconstruction_loss,
)
from macad_gym.agents.multimodal_encoder.runtime import resolve_device

DEFAULT_IMAGE_INITIALIZATION = os.path.join(PROJECT_ROOT, "src", "macad_gym", "agents", "imitationmodel", "model", "model.ckpt-450000")
DEFAULT_LIDAR_INITIALIZATION = os.path.join(PROJECT_ROOT, "src", "macad_gym", "PointNet2", "log", "classification", "pointnet2_msg_normals", "checkpoints", "best_model.pth")

RUN_CONFIG = {
    "dataset_dir": os.path.join(PROJECT_ROOT, "artifacts", "multimodal_dataset"),
    "output_dir": os.path.join(PROJECT_ROOT, "artifacts", "multimodal_joint_adaptation"),
    # Epochs to execute per Run click. With 5, consecutive completed runs cover
    # 1-5, 6-10, and so on. An interrupted run finishes its original range.
    "epochs": 5,
    "resume": "auto",
    "reset_early_stop": False,
    "alignment_warmup_epochs": 0,
    "batch_size": 8,
    "num_workers": 0,
    "device": "cuda:0",
    "seed": 2026,
    "held_out_towns": ["Town05"],
    "image_init_checkpoint": DEFAULT_IMAGE_INITIALIZATION,
    "lidar_init_checkpoint": DEFAULT_LIDAR_INITIALIZATION,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=RUN_CONFIG["dataset_dir"])
    parser.add_argument("--output-dir", default=RUN_CONFIG["output_dir"])
    parser.add_argument(
        "--epochs", type=int, default=RUN_CONFIG["epochs"],
        help="Number of epochs for this run; auto-resume continues numbering.",
    )
    parser.add_argument(
        "--resume", default=RUN_CONFIG["resume"],
        help="Resume from a checkpoint path, or use 'auto' for output_dir/latest.pt.",
    )
    parser.add_argument(
        "--reset-early-stop", action="store_true",
        default=RUN_CONFIG["reset_early_stop"],
        help="Reset the restored stale-epoch counter when intentionally extending a run.",
    )
    parser.add_argument("--batch-size", type=int, default=RUN_CONFIG["batch_size"])
    parser.add_argument("--num-workers", type=int, default=RUN_CONFIG["num_workers"])
    parser.add_argument("--device", default=RUN_CONFIG["device"])
    parser.add_argument("--seed", type=int, default=RUN_CONFIG["seed"])
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--held-out-towns", nargs="*", default=RUN_CONFIG["held_out_towns"])
    parser.add_argument("--image-height", type=int, default=88)
    parser.add_argument("--image-width", type=int, default=200)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--num-ray-queries", type=int, default=384)
    parser.add_argument("--lidar-range", type=float, default=20.0)
    parser.add_argument("--voxel-size", type=float, default=0.1)
    parser.add_argument("--image-mask-ratio", type=float, default=0.4)
    parser.add_argument("--image-mask-patch", type=int, default=8)
    parser.add_argument("--lidar-mask-ratio", type=float, default=0.5)
    parser.add_argument("--encoder-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--image-weight", type=float, default=1.0)
    parser.add_argument("--image-l1-weight", type=float, default=1.0)
    parser.add_argument("--image-ssim-weight", type=float, default=0.2)
    parser.add_argument("--image-edge-weight", type=float, default=0.1)
    parser.add_argument(
        "--image-decoder-version", choices=("legacy", "enhanced"),
        default="enhanced",
        help="Training-only image decoder; the PPO runtime loads encoders only.",
    )
    parser.add_argument("--ray-weight", type=float, default=0.2)
    parser.add_argument("--alignment-weight", type=float, default=0.1,
                        help="Set to 0 for the modality-internal ablation.")
    parser.add_argument(
        "--alignment-warmup-epochs", type=int,
        default=RUN_CONFIG["alignment_warmup_epochs"],
        help="Alignment-weight warmup length; 0 keeps the full weight from epoch 1.",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--region-rows", type=int, default=2)
    parser.add_argument("--region-columns", type=int, default=4)
    parser.add_argument("--minimum-region-points", type=int, default=2)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--image-init-checkpoint", default=RUN_CONFIG["image_init_checkpoint"])
    parser.add_argument("--lidar-init-checkpoint", default=RUN_CONFIG["lidar_init_checkpoint"])
    parser.add_argument("--random-init", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EpisodeInterleavedBatchSampler(object):
    def __init__(self, entries, batch_size, seed=2026, shuffle=True, drop_last=False):
        self.entries = list(entries)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    @staticmethod
    def group(entry):
        return "{}:{}".format(entry.get("scenario", "unknown"), entry.get("episode", -1))

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        groups = collections.OrderedDict()
        for index, entry in enumerate(self.entries):
            groups.setdefault(self.group(entry), []).append(index)
        for indices in groups.values():
            if self.shuffle:
                rng.shuffle(indices)
        order, active = [], list(groups)
        while active:
            if self.shuffle:
                rng.shuffle(active)
            next_active = []
            for name in active:
                order.append(groups[name].pop())
                if groups[name]:
                    next_active.append(name)
            active = next_active
        for start in range(0, len(order), self.batch_size):
            batch = order[start:start + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self):
        return len(self.entries) // self.batch_size if self.drop_last else int(np.ceil(len(self.entries) / float(self.batch_size)))


def make_loader(dataset, entries, args, train):
    return DataLoader(
        dataset,
        batch_sampler=EpisodeInterleavedBatchSampler(entries, args.batch_size, args.seed, shuffle=train, drop_last=train),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def create_loaders(args):
    entries = load_manifest(args.dataset_dir)
    held_out_prefixes = tuple(town + "_" for town in args.held_out_towns)
    test_entries = [e for e in entries if str(e.get("scenario", "")).startswith(held_out_prefixes)]
    fit_entries = [e for e in entries if e not in test_entries]
    train_entries, val_entries = split_entries_by_episode(fit_entries, val_ratio=args.val_ratio, seed=args.seed)
    common = dict(
        dataset_dir=args.dataset_dir,
        image_size=(args.image_height, args.image_width),
        num_points=args.num_points,
        lidar_range=args.lidar_range,
        voxel_size=args.voxel_size,
        lidar_mask_ratio=args.lidar_mask_ratio,
        num_ray_queries=args.num_ray_queries,
        seed=args.seed,
    )
    train_dataset = MultimodalPairDataset(entries=train_entries, augment=True, **common)
    val_dataset = MultimodalPairDataset(entries=val_entries, augment=False, **common)
    test_dataset = MultimodalPairDataset(entries=test_entries, augment=False, **common)
    return (
        make_loader(train_dataset, train_entries, args, True),
        make_loader(val_dataset, val_entries, args, False),
        make_loader(test_dataset, test_entries, args, False),
        (len(train_entries), len(val_entries), len(test_entries)),
    )


def mask_images(images, ratio, patch_size, paths=None, seed=2026):
    batch, _, height, width = images.shape
    grid_h = int(np.ceil(height / float(patch_size)))
    grid_w = int(np.ceil(width / float(patch_size)))
    masks = []
    for index in range(batch):
        generator = torch.Generator(device="cpu")
        if paths is None:
            generator.manual_seed(random.randint(0, 2 ** 31 - 1))
        else:
            import hashlib
            digest = hashlib.sha1((str(paths[index]) + ":" + str(seed)).encode("utf-8")).hexdigest()
            generator.manual_seed(int(digest[:8], 16))
        masks.append(torch.rand(1, grid_h, grid_w, generator=generator) < float(ratio))
    mask = torch.stack(masks).to(images.device, dtype=images.dtype)
    mask = F.interpolate(mask, size=(height, width), mode="nearest")
    return images * (1.0 - mask), mask


def ray_loss_and_stats(logits, labels):
    positive = torch.clamp(labels.sum(), min=1.0)
    negative = labels.numel() - positive
    pos_weight = torch.clamp(negative / positive, min=1.0, max=10.0)
    loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    predictions, truth = torch.sigmoid(logits) >= 0.5, labels >= 0.5
    tp = (predictions & truth).sum().float()
    fp = (predictions & ~truth).sum().float()
    fn = (~predictions & truth).sum().float()
    precision = tp / torch.clamp(tp + fp, min=1.0)
    recall = tp / torch.clamp(tp + fn, min=1.0)
    f1 = 2.0 * precision * recall / torch.clamp(precision + recall, min=1.0e-6)
    return loss, f1


def compute_losses(model, batch, args, device, training, alignment_weight):
    images = batch["image"].to(device, dtype=torch.float32, non_blocking=True)
    image_targets = batch["image_target"].to(device, dtype=torch.float32, non_blocking=True)
    full_points = batch["points"].to(device, dtype=torch.float32, non_blocking=True)
    masked_points = batch["masked_points"].to(device, dtype=torch.float32, non_blocking=True)
    queries = batch["ray_queries"].to(device, dtype=torch.float32, non_blocking=True)
    labels = batch["ray_labels"].to(device, dtype=torch.float32, non_blocking=True)
    masked_images, mask = mask_images(
        images, args.image_mask_ratio, args.image_mask_patch,
        paths=None if training else batch["path"], seed=args.seed,
    )
    reconstruction = model.forward_reconstruction(masked_images, masked_points, queries)
    image_loss, image_components = masked_image_reconstruction_loss(
        reconstruction["image_reconstruction"], image_targets, mask,
        l1_weight=args.image_l1_weight,
        ssim_weight=args.image_ssim_weight,
        edge_weight=args.image_edge_weight,
    )
    ray_loss, ray_f1 = ray_loss_and_stats(
        reconstruction["ray_occupancy_logits"], labels
    )
    output = model.forward_alignment(images, full_points)
    local_loss, image_projection, lidar_projection = local_region_distillation(
        output["image_local_map"], output["lidar_local_xyz"], output["lidar_local_features"],
        model.image_local_projector, model.lidar_local_projector,
        temperature=args.temperature, rows=args.region_rows, columns=args.region_columns,
        minimum_points=args.minimum_region_points, lidar_range=args.lidar_range,
    )
    total = (
        args.image_weight * image_loss
        + args.ray_weight * ray_loss
        + float(alignment_weight) * local_loss
    )
    regions = 0 if image_projection is None else int(image_projection.shape[0])
    return {
        "total": total,
        "image": image_loss,
        "image_l1": image_components["l1"],
        "image_ssim": image_components["ssim"],
        "image_edge": image_components["edge"],
        "ray": ray_loss,
        "ray_f1": ray_f1,
        "local": local_loss,
        "regions": regions,
    }


def run_epoch(model, loader, args, device, alignment_weight, optimizer=None):
    training = optimizer is not None
    model.train(training)
    sums = {
        key: 0.0 for key in (
            "total", "image", "image_l1", "image_ssim", "image_edge",
            "ray", "ray_f1", "local",
        )
    }
    samples, region_total = 0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            if training:
                optimizer.zero_grad()
            losses = compute_losses(
                model, batch, args, device, training, alignment_weight
            )
            if training:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=10.0)
                optimizer.step()
            count = int(batch["image"].shape[0])
            samples += count
            region_total += int(losses["regions"])
            for key in sums:
                sums[key] += float(losses[key].detach().cpu()) * count
    metrics = {key: value / max(1, samples) for key, value in sums.items()}
    metrics["regions_per_frame"] = region_total / float(max(1, samples))
    return metrics


def configure_optimizer(model, args):
    encoder_parameters = (
        list(model.image_encoder.parameters())
        + list(model.lidar_encoder.parameters())
    )
    head_parameters = (
        list(model.image_decoder.parameters())
        + list(model.ray_occupancy_head.parameters())
        + list(model.image_local_projector.parameters())
        + list(model.lidar_local_projector.parameters())
    )
    return torch.optim.AdamW(
        [{"params": encoder_parameters, "lr": args.encoder_learning_rate}, {"params": head_parameters, "lr": args.head_learning_rate}],
        weight_decay=args.weight_decay,
    )


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def checkpoint_payload(
    model, optimizer, args, epoch, best_value, stale_epochs,
    run_target_epoch, run_complete,
):
    return {
        "format_version": 3,
        "method": "joint_masked_reconstruction+ray_occupancy+local_alignment",
        "epoch": int(epoch),
        "best_val_loss": float(best_value),
        "stale_epochs": int(stale_epochs),
        "run_target_epoch": int(run_target_epoch),
        "run_complete": bool(run_complete),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": capture_rng_state(),
        "image_encoder": model.image_encoder.state_dict(),
        "lidar_encoder": model.lidar_encoder.state_dict(),
        "model_config": {
            "image_height": args.image_height, "image_width": args.image_width,
            "image_feature_dim": 512, "lidar_feature_dim": 256,
            "num_points": args.num_points, "lidar_range": args.lidar_range,
            "voxel_size": args.voxel_size, "projection_dim": 128,
            "image_decoder_version": args.image_decoder_version,
        },
        "training_config": vars(args),
    }


def append_log(path, epoch, alignment_weight, train, val, elapsed):
    exists = os.path.isfile(path)
    fields = [
        "epoch", "alignment_weight", "train_total", "val_total",
        "train_image", "val_image", "train_image_l1", "val_image_l1",
        "train_image_ssim", "val_image_ssim", "train_image_edge",
        "val_image_edge", "train_ray", "val_ray", "train_ray_f1",
        "val_ray_f1", "train_local", "val_local",
        "train_regions_per_frame", "val_regions_per_frame", "seconds",
    ]
    with open(path, "a") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        row = {"epoch": epoch, "alignment_weight": alignment_weight, "seconds": elapsed}
        for prefix, values in (("train", train), ("val", val)):
            for key in (
                "total", "image", "image_l1", "image_ssim", "image_edge",
                "ray", "ray_f1", "local", "regions_per_frame",
            ):
                row[prefix + "_" + key] = values[key]
        writer.writerow(row)


def resolve_resume_checkpoint(args):
    value = str(args.resume or "").strip()
    if not value:
        return None
    if value.lower() == "auto":
        path = os.path.join(args.output_dir, "latest.pt")
        if not os.path.isfile(path):
            return None
    else:
        path = os.path.abspath(os.path.expanduser(value))
    if not os.path.isfile(path):
        raise IOError("Resume checkpoint not found: {}".format(path))
    return path


def ensure_new_run_does_not_overwrite(output_dir, resume_path):
    if resume_path:
        return
    occupied = [
        os.path.join(output_dir, name)
        for name in ("best.pt", "latest.pt", "training_log.csv")
        if os.path.exists(os.path.join(output_dir, name))
    ]
    if occupied:
        raise RuntimeError(
            "Output directory already contains a training run. Use "
            "--resume auto to continue it, or choose a new --output-dir: {}"
            .format(", ".join(occupied))
        )


def train_joint(model, train_loader, val_loader, args, device, output_dir, resume_path=None):
    optimizer = configure_optimizer(model, args)
    best, stale, start_epoch = float("inf"), 0, 1
    run_target_epoch = int(args.epochs)
    best_path = os.path.join(output_dir, "best.pt")
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        if int(checkpoint.get("format_version", 0)) < 3:
            raise ValueError(
                "Only single-stage format_version >= 3 checkpoints can resume."
            )
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" not in checkpoint:
            raise ValueError("Resume checkpoint does not contain optimizer_state")
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best = float(checkpoint.get("best_val_loss", float("inf")))
        stale = int(checkpoint.get("stale_epochs", 0))
        if args.reset_early_stop:
            stale = 0
        restore_rng_state(checkpoint.get("rng_state"))
        if not os.path.isfile(best_path):
            torch.save(checkpoint, best_path)
        if stale >= args.early_stop_patience and not args.reset_early_stop:
            print(
                "run already reached early stopping; use --reset-early-stop "
                "only if you intentionally want to continue"
            )
            return best_path
        previous_target = int(
            checkpoint.get("run_target_epoch", start_epoch - 1)
        )
        previous_complete = bool(checkpoint.get("run_complete", True))
        if not previous_complete and start_epoch <= previous_target:
            run_target_epoch = previous_target
            resume_mode = "finish interrupted run"
        else:
            run_target_epoch = start_epoch - 1 + int(args.epochs)
            resume_mode = "new epoch block"
        print(
            "resumed {} at epoch {}; {} target={} best={:.6f} stale={}".format(
                resume_path, start_epoch - 1, resume_mode,
                run_target_epoch, best, stale
            )
        )
    for epoch in range(start_epoch, run_target_epoch + 1):
        started = time.time()
        if args.alignment_warmup_epochs > 0:
            warmup = min(1.0, epoch / float(args.alignment_warmup_epochs))
        else:
            warmup = 1.0
        active_alignment_weight = args.alignment_weight * warmup
        train_metrics = run_epoch(
            model, train_loader, args, device, active_alignment_weight, optimizer
        )
        val_metrics = run_epoch(
            model, val_loader, args, device, args.alignment_weight
        )
        elapsed = time.time() - started
        improved = val_metrics["total"] < best
        if improved:
            best, stale = val_metrics["total"], 0
        else:
            stale += 1
        run_complete = (
            epoch >= run_target_epoch or stale >= args.early_stop_patience
        )
        payload = checkpoint_payload(
            model, optimizer, args, epoch, best, stale,
            run_target_epoch, run_complete,
        )
        if improved:
            torch.save(payload, best_path)
        torch.save(payload, os.path.join(output_dir, "latest.pt"))
        append_log(os.path.join(output_dir, "training_log.csv"), epoch, active_alignment_weight, train_metrics, val_metrics, elapsed)
        print("epoch {:03d} train={:.6f} val={:.6f} image={:.6f} ray={:.6f} ray_f1={:.3f} align={:.6f} align_w={:.4f} regions={:.2f} time={:.1f}s{}".format(
            epoch, train_metrics["total"], val_metrics["total"], val_metrics["image"], val_metrics["ray"], val_metrics["ray_f1"], val_metrics["local"], active_alignment_weight, val_metrics["regions_per_frame"], elapsed, " best" if improved else ""))
        if stale >= args.early_stop_patience:
            print("early stop after {} stale epochs".format(stale))
            break
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return best_path


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    resume_path = resolve_resume_checkpoint(args)
    ensure_new_run_does_not_overwrite(args.output_dir, resume_path)
    if resume_path:
        resume_metadata = torch.load(resume_path, map_location="cpu")
        args.image_decoder_version = resume_metadata.get(
            "model_config", {}
        ).get("image_decoder_version", "legacy")
    train_loader, val_loader, test_loader, sizes = create_loaders(args)
    print("split train={} validation={} Town05={}".format(*sizes))
    model = MultimodalAdaptationModel(
        image_decoder_version=args.image_decoder_version
    ).to(device)
    if not resume_path and not args.random_init:
        load_image_encoder_initialization(model.image_encoder, args.image_init_checkpoint)
        load_lidar_encoder_initialization(model.lidar_encoder, args.lidar_init_checkpoint)
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)
    final_path = train_joint(
        model, train_loader, val_loader, args, device, args.output_dir,
        resume_path=resume_path,
    )
    final_checkpoint = torch.load(final_path, map_location=device)
    model.load_state_dict(final_checkpoint["model_state"])
    test_metrics = run_epoch(
        model, test_loader, args, device, args.alignment_weight
    )
    with open(os.path.join(args.output_dir, "Town05_test_metrics.json"), "w") as handle:
        json.dump(test_metrics, handle, indent=2, sort_keys=True)
    print("Town05 held-out:", test_metrics)
    print("final checkpoint:", os.path.join(args.output_dir, "best.pt"))


if __name__ == "__main__":
    main()
