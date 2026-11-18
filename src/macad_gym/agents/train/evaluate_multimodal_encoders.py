#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate single-stage joint CARLA multimodal encoder adaptation."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import os
import sys
from types import SimpleNamespace

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
TRAIN_DIR = os.path.dirname(__file__)
for path in (SRC_DIR, TRAIN_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from macad_gym.agents.multimodal_encoder.models import MultimodalAdaptationModel, local_region_distillation, masked_image_mse
from macad_gym.agents.multimodal_encoder.runtime import resolve_device
from train_multimodal_encoders import create_loaders, mask_images


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--visual-samples", type=int, default=6)
    return parser.parse_args()


def build_training_args(checkpoint, cli):
    config = dict(checkpoint.get("training_config", {}))
    config["dataset_dir"] = cli.dataset_dir or config.get("dataset_dir")
    config["batch_size"] = cli.batch_size
    config["num_workers"] = cli.num_workers
    defaults = {
        "seed": 2026, "val_ratio": 0.1, "held_out_towns": ["Town05"],
        "image_height": 88, "image_width": 200, "num_points": 1024,
        "num_ray_queries": 384, "lidar_range": 20.0, "voxel_size": 0.1,
        "lidar_mask_ratio": 0.5, "image_mask_ratio": 0.5,
        "image_mask_patch": 10, "temperature": 0.1, "region_rows": 2,
        "region_columns": 4, "minimum_region_points": 2,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)
    return SimpleNamespace(**config)


def confusion_update(logits, labels, totals):
    prediction = torch.sigmoid(logits) >= 0.5
    truth = labels >= 0.5
    totals["tp"] += int((prediction & truth).sum().item())
    totals["fp"] += int((prediction & ~truth).sum().item())
    totals["fn"] += int((~prediction & truth).sum().item())
    totals["tn"] += int((~prediction & ~truth).sum().item())


def safe_ratio(numerator, denominator):
    return float(numerator) / float(max(1, denominator))


def evaluate_split(model, loader, args, device, save_visual_path=None, visual_samples=6):
    model.eval()
    totals = {"samples": 0, "image_mse": 0.0, "local_loss": 0.0, "regions": 0,
              "positive_similarity": 0.0, "negative_similarity": 0.0,
              "positive_count": 0, "negative_count": 0, "top1": 0, "top5": 0,
              "retrieval_count": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
    visual_rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, dtype=torch.float32)
            targets = batch["image_target"].to(device, dtype=torch.float32)
            masked_points = batch["masked_points"].to(device, dtype=torch.float32)
            full_points = batch["points"].to(device, dtype=torch.float32)
            queries = batch["ray_queries"].to(device, dtype=torch.float32)
            labels = batch["ray_labels"].to(device, dtype=torch.float32)
            masked_images, image_mask = mask_images(
                images, args.image_mask_ratio, args.image_mask_patch,
                paths=batch["path"], seed=args.seed,
            )
            reconstruction = model.forward_reconstruction(
                masked_images, masked_points, queries
            )
            mse = masked_image_mse(
                reconstruction["image_reconstruction"], targets, image_mask
            )
            count = int(images.shape[0])
            totals["samples"] += count
            totals["image_mse"] += float(mse.cpu()) * count
            confusion_update(
                reconstruction["ray_occupancy_logits"], labels, totals
            )

            alignment = model.forward_alignment(images, full_points)
            local_loss, image_projection, lidar_projection = local_region_distillation(
                alignment["image_local_map"], alignment["lidar_local_xyz"],
                alignment["lidar_local_features"], model.image_local_projector,
                model.lidar_local_projector, temperature=args.temperature,
                rows=args.region_rows, columns=args.region_columns,
                minimum_points=args.minimum_region_points, lidar_range=args.lidar_range,
            )
            if image_projection is not None:
                similarity = torch.matmul(lidar_projection, image_projection.t())
                region_count = int(similarity.shape[0])
                totals["regions"] += region_count
                totals["local_loss"] += float(local_loss.cpu()) * region_count
                diagonal = similarity.diag()
                totals["positive_similarity"] += float(diagonal.sum().cpu())
                totals["positive_count"] += region_count
                off_diagonal = ~torch.eye(region_count, dtype=torch.bool, device=device)
                totals["negative_similarity"] += float(similarity[off_diagonal].sum().cpu())
                totals["negative_count"] += int(off_diagonal.sum().item())
                order = similarity.argsort(dim=1, descending=True)
                labels_index = torch.arange(region_count, device=device)
                totals["top1"] += int((order[:, 0] == labels_index).sum().item())
                totals["top5"] += int((order[:, :min(5, region_count)] == labels_index[:, None]).any(dim=1).sum().item())
                totals["retrieval_count"] += region_count

            if save_visual_path and len(visual_rows) < visual_samples:
                original = targets.detach().cpu().numpy().transpose(0, 2, 3, 1)
                masked = (masked_images / 255.0).detach().cpu().numpy().transpose(0, 2, 3, 1)
                reconstructed = reconstruction["image_reconstruction"].detach().cpu().numpy().transpose(0, 2, 3, 1)
                mask = image_mask.detach().cpu().numpy().transpose(0, 2, 3, 1)
                # The reconstruction loss supervises masked pixels only.  Keep
                # observed pixels from the input and use decoder predictions
                # only where patches were masked for a faithful MAE-style view.
                completed = original * (1.0 - mask) + reconstructed * mask
                for index in range(min(count, visual_samples - len(visual_rows))):
                    visual_rows.append((original[index], masked[index], completed[index]))

    precision = safe_ratio(totals["tp"], totals["tp"] + totals["fp"])
    recall = safe_ratio(totals["tp"], totals["tp"] + totals["fn"])
    f1 = 2.0 * precision * recall / max(1.0e-12, precision + recall)
    iou = safe_ratio(totals["tp"], totals["tp"] + totals["fp"] + totals["fn"])
    metrics = {
        "samples": totals["samples"],
        "masked_image_mse": totals["image_mse"] / max(1, totals["samples"]),
        "masked_image_psnr": -10.0 * np.log10(max(1.0e-12, totals["image_mse"] / max(1, totals["samples"]))),
        "ray_occupancy_precision": precision, "ray_occupancy_recall": recall,
        "ray_occupancy_f1": f1, "ray_occupancy_iou": iou,
        "regions_per_frame": safe_ratio(totals["regions"], totals["samples"]),
        "local_alignment_loss": safe_ratio(totals["local_loss"], totals["regions"]),
        "positive_similarity": safe_ratio(totals["positive_similarity"], totals["positive_count"]),
        "negative_similarity": safe_ratio(totals["negative_similarity"], totals["negative_count"]),
        "local_retrieval_top1": safe_ratio(totals["top1"], totals["retrieval_count"]),
        "local_retrieval_top5": safe_ratio(totals["top5"], totals["retrieval_count"]),
    }
    if save_visual_path and visual_rows:
        figure, axes = plt.subplots(len(visual_rows), 3, figsize=(12, 3 * len(visual_rows)))
        if len(visual_rows) == 1:
            axes = axes[None, :]
        for row, (original, masked, completed) in enumerate(visual_rows):
            for column, (title, image) in enumerate((("original", original), ("masked input", masked), ("masked-region completion", completed))):
                axes[row, column].imshow(np.clip(image, 0.0, 1.0))
                axes[row, column].set_title(title)
                axes[row, column].axis("off")
        figure.tight_layout()
        figure.savefig(save_visual_path, dpi=180)
        plt.close(figure)
    return metrics


def plot_curves(log_path, output_path):
    if not os.path.isfile(log_path):
        return
    rows = list(csv.DictReader(open(log_path)))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = [int(row["epoch"]) for row in rows]
    axes[0].plot(epochs, [float(row["train_total"]) for row in rows], linestyle="--", label="train")
    axes[0].plot(epochs, [float(row["val_total"]) for row in rows], label="validation")
    axes[1].plot(epochs, [float(row["val_ray_f1"]) for row in rows], label="validation")
    axes[2].plot(epochs, [float(row["val_local"]) for row in rows], label="validation")
    for axis, title in zip(axes, ("total loss", "ray occupancy F1", "local alignment loss")):
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    cli = parse_args()
    device = resolve_device(cli.device)
    checkpoint = torch.load(cli.checkpoint, map_location=device)
    args = build_training_args(checkpoint, cli)
    output_dir = cli.output_dir or os.path.join(os.path.dirname(os.path.abspath(cli.checkpoint)), "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    _, val_loader, test_loader, _ = create_loaders(args)
    decoder_version = checkpoint.get("model_config", {}).get(
        "image_decoder_version", "legacy"
    )
    model = MultimodalAdaptationModel(
        image_decoder_version=decoder_version
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    validation = evaluate_split(model, val_loader, args, device, os.path.join(output_dir, "validation_reconstruction.png"), cli.visual_samples)
    town05 = evaluate_split(model, test_loader, args, device, os.path.join(output_dir, "Town05_reconstruction.png"), cli.visual_samples)
    metrics = {"checkpoint": os.path.abspath(cli.checkpoint), "checkpoint_epoch": checkpoint.get("epoch"), "validation": validation, "Town05": town05}
    with open(os.path.join(output_dir, "metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    plot_curves(os.path.join(os.path.dirname(os.path.abspath(cli.checkpoint)), "training_log.csv"), os.path.join(output_dir, "loss_curves.png"))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("evaluation outputs:", output_dir)


if __name__ == "__main__":
    main()
