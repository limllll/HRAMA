#!/usr/bin/env python
"""Cluster RSS-reference-avoided collision archives from one or both scenarios."""

import argparse
import os

from macad_gym.rss_analysis.clustering import (
    cluster_records,
    load_attributable_records,
    write_cluster_outputs,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", nargs="+", help="Counterfactual JSONL archive(s)")
    parser.add_argument("--output-dir", default="rss_failure_clusters")
    parser.add_argument("--eps", type=float, default=1.25)
    parser.add_argument("--min-samples", type=int, default=3)
    args = parser.parse_args()

    records = load_attributable_records(args.archives)
    labels, metrics = cluster_records(
        records, eps=args.eps, min_samples=args.min_samples)
    assignments, summary = write_cluster_outputs(
        records, labels, metrics, args.output_dir)
    print("Reference-avoided collision samples: {}".format(len(records)))
    print("Failure clusters: {}".format(metrics["cluster_count"]))
    print("Noise samples: {}".format(metrics["noise_count"]))
    print("Assignments: {}".format(os.path.abspath(assignments)))
    print("Metrics: {}".format(os.path.abspath(summary)))


if __name__ == "__main__":
    main()
