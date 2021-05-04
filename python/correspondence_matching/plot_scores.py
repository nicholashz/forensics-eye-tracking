#!/usr/bin/env python
# coding: utf-8

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deciding_only", action="store_true")
    parser.add_argument("--bandwidth", type=int, default=66)

    parser.add_argument("--weight_clusters", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    # Read data
    weighted_str = "weighted" if args.weight_clusters else "nonweighted"
    output_dir = os.path.join(
        "correspondence_matching",
        "data",
        f"BW{args.bandwidth}_DecidingOnly{args.deciding_only}",
        weighted_str,
    )
    scores_path = os.path.join(output_dir, "scores.csv",)
    df = pd.read_csv(scores_path)
    plt.style.use("seaborn")
    outcome_order = ["TP", "TN", "FN", "Inc", "FP"]

    # Normalize
    # df["CorrespScore"] = df.groupby("NumClusters")["CorrespScore"].transform(
    #     lambda x: (x - x.mean()) / x.std()
    # )
    # df["CorrespScore"] = np.nan_to_num(df["CorrespScore"])

    # Boxplot
    sns.boxplot(x="Outcome", y="CorrespScore", data=df, order=outcome_order)
    plt.savefig(os.path.join(output_dir, "score_by_outcome.jpg"))

    # Barplot by group
    plt.clf()
    num_groups = 5
    df["Grouping"] = pd.qcut(df["CorrespScore"], num_groups)
    # df["Grouping"] = df["Grouping"].cat.add_categories("null")
    # df["Grouping"] = df["Grouping"].fillna("null")

    grouping_counts = (
        df.groupby(["Outcome", "Grouping"])
        .size()
        .reset_index()
        .pivot(columns="Outcome", index="Grouping", values=0)
    )
    grouping_counts = grouping_counts[grouping_counts.columns.reindex(outcome_order)[0]]
    grouping_counts.plot(kind="bar", stacked=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "outcome_counts.jpg"))


if __name__ == "__main__":
    main()
