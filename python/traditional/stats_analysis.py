#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import os

import coloredlogs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kstest, normaltest, shapiro, ks_2samp, chisquare
import seaborn as sns

from train_model import Config, prepare_data

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO, fmt="[%(asctime)s] %(name)s [%(levelname)s] %(message)s")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment")
    parser.add_argument("-t", "--trials")
    parser.add_argument("-g", "--use_group_stats", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = os.path.join("results", f"{args.experiment}_{args.trials}_{args.use_group_stats}")
    os.makedirs(output_dir, exist_ok=True)

    # Set up config
    # config = Config(
    #     experiment=args.experiment,
    #     trials=args.trials,
    #     group_pairs=False,
    #     use_group_stats=args.use_group_stats,
    #     use_clustering=False,
    #     separate=None,
    # )

    # df = prepare_data(config)
    stats_cols = [
        "ImagePair",
        "Examiner",
        "Prefix",
        "Mating",
        "Outcome",
        "Conclusion-Simple",
        "EMDLeftCToRightCSelfDeciding",
        "Pct Deciding",
        "Pct NA",
        "Pct A-Left Fixations",
        "Phase C Time",
        "Num Fixations",
        "Pct C-Left Fixations",
        "Pct C-Right Fixations",
        "Phase A Time",
        "AllIndvClustLinksFoundBW60",
        "Switched",
        "FixationsBeforeSwitch",
        "C-Left Fix Variance",
        "C-Right Fix Variance",
        "A-Left Fix Variance",
        "PctClarBlue",
        "EMDLeftCToRightCSelf",
        "Pct Scanning",
        "PctClarRedYellow",
        "PctClarGreen",
    ]
    df = pd.read_csv(os.path.join("data", "cleaned_data.csv"), usecols=stats_cols)

    df = df[~df["Outcome"].isin(["NV", "FP"])]
    df = df[df["Prefix"] != 0]

    metadata_cols = ["Mating", "Conclusion-Simple", "Outcome", "Examiner", "ImagePair", "Prefix"]
    metadata = df[metadata_cols]
    df = df.drop(columns=metadata_cols)
    ranked = df.rank(pct=True)
    # ranked = ranked.apply(pd.cut, bins=[0, 0.2, 0.4, 0.6, 0.8, 1])

    df = metadata.merge(ranked, left_index=True, right_index=True)

    results_df = pd.DataFrame({"ImagePair": df["ImagePair"].unique()})
    for feature in df.columns:
        if feature in metadata_cols:
            continue

        # num_trials_per_bin = df.groupby(["ImagePair", feature]).size().unstack()
        pvalues = (
            df.groupby("ImagePair")[feature]
            .apply(func=lambda x: kstest(x, cdf="uniform")[1])
            .reset_index()
        )

        results_df = results_df.merge(pvalues, on="ImagePair")

    results_df = results_df.set_index(keys="ImagePair")
    tests_passed = results_df.apply(lambda x: (x < 0.05).sum())
    results_df = results_df.reindex(tests_passed.sort_values().index, axis=1)

    fig, ax = plt.subplots(figsize=(14, 14))

    sns.heatmap(
        results_df.T,
        square=True,
        cmap=["red", "green"],
        center=0.05,
        ax=ax,
        linewidths=2,
        linecolor="black",
    )
    plt.tight_layout()
    plt.savefig(os.path.join("visualizations", "kstest_by_imagepair.jpg"))
    plt.close()


if __name__ == "__main__":
    main()
