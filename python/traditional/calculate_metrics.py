#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import logging
import os
import pickle

import coloredlogs
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import Config
from metric_utils import (
    calculate_trial_metrics, save_confusion_matrix, save_roc_curve, save_feature_importance_plot
)

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO, fmt="[%(asctime)s] %(name)s [%(levelname)s] %(message)s")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", dest="experiment")
    parser.add_argument("-t", "--trials", dest="trials")
    parser.add_argument("-g", "--use_group_stats", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    results_dir = os.path.join(
        "results", "_".join((args.experiment, args.trials, str(args.use_group_stats)))
    )
    data_path = os.path.join(results_dir, "data.json")
    config_path = os.path.join(results_dir, "config.pickle")

    with open(data_path, "r") as f:
        data = json.load(f)
    with open(config_path, "rb") as f:
        config = pickle.load(f)

    logging.info("Loaded config and data from files.")

    # Set up metrics
    stats = dict()
    tprs = dict()  # tprs[class_num] is a list, with len == TRIALS, where each element is an ndarray
    mean_fpr = np.linspace(0, 1, 100)

    num_trials = len(data)
    for trial_key in tqdm(range(num_trials), "Calculating stats for each trial"):
        data_dict = data[str(trial_key)]
        trial_df = pd.DataFrame.from_dict(data_dict)
        trial_metrics, fpr, tpr = calculate_trial_metrics(df=trial_df, config=config)

        for key in trial_metrics:
            if key not in stats:
                stats[key] = []
            stats[key].append(trial_metrics[key])

            for class_num in fpr:
                if class_num not in tprs:
                    tprs[class_num] = []
                tprs[class_num].append(np.interp(mean_fpr, fpr[class_num], tpr[class_num]))
                tprs[class_num][-1][0] = 0.0

    results_data = {}
    for key in stats:
        if key == "conf_matrix":
            save_confusion_matrix(
                conf_matrices=stats[key], class_names=config.class_names, output_dir=results_dir
            )
        else:
            results_data[f"{key}_mean"] = np.mean(stats[key], axis=0)
            results_data[f"{key}_std"] = np.std(stats[key], axis=0)

    results_df = pd.DataFrame(data=results_data, index=config.class_names)

    output_path = os.path.join(results_dir, "stats.csv")
    results_df.to_csv(output_path)
    logger.info(f"Summary stats written to {output_path}")

    # Plot ROCs
    roc_auc_array = np.array(stats["roc_auc"])
    for class_num in range(len(config.class_names)):
        class_name = config.class_names[class_num]
        save_roc_curve(
            mean_fpr=mean_fpr,
            tprs=tprs[class_num],
            roc_aucs=roc_auc_array[:, class_num],
            class_name=class_name,
            output_dir=results_dir,
        )

    # Plot feature importances
    feature_importance_df = pd.read_csv(os.path.join(results_dir, "feature_importances.csv"))
    save_feature_importance_plot(feature_importance_df, results_dir)


if __name__ == "__main__":
    main()
