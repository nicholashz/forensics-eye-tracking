#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import os
import pickle

import coloredlogs
import matplotlib.pyplot as plt
import pandas as pd
import shap

from calculate_metrics import calculate_trial_metrics
from train_model import Config, prepare_data, run_trial

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
    config_path = os.path.join(results_dir, "config.pickle")

    with open(config_path, "rb") as f:
        config = pickle.load(f)

    logging.info("Loaded config from file.")

    df = prepare_data(config)
    test_df = df.groupby("ImagePair").apply(lambda x: x.sample(frac=0.2))
    test_idx = [x[1] for x in test_df.index]
    train_df = df[~df.index.isin(test_idx)]

    # SHAP
    trial_output, feature_weights, model = run_trial(
        train_df=train_df, test_df=test_df, config=config
    )
    trial_metrics, fpr, tpr = calculate_trial_metrics(trial_output, config)

    explainer = shap.TreeExplainer(model)
    features = train_df.drop(columns=["ImagePair", "Examiner", "Outcome"])
    shap_values = explainer.shap_values(features)
    shap_df = pd.DataFrame(shap_values, columns=features.columns, index=features.index)
    shap_df = shap_df.merge(train_df[["ImagePair", "Examiner"]], left_index=True, right_index=True)

    # TODO groupby image pairs and analyze shap values to check for image effects

    plt.clf()
    shap.summary_plot(shap_values, features, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "shap_summary.jpg"))

    plt.clf()
    shap.dependence_plot("rank(1)", shap_values, features, interaction_index=None)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "dependence.jpg"))


if __name__ == "__main__":
    main()
