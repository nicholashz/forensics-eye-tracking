#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import logging
import os
import pickle

import coloredlogs

from config import Config
from train_utils import prepare_data, run_trial, split_generator

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

    logger.info(f"Running trials for params: {args.experiment, args.trials, args.use_group_stats}")

    output_dir = os.path.join("results", f"{args.experiment}_{args.trials}_{args.use_group_stats}")
    os.makedirs(output_dir, exist_ok=True)

    # Set up config
    config = Config(
        experiment=args.experiment,
        trials=args.trials,
        group_pairs=False,
        use_group_stats=args.use_group_stats,
        use_clustering=False,
        separate=None,
    )

    # Set up trials to be run
    output_data = dict()
    feature_importance_df = None

    df = prepare_data(config)
    splits = split_generator(df, config)
    split_num = 0
    for train_df, test_df in splits:
        logger.info(f"Starting trial on split: {split_num + 1}")

        test_info, feature_weights, model = run_trial(
            train_df=train_df, test_df=test_df, config=config
        )

        # Fill trial data with output
        feature_weights = feature_weights.rename(
            columns={"Weight": f"Weight{split_num}", "Var": f"Var{split_num}"}
        )
        output_data[split_num] = test_info.to_dict()

        # Add feature importances from trial to dataframe
        if feature_importance_df is None:
            feature_importance_df = feature_weights
        else:
            feature_importance_df = feature_importance_df.merge(
                feature_weights, on="Feature", how="outer"
            )

        logger.info(f"Finished trial on split: {split_num + 1}")

        split_num += 1

    # Write data and config to files
    config_path = os.path.join(output_dir, "config.pickle")
    with open(config_path, "wb") as f:
        pickle.dump(config, f)
    logger.info(f"Config written to {config_path}")

    data_path = os.path.join(output_dir, "data.json")
    with open(data_path, "w") as f:
        json.dump(output_data, f)
    logger.info(f"Data written to {data_path}")

    feature_importance_path = os.path.join(output_dir, "feature_importances.csv")
    feature_importance_df.to_csv(feature_importance_path, index=False)
    logger.info(f"Feature importance data written to {feature_importance_path}")


if __name__ == "__main__":
    main()
