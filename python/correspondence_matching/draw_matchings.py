#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os
import random

import cv2
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deciding_only", action="store_true")
    parser.add_argument("--bandwidth", type=int, default=66)

    parser.add_argument("--weight_clusters", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = os.path.join(
        "correspondence_matching", "data", f"BW{args.bandwidth}_DecidingOnly{args.deciding_only}"
    )
    weighted_str = "weighted" if args.weight_clusters else "nonweighted"

    for outcome in ["FP", "FN", "TP", "TN", "Inc"]:
        os.makedirs(os.path.join(data_dir, weighted_str, outcome))

    with open(os.path.join(data_dir, weighted_str, "matched_cluster_data.json"), "r") as f:
        matched_cluster_data = json.load(f)

    trial_stats = pd.read_csv(
        os.path.join("data", "CwCeTrialStats_20200324.csv"),
        usecols=["ImagePair", "Examiner", "Outcome"],
    )

    for image_pair in tqdm(matched_cluster_data):
        pair_data = matched_cluster_data[image_pair]
        for examiner in pair_data:
            trial_data = pair_data[examiner]

            # Pull outcome from trial_stats
            outcome = trial_stats[
                (trial_stats["ImagePair"] == image_pair) & (trial_stats["Examiner"] == examiner)
            ]["Outcome"]
            assert outcome.shape[0] == 1
            outcome = outcome.iloc[0]

            left_img = cv2.imread(os.path.join("WBeyeDataset105", f"{image_pair}_Left.png"))
            right_img = cv2.imread(os.path.join("WBeyeDataset105", f"{image_pair}_Right.png"))
            images = {"Left": left_img, "Right": right_img}

            assert len(trial_data["Left"]) == len(trial_data["Right"])
            num_points = len(trial_data["Left"])
            colors = dict()
            for i in range(num_points):
                colors[i] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            for side in images:
                for i in range(num_points):
                    coord = trial_data[side][i]
                    x = int(coord[0])
                    y = int(coord[1])
                    cv2.circle(
                        img=images[side],
                        center=(x, y),
                        radius=10,
                        color=colors[i],
                        thickness=-1,
                    )

                cv2.imwrite(
                    os.path.join(
                        data_dir, weighted_str, outcome, f"{image_pair}_{examiner}_{side}.jpg"
                    ),
                    images[side],
                )


if __name__ == "__main__":
    main()
