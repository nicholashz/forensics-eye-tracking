#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import json

import numpy as np
import pandas as pd
from scipy import optimize
from tqdm import tqdm

from matching_utils import assign_cluster_matches, transform_distance


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deciding_only", action="store_true")
    parser.add_argument("--bandwidth", type=int, default=66)

    parser.add_argument("--weight_clusters", action="store_true")

    parser.add_argument("-a", "--algo", default="greedy")
    parser.add_argument("-t", "--threshold", type=float, default=0.3)

    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = os.path.join(
        "correspondence_matching", "data", f"BW{args.bandwidth}_DecidingOnly{args.deciding_only}"
    )
    weighted_str = "weighted" if args.weight_clusters else "nonweighted"
    json_path = os.path.join(data_dir, weighted_str, "transitions.json")

    with open(json_path, "r") as f:
        transitions = json.load(f)

    trial_stats = pd.read_csv(
        os.path.join("data", "CwCeTrialStats_20200324.csv"),
        usecols=["ImagePair", "Examiner", "Outcome"],
    )

    cluster_centers = pd.read_csv(os.path.join(data_dir, "cluster_centers.csv"))
    unique_trials = cluster_centers[["ImagePair", "Examiner"]].drop_duplicates()
    unique_trials = unique_trials[unique_trials["ImagePair"].str.contains("CW")]
    unique_trials = unique_trials.values.tolist()

    scores_df = pd.DataFrame(
        columns=["ImagePair", "Examiner", "Outcome", "CorrespScore", "NumClusters"]
    )
    matched_cluster_data = dict()
    for image_pair, examiner in tqdm(unique_trials):

        # Pull outcome from trial_stats
        outcome = trial_stats[
            (trial_stats["ImagePair"] == image_pair) & (trial_stats["Examiner"] == examiner)
        ]["Outcome"]
        assert outcome.shape[0] == 1
        outcome = outcome.iloc[0]

        # Assign cluster matches for the trial
        try:
            matrix = transitions[image_pair][examiner]
        except KeyError:
            tqdm.write(f"{image_pair}, {examiner} not found in transition data. Skipping...")
            continue
        matrix = np.array(matrix)
        match_strengths, row_ind, col_ind = assign_cluster_matches(matrix, algo=args.algo)
        if row_ind.size == 0 or col_ind.size == 0:
            continue
        high_strength_match_idx = match_strengths[row_ind, col_ind] > args.threshold
        row_ind = row_ind[high_strength_match_idx]
        col_ind = col_ind[high_strength_match_idx]

        assert len(row_ind) == len(col_ind)
        num_clusters = len(row_ind)

        trial_centers = cluster_centers[
            (cluster_centers["ImagePair"] == image_pair) & (cluster_centers["Examiner"] == examiner)
        ]

        # Produce scores
        cluster_points = dict()
        for side in ["Left", "Right"]:
            if side == "Left":
                cluster_idx = row_ind
            else:
                cluster_idx = col_ind

            # Get cluster centers and reorder so that the left/right distances match up
            image_centers = trial_centers[
                (trial_centers["Image"] == side)
                & (trial_centers["Cluster"].isin(cluster_idx))
            ]
            image_centers = image_centers.set_index("Cluster")
            image_centers = image_centers.iloc[np.argsort(cluster_idx)]
            center_coords = image_centers[["CenterX", "CenterY"]].values
            cluster_points[side] = center_coords

        if num_clusters == 0:
            score = np.nan
        else:
            cluster_center_diffs = cluster_points["Right"] - cluster_points["Left"]
            initial_guess = np.array(
                [cluster_center_diffs[0, 0], cluster_center_diffs[0, 1], np.pi / 16]
            )
            xopt, fopt, _ = optimize.fmin_l_bfgs_b(
                func=transform_distance,
                x0=initial_guess,
                args=(cluster_points["Left"], cluster_points["Right"],),
                approx_grad=True,
                bounds=[(None, None), (None, None), (-np.pi / 4, np.pi / 4)],
                disp=False,
            )
            score = fopt

        if image_pair not in matched_cluster_data:
            matched_cluster_data[image_pair] = dict()
        matched_cluster_data[image_pair][examiner] = dict()
        matched_cluster_data[image_pair][examiner]["Score"] = score
        for side in ["Left", "Right"]:
            matched_cluster_data[image_pair][examiner][side] = cluster_points[side].tolist()

        scores_df = scores_df.append(
            {
                "ImagePair": image_pair,
                "Examiner": examiner,
                "Outcome": outcome,
                "CorrespScore": score,
                "NumClusters": num_clusters,
            },
            ignore_index=True,
        )

    # Write to file
    scores_df.to_csv(os.path.join(data_dir, weighted_str, "scores.csv"), index=False)
    with open(os.path.join(data_dir, weighted_str, "matched_cluster_data.json"), "w+") as f:
        json.dump(matched_cluster_data, f)


if __name__ == "__main__":
    main()
