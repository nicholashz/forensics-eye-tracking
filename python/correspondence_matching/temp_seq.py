#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os

import numpy as np
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

    # Load fixation data
    fixations = pd.read_csv(os.path.join(data_dir, "clustered_fixations.csv"))

    transition_data = dict()
    unique_trials = fixations[["ImagePair", "Examiner"]].drop_duplicates()
    trials_missing_clusters = []
    for i, row in tqdm(unique_trials.iterrows(), total=unique_trials.shape[0]):
        image_pair = row["ImagePair"]
        examiner = row["Examiner"]

        # Select fixations for trial
        matches_pair = fixations["ImagePair"] == image_pair
        matches_examiner = fixations["Examiner"] == examiner
        trial_fixations = fixations[matches_pair & matches_examiner]

        # Set up transition matrix
        left_image_idxs = trial_fixations["Image"] == "Left"
        right_image_idxs = trial_fixations["Image"] == "Right"
        num_left_clusters = trial_fixations[left_image_idxs]["Cluster"].nunique()
        num_right_clusters = trial_fixations[right_image_idxs]["Cluster"].nunique()
        if num_left_clusters == 0 or num_right_clusters == 0:
            trials_missing_clusters.append((image_pair, examiner))
            continue
        transition_matrix = np.zeros(shape=(num_left_clusters, num_right_clusters))

        # Count the clusters visited on each side per sequence
        # One sequence contains all the fixations on one side with all the fixations on the other after switching
        found_first_left_fixation = False
        image_switched = False
        left_clusters_visited = np.zeros(shape=num_left_clusters, dtype=np.int)
        right_clusters_visited = np.zeros(shape=num_right_clusters, dtype=np.int)
        for j, fix_row in trial_fixations.iterrows():
            image = fix_row["Image"]

            # Start with first left fixation
            if not found_first_left_fixation:
                if image == "Left":
                    found_first_left_fixation = True
                else:
                    continue

            # Compare 'current' image to one from previous fixation
            if image != "Left":
                # If they've already switched, end the sequence and count transitions
                if image_switched:
                    image_switched = False
                    expanded_left_visited = left_clusters_visited[:, None].repeat(
                        num_right_clusters, axis=1
                    )
                    expanded_right_visited = right_clusters_visited[None, :].repeat(
                        num_left_clusters, axis=0
                    )
                    transitions = np.where(
                        np.logical_and(expanded_left_visited > 0, expanded_right_visited > 0),
                        np.sum([expanded_left_visited, expanded_right_visited], axis=0),
                        np.zeros_like(transition_matrix),
                    )
                    transition_matrix += transitions
                    left_clusters_visited = np.zeros(shape=num_left_clusters)
                    right_clusters_visited = np.zeros(shape=num_right_clusters)
                else:
                    image_switched = True

            # Increase indicator for cluster visited
            cluster_idx = int(fix_row["Cluster"])
            if args.weight_clusters:
                if image == "Left":
                    left_clusters_visited[cluster_idx] += 0.5
                elif image == "Right":
                    right_clusters_visited[cluster_idx] += 0.5
            else:
                if image == "Left":
                    left_clusters_visited[cluster_idx] = 0.5
                elif image == "Right":
                    right_clusters_visited[cluster_idx] = 0.5

        # Save results in dict
        if image_pair not in transition_data:
            transition_data[image_pair] = dict()
        transition_data[image_pair][examiner] = transition_matrix.tolist()

    # Write all results to JSON file
    weighted_str = "weighted" if args.weight_clusters else "nonweighted"
    os.makedirs(os.path.join(data_dir, weighted_str), exist_ok=True)
    json_path = os.path.join(data_dir, weighted_str, "transitions.json")
    with open(json_path, "w") as f:
        json.dump(transition_data, f)

    print("Done! Transition data written to json file.")
    print(f"{len(trials_missing_clusters)} trials have no clusters in one or both images.")


if __name__ == "__main__":
    main()
