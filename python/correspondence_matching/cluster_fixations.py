#!/usr/bin/env python
# coding: utf-8

import argparse
import os

import numpy as np
import pandas as pd

from sklearn import cluster
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deciding_only", action="store_true")
    parser.add_argument("--bandwidth", type=int, default=66)

    return parser.parse_args()


def main():
    args = parse_args()

    use_cols = ["ImagePair", "Examiner", "Outcome", "Phase", "Subphase", "Image", "FixX", "FixY"]
    df = pd.read_csv(os.path.join("data", "CwCe_OK_Fixations_20180703.csv"), usecols=use_cols)

    df = df[df["Outcome"] != "NV"]
    df = df[df["Phase"] == "C"]
    if args.deciding_only:
        df = df[df["Subphase"] == "Deciding"]
    df = df.drop(columns=["Outcome", "Phase", "Subphase"])

    df["Cluster"] = np.nan
    grouped = df.groupby(["ImagePair", "Examiner", "Image"])

    cluster_center_df = pd.DataFrame(
        columns=["ImagePair", "Examiner", "Image", "Cluster", "CenterX", "CenterY"]
    )
    for name, group in tqdm(grouped):
        image_pair, examiner, image = name
        mean_shift = cluster.MeanShift(bandwidth=args.bandwidth, n_jobs=-1)
        pts = group[["FixX", "FixY"]].values

        clusters = mean_shift.fit_predict(pts)
        cluster_df = pd.DataFrame(
            data={
                "ImagePair": image_pair,
                "Examiner": examiner,
                "Image": image,
                "FixX": group["FixX"],
                "FixY": group["FixY"],
                "Cluster": clusters,
            }
        )

        # Record cluster centers
        cluster_centers = pd.DataFrame(
            data={
                "ImagePair": image_pair,
                "Examiner": examiner,
                "Image": image,
                "Cluster": [i for i in range(mean_shift.cluster_centers_.shape[0])],
                "CenterX": mean_shift.cluster_centers_[:, 0],
                "CenterY": mean_shift.cluster_centers_[:, 1],
            }
        )
        cluster_center_df = cluster_center_df.append(cluster_centers, ignore_index=True)

        # Update the df with the newly-found clusters
        df.update(cluster_df)

    # Save the dataframes
    data_dir = os.path.join(
        "correspondence_matching", "data", f"BW{args.bandwidth}_DecidingOnly{args.deciding_only}"
    )
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(os.path.join(data_dir, "clustered_fixations.csv"), index=False)
    cluster_center_df.to_csv(
        os.path.join(data_dir, "cluster_centers.csv"), index=False
    )


if __name__ == "__main__":
    main()
