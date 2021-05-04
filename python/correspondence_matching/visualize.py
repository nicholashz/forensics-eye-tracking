#!/usr/bin/env python
# coding: utf-8

import os

import cv2
import numpy as np
import pandas as pd


def main():
    fixations = pd.read_csv(os.path.join("correspondence_matching", "clustered_fixations.csv"))
    cluster_centers = pd.read_csv(os.path.join("correspondence_matching", "cluster_centers.csv"))

    image_pair = "CW022"
    examiner = "Y614"

    left_img = cv2.imread(os.path.join("WBeyeDataset105", f"{image_pair}_Left.png"))
    right_img = cv2.imread(os.path.join("WBeyeDataset105", f"{image_pair}_Right.png"))
    images = {"Left": left_img, "Right": right_img}

    trial_centers = cluster_centers[
        (cluster_centers["ImagePair"] == image_pair) & (cluster_centers["Examiner"] == examiner)
    ]
    trial_fixations = fixations[
        (fixations["ImagePair"] == image_pair) & (fixations["Examiner"] == examiner)
    ]

    for image in ["Left", "Right"]:
        centers = trial_centers[trial_centers["Image"] == image]
        image_fixations = trial_fixations[trial_fixations["Image"] == image]

        for i, row in centers.iterrows():
            cv2.circle(
                img=images[image],
                center=(int(row["CenterX"]), int(row["CenterY"])),
                radius=10,
                color=(0, 0, 255),
                thickness=-1,
            )

        for i, row in image_fixations.iterrows():
            cv2.circle(
                img=images[image],
                center=(int(row["FixX"]), int(row["FixY"])),
                radius=2,
                color=(255, 0, 0),
                thickness=-1,
            )

        cv2.imwrite(f"{image}.jpg", images[image])


if __name__ == "__main__":
    main()
