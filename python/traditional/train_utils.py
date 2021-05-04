import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, train_test_split


def split_generator(df, config):
    if config.group_pairs:
        # Remove image pairs that were used less than 5 times
        num_obs_per_pair = df.groupby("ImagePair").size()
        df = df[df["ImagePair"].isin(num_obs_per_pair[num_obs_per_pair > 9].index)]

        # Shuffle df
        df = df.sample(frac=1).reset_index(drop=True)

        # Split with equal sampling from image pairs
        num_splits = 5
        num_obs_per_split = round(num_obs_per_pair / num_splits).astype(np.int)
        for split in range(num_splits):
            train, test = None, None
            for image_pair in num_obs_per_pair[num_obs_per_pair >= 5].index:
                image_trials = df[df["ImagePair"] == image_pair]
                n = num_obs_per_split[image_pair]

                end_idx = (split + 1) * n if split != num_splits - 1 else len(image_trials)
                test_trials = image_trials.iloc[(split * n) : end_idx].copy()

                if train is None or test is None:
                    test = test_trials
                    train = image_trials[
                        ~image_trials["Examiner"].isin(test_trials["Examiner"])
                    ].copy()
                else:
                    test = test.append(test_trials, ignore_index=True)
                    train = train.append(
                        image_trials[
                            ~image_trials["Examiner"].isin(test_trials["Examiner"])
                        ].copy(),
                        ignore_index=True,
                    )
            yield train, test
    else:
        num_splits = 25
        for i in range(num_splits):
            yield train_test_split(df, test_size=0.2)


def prepare_data(config):
    df = pd.read_csv(config.data_csv)
    # corresp_scores = pd.read_csv(
    #     os.path.join(
    #         "correspondence_matching",
    #         "data",
    #         "BW44_DecidingOnlyFalse",
    #         "nonweighted",
    #         "scores.csv",
    #     ),
    #     usecols=["ImagePair", "Examiner", "CorrespScore", "NumClusters"],
    # )
    # df = df.merge(corresp_scores, on=["ImagePair", "Examiner"])

    # TODO TEMPORARY REMOVING TOM'S CORRESPONDENCE COLUMNS
    toms_cols = []
    for col in df.columns:
        if col.endswith("BW22"):
            toms_cols.append(col)
    df = df.drop(columns=toms_cols)

    # TODO TEMPORARILY DROPPING CLARITY
    df = df.drop(columns=["PctClarBlue", "PctClarGreen", "PctClarRedYellow"])

    if config.use_clustering:
        cluster_df = pd.read_csv(os.path.join("data", "clusters.csv"))
        df = df.merge(cluster_df, on="Examiner")

    # Drop data for the experiment being run
    for col in config.drop_info:
        if config.drop_info[col] == "all":
            df = df.drop(columns=col)
        else:
            df = df[~df[col].isin(config.drop_info[col])]

    # df = pd.get_dummies(df, columns=["Conclusion-Simple"], drop_first=True)

    # Drop columns with only 1 unique value
    nunique = df.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    df = df.drop(cols_to_drop, axis=1)

    return df


def run_trial(train_df, test_df, config):
    """
    """

    # Fill with median
    # links_str = "TempSeqFits_DetailBF88_IncRLTrans_UseLstFrstClust_BW22"
    median_fill_features = [
        # "CorrespScore",
        "EMDLeftCToRightCSelf",
        "EMDLeftCToRightCSelfDeciding",
        "Difficulty",
        # "ActivConc" + links_str,
        # "Angle" + links_str,
        # "Scale" + links_str,
        # "RelativeAngle" + links_str,
        # "AveMinDist" + links_str,
    ]
    if config.use_group_stats:
        median_fill_features.append("EMDDistanceToCorrect_C_Left")
        median_fill_features.append("EMDDistanceToCorrect_C_Right")
        median_fill_features.append("EMDDistanceJustDecidingToCorrect_C_Left")
        median_fill_features.append("EMDDistanceJustDecidingToCorrect_C_Right")

    # Split features from targets
    train_features = train_df.drop(columns=["ImagePair", "Examiner", config.target])
    test_features = test_df.drop(columns=["ImagePair", "Examiner", config.target])

    train_targets = train_df[config.target].map(config.target_mapping)

    if config.group_pairs:
        # Fill and normalize features by image pair
        feature_medians = train_df.groupby("ImagePair").median()
        feature_means = train_df.groupby("ImagePair").mean()
        feature_stds = train_df.groupby("ImagePair").std()
        feature_stds[feature_stds == 0] = 1
        for image_pair in train_df["ImagePair"].unique():
            train_features[train_df["ImagePair"] == image_pair] = train_features[
                train_df["ImagePair"] == image_pair
            ].fillna(feature_medians.loc[image_pair])
            test_features[test_df["ImagePair"] == image_pair] = test_features[
                test_df["ImagePair"] == image_pair
            ].fillna(feature_medians.loc[image_pair])

            train_features[train_df["ImagePair"] == image_pair] -= feature_means.loc[image_pair]
            train_features[train_df["ImagePair"] == image_pair] /= feature_stds.loc[image_pair]

            test_features[test_df["ImagePair"] == image_pair] -= feature_means.loc[image_pair]
            test_features[test_df["ImagePair"] == image_pair] /= feature_stds.loc[image_pair]
    else:
        fill_dict = dict()
        for feature in median_fill_features:
            fill_dict[feature] = train_features[feature].median()
        train_features = train_features.fillna(fill_dict)
        test_features = test_features.fillna(fill_dict)

    # Check for missing values
    for df in [train_features, test_features]:
        if df.isna().any().any():
            col_has_nan = df.isna().any()
            nan_col_names = list(col_has_nan.index[col_has_nan])
            raise ValueError(f"Need to fix missing values in columns: {nan_col_names}")

    # Drop highly correlated features
    corr_matrix = train_features.corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [
        column for column in upper.columns if any(np.abs(upper[column]) > config.corr_threshold)
    ]
    train_features = train_features.drop(columns=to_drop)
    test_features = test_features.drop(columns=to_drop)

    # Set up classifiers to train
    clf = GradientBoostingClassifier()

    # Define parameters to be considered for each classifier
    param_grid = dict(
        # learning_rate=[0.75, 1, 1.25],
        n_estimators=[250, 500, 750],
        max_depth=[5, 10, 20, 30],
        min_samples_split=[5],
        min_samples_leaf=[5, 10, 20, 30],
        min_weight_fraction_leaf=[0.0],
        max_features=[None, "auto"],
        max_leaf_nodes=[None],
        min_impurity_decrease=[0.0],
    )

    # Perform grid search over parameters to select best model
    grid_search = GridSearchCV(
        clf, param_grid=param_grid, scoring=config.scorer, n_jobs=-1, cv=config.cv, verbose=1
    )
    grid_search.fit(train_features, train_targets)

    # Select best model
    best_model = grid_search.best_estimator_

    # Select features that positively contributed to model performance
    selector = SelectFromModel(best_model, threshold=0.01, prefit=True)
    feature_idx = selector.get_support()
    train_features = train_features.iloc[:, feature_idx]
    test_features = test_features.iloc[:, feature_idx]

    # Retrain on the positively-contributing features
    best_model.fit(train_features, train_targets)

    # Save feature weights
    feature_weights = pd.DataFrame(
        {"Feature": train_features.columns, "Weight": best_model.feature_importances_}
    )
    feature_weights = feature_weights.sort_values(by="Weight", ascending=False)

    # Make predictions
    test_info = pd.DataFrame(
        {
            "ImagePair": test_df["ImagePair"],
            "Examiner": test_df["Examiner"],
            config.target: test_df[config.target],
        }
    )
    probas = best_model.predict_proba(test_features)
    for class_num in range(probas.shape[1]):
        test_info[f"Pred{class_num}"] = probas[:, class_num]
    test_info["ModelPrediction"] = np.argmax(probas, axis=1)
    test_info["ModelPrediction"] = test_info["ModelPrediction"].map(config.decode_target)

    return test_info, feature_weights, best_model
