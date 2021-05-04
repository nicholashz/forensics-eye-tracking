import logging
import os

import coloredlogs
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO, fmt="[%(asctime)s] %(name)s [%(levelname)s] %(message)s")


def save_roc_curve(mean_fpr, tprs, roc_aucs, class_name, output_dir):
    """
    :param mean_fpr: ndarray, shape (num_total_trials,)
    :param tprs: list, with len == num_total_trials, where elements are TPR arrays of equal length
    :param roc_aucs: ndarray, shape (num_total_trials,)
    :param class_name: str, class name for the plot title and filename
    :param output_dir: str, output directory in which to save curve image

    Plots ROC curve for the class across multiple trials. Saves to output_dir.
    """

    plt.clf()
    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Luck", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(roc_aucs)
    plt.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=f"Mean ROC (AUC = {mean_auc:0.2f} ± {std_auc:0.2f})",
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(
        mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2, label="±1 std. dev."
    )

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{class_name} ROC")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, f"roc_{class_name}.jpg"))
    logger.info(f"ROC curve for class {class_name} saved in directory '{output_dir}'")


def save_confusion_matrix(conf_matrices, class_names, output_dir):
    """
    :param conf_matrices: list of non-normalized confusion matrices
    :param output_dur: str, name of directory in which to save plots
    """

    num_classes = len(class_names)

    plt.clf()
    fig, ax = plt.subplots(1, 2, figsize=(num_classes * 5, num_classes * 2.5))

    # Normalize along each axis and produce two confusion matrices
    for dim in [0, 1]:
        totals = np.repeat(
            np.expand_dims(np.sum(conf_matrices, axis=dim + 1), axis=dim + 1),
            repeats=num_classes,
            axis=dim + 1,
        )
        conf_normalized = conf_matrices / totals
        conf_mean = np.mean(conf_normalized, axis=0)
        conf_std = np.std(conf_normalized, axis=0)

        # Create annotations to display mean and std dev
        annot_array = []
        for j in range(conf_mean.shape[0]):
            row_strs = []
            for i in range(conf_mean.shape[1]):
                disp_str = f"{conf_mean[j, i]:.2f} ± {conf_std[j, i]:.2f}"
                row_strs.append(disp_str)
            annot_array.append(row_strs)

        # Produce heatmap
        sns.heatmap(
            np.mean(conf_normalized, axis=0),
            vmin=0,
            vmax=1,
            cmap=sns.color_palette("Blues", n_colors=50),
            annot=annot_array,
            fmt="",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax[dim],
        )

        ax[dim].set_xlabel("Predicted")
        ax[dim].set_ylabel("Actual")

    ax[0].set_title("Ratio of Predicted to Predicted Totals")
    ax[1].set_title("Ratio of Predicted to Actual Totals")

    plt.savefig(os.path.join(output_dir, "confusion.jpg"))
    logger.info(f"Confusion matrix plot saved in directory '{output_dir}'")


def save_feature_importance_plot(feature_importances, output_dir):
    """
    """

    plt.clf()
    plt.figure(figsize=(15, 5))

    feature_importances = feature_importances.set_index("Feature")
    transpose_df = feature_importances.T
    transpose_df = transpose_df.reindex(
        transpose_df.mean().sort_values(ascending=False).index, axis=1
    )
    bplot = sns.barplot(data=transpose_df, orient="h")

    null_counts = transpose_df.isnull().sum()
    num_trials = transpose_df.shape[0]
    for feature_idx, p in enumerate(bplot.patches):
        num_values = num_trials - null_counts.iloc[feature_idx]
        bplot.annotate(
            f"{num_values} / {num_trials}",
            (0.01, p.get_y() + 2 * p.get_height() / 3),
            fontsize=11,
            color="black",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_plot.jpg"))
    logger.info(f"Feature importance plot saved in directory '{output_dir}'")


def calculate_trial_metrics(df, config):
    """
    """

    labels = df[config.target].map(config.target_mapping)
    preds = df["ModelPrediction"].map(config.target_mapping)

    # Basic metrics
    accuracy = metrics.accuracy_score(labels, preds, normalize=True)
    precision = metrics.precision_score(labels, preds, average=None)
    recall = metrics.recall_score(labels, preds, average=None)
    conf_matrix = metrics.confusion_matrix(labels, preds)

    # Calculate False Pos Rate, True Pos Rate, and AUC under ROC curve
    fpr = dict()
    tpr = dict()
    binarized_labels = np.zeros((labels.size, labels.max() + 1))
    binarized_labels[np.arange(labels.size), labels] = 1

    roc_auc = np.empty(shape=binarized_labels.shape[1])
    for i in range(binarized_labels.shape[1]):
        probas = df[f"Pred{i}"]
        fpr[i], tpr[i], _ = metrics.roc_curve(binarized_labels[:, i], probas)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    trial_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "conf_matrix": conf_matrix,
    }

    return trial_metrics, fpr, tpr
