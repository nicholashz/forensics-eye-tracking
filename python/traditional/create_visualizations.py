#!/usr/bin/env python
# coding: utf-8

import os

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kstest, uniform, chisquare
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.graphics.mosaicplot import mosaic
from tqdm import tqdm

from config import PLOT_DETAILS, STATS_COLS, NON_NUMERIC_FEATURES

plt.style.use("seaborn")


def get_details(feature):
    try:
        details = PLOT_DETAILS[feature]
        name = details["name"] if "name" in details else feature
        units = details["units"] if "units" in details else None
    except KeyError:
        name = feature
        units = None

    return name, units


def ranked_df(df, by_image_pair=False):
    df_copy = df.copy()

    metadata = df_copy[["ImagePair", "Mating", "Conclusion-Simple", "Outcome", "Examiner"]]
    df_copy = df_copy.drop(columns=["Mating", "Conclusion-Simple", "Outcome", "Examiner"])
    if by_image_pair:
        ranked = df_copy.groupby("ImagePair").rank(pct=True)
    else:
        ranked = df_copy.rank(pct=True)

    df_copy = metadata.merge(ranked, left_index=True, right_index=True)

    return df_copy


def calculate_ks_test_results(df, features, outcomes):
    results = pd.DataFrame()
    for feat in features:
        data_row = {"Feature": feat}
        for outcome in outcomes:
            subset = df[df["Outcome"] == outcome]

            sample = subset[feat].values
            sample = sample[~np.isnan(sample)]

            # Catch error when sample is all NaNs (such as AveDeviation for nonmates)
            try:
                statistic, pvalue = kstest(sample, "uniform")
            except ValueError:
                statistic, pvalue = None, None
            data_row[f"{outcome}_statistic"] = statistic
            data_row[f"{outcome}_pvalue"] = pvalue
        results = results.append(data_row, ignore_index=True,)

    results = results.set_index("Feature")

    return results


def plot_outcome_proportion_by_difficulty(df, output_fp):
    df_copy = df.copy()
    plt.clf()

    # Count each outcome in each difficulty
    df_copy["Difficulty"] = df_copy["Difficulty"].map(
        {0: "VeryEasy", 0.25: "Easy", 0.5: "Moderate", 0.75: "Difficult", 1: "VeryDifficult"}
    )
    diff_counts = (
        df_copy.groupby(["Outcome", "Difficulty"])
        .size()
        .reset_index()
        .pivot(columns="Difficulty", index="Outcome", values=0)
    )

    # Create logical ordering of difficulty/outcomes
    difficulties = ["VeryEasy", "Easy", "Moderate", "Difficult", "VeryDifficult"]
    outcomes = ["TP", "FN", "IncMated", "IncNonMated", "FP", "TN"]
    diff_counts = diff_counts[diff_counts.columns.reindex(difficulties)[0]].fillna(0).T
    diff_counts = diff_counts[diff_counts.columns.reindex(outcomes)[0]]

    # Normalize so we get proportions
    norm_counts = diff_counts / diff_counts.values.sum(axis=1)[:, None]

    # Plot
    norm_counts.plot(kind="bar", stacked=True)

    # Label with count of each bar
    for idx in range(len(difficulties)):
        count_in_bar = int(diff_counts.iloc[idx].sum())
        plt.text(idx, 1.01, count_in_bar, ha="center")

    # Hatch to show mates vs nonmates
    plt.rcParams["hatch.linewidth"] = 0.3
    bars = plt.gca().patches
    for i in range(len(bars) // 2, len(bars)):
        bars[i].set_hatch("/")

    plt.title("Outcome Proportions by Difficulty")
    plt.ylabel("Proportion of Trials")

    # Draw base rate reference line
    mates = df_copy[df_copy["Outcome"].isin(outcomes[:3])]
    pct_mates = len(mates) / len(df_copy)
    plt.axhline(pct_mates, color=(0.5, 0.5, 0.5), label=f"Mates base rate ({pct_mates:.3f})")

    # Configure legend
    handles, labels = plt.gca().get_legend_handles_labels()
    legend_order = [0, 6, 5, 4, 3, 2, 1]
    plt.legend(
        [handles[i] for i in legend_order],
        [labels[i] for i in legend_order],
        bbox_to_anchor=(1.04, 1),
        loc="upper left",
    )

    plt.text(
        1.3, -0.25, f"n={diff_counts.sum().sum():.0f}", transform=plt.gca().transAxes,
    )

    plt.savefig(output_fp, bbox_inches="tight")
    plt.close()


def plot_feature_boxplot(output_fp, outcome_order, **kwargs):
    feature = kwargs["x"]
    data = kwargs["data"]
    name, units = get_details(feature)
    xlabel = f"{name} ({units})" if units is not None else name

    # remove outcomes that have all NaN trials for the feature (e.g. Nonmated for AveDeviation)
    not_nan_order = []
    for outcome in outcome_order:
        if ~data[data["Outcome"] == outcome][feature].isna().all():
            not_nan_order.append(outcome)

    plt.clf()
    fig, ax = plt.subplots(figsize=(14, 7))

    sns.boxplot(
        ax=ax,
        **kwargs,
        order=not_nan_order,
        orient="h",
        boxprops=dict(alpha=0.3),
    )

    sns.swarmplot(ax=ax, **kwargs, linewidth=0.5, size=4, order=not_nan_order, orient="h")
    plt.xlabel(xlabel)

    plt.text(
        1,
        -0.1,
        f"n={len(data[~np.isnan(data[feature])])}",
        transform=plt.gca().transAxes,
        ha="center",
    )

    plt.savefig(output_fp, bbox_inches="tight")
    plt.close()


def plot_feature_decileplot(df, feature, feature_ks, bins, outcomes, title, output_fp):
    df_copy = df.copy()

    plt.clf()
    fig, ax = plt.subplots(figsize=(7, 7))
    df_copy["percentile"] = pd.cut(df_copy[feature], bins=bins)
    if df_copy["percentile"].isna().any():
        df_copy["percentile"] = df_copy["percentile"].cat.add_categories("nan").fillna("nan")

    grouping_counts = (
        df_copy.groupby(["Outcome", "percentile"])
        .size()
        .reset_index()
        .pivot(columns="Outcome", index="percentile", values=0)
    )
    grouping_counts = grouping_counts[grouping_counts.columns.reindex(outcomes)[0]]
    grouping_counts = grouping_counts[outcomes]

    norm_counts = grouping_counts / grouping_counts.values.sum(axis=1)[:, None]
    colors = [(0.8, 0.4, 0), (0.5, 0.5, 0.5), (0, 0, 0.8)]
    norm_counts.plot(kind="bar", stacked=True, color=colors, ax=ax)

    # Label with count of each bar
    num_bars = len(bins) if "nan" in grouping_counts.index else len(bins) - 1
    for idx in range(num_bars):
        count_in_bar = int(grouping_counts.iloc[idx].sum())
        plt.text(idx, 1.01, count_in_bar, ha="center")

    # Draw base rate reference line
    samples_1 = df_copy[df_copy["Outcome"] == outcomes[0]][feature]
    samples_2 = df_copy[df_copy["Outcome"] == outcomes[1]][feature]
    top_line = (len(samples_1) + len(samples_2)) / len(df_copy)
    plt.axhline(
        top_line, color=(0.3, 0, 0.6), label=f"{outcomes[2]} base rate ({(1 - top_line):.3f})",
    )
    bot_line = len(samples_1) / len(df_copy)
    plt.axhline(
        bot_line, color=(0.6, 0, 0), label=f"{outcomes[0]} base rate ({bot_line:.3f})",
    )

    plt.title(title)
    plt.ylabel("Proportion of Trials")
    plt.xlabel("Percentile (Relative to Trials on Same Image Pair)")

    # reorder the legend for consistency
    legend_order = [0, 1, 4, 3, 2]
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [handles[i] for i in legend_order]
    labels = [labels[i] for i in legend_order]

    if feature_ks is not None:
        for i in range(len(labels)):
            if labels[i] in outcomes and "Inc" not in labels[i] and labels[i] != "FP":
                ks_stat = feature_ks[f"{labels[i]}_statistic"]
                pvalue = feature_ks[f"{labels[i]}_pvalue"]
                labels[i] = f"{labels[i]} [KS={ks_stat:.3f}, p={pvalue:.3f}]"

    plt.legend(
        handles, labels, bbox_to_anchor=(1.04, 1), loc="upper left",
    )

    plt.text(
        1.4,
        -0.2,
        f"n={grouping_counts.sum().sum():.0f}",
        transform=plt.gca().transAxes,
        ha="center",
    )

    # Save to file
    plt.savefig(output_fp, bbox_inches="tight")
    plt.close()


def two_feature_heatmap(df, features, output_fp):
    assert len(features) == 2
    y_name, _ = get_details(features[0])
    x_name, _ = get_details(features[1])

    plt.clf()
    fig, ax = plt.subplots(figsize=(7, 7))

    outcome = df["Outcome"]
    num_bins = 3
    feat1 = pd.cut(
        df[features[0]], [df[features[0]].max() * i / num_bins for i in range(num_bins + 1)]
    )
    feat2 = pd.cut(
        df[features[1]], [df[features[1]].max() * i / num_bins for i in range(num_bins + 1)]
    )

    combined = pd.concat([outcome, feat1, feat2], axis=1)
    x = combined.groupby([*features, "Outcome"]).size().unstack()
    # x = x["TP"] / (x["FN"] + x["TP"] + x["IncMated"])
    bin_counts = x.sum(axis=1)
    base_proportions = bin_counts / bin_counts.sum()
    base_proportions = base_proportions.unstack()
    base_proportions = base_proportions.reindex(base_proportions.index[::-1])

    missed_ids = x["FN"] + x["IncMated"]
    missed_proportions = missed_ids / missed_ids.sum()
    missed_proportions = missed_proportions.unstack()
    missed_proportions = missed_proportions.reindex(missed_proportions.index[::-1])

    relative_proportions = missed_proportions / base_proportions

    miss_list = missed_proportions.values.tolist()
    base_list = base_proportions.values.tolist()
    annot = []
    for i in range(len(miss_list)):
        row_annots = []
        for j in range(len(miss_list[i])):
            row_annots.append(f"{miss_list[i][j]:.2f} ({base_list[i][j]:.2f})")
        annot.append(row_annots)

    tp_trials = df[df["Outcome"] == "TP"]
    fn_trials = df[df["Outcome"] == "FN"]
    inc_trials = df[df["Outcome"] == "IncMated"]
    sns.heatmap(
        relative_proportions,
        cmap=sns.color_palette("coolwarm", 21),
        vmin=0.5,
        vmax=1.5,
        center=1,
        annot=annot,
        fmt="",
        ax=ax,
        cbar=False,
    )
    ax.scatter(
        (tp_trials[features[1]] / df[features[1]].max()) * num_bins,
        (1 - (tp_trials[features[0]] / df[features[0]].max())) * num_bins,
        marker=".",
        color="blue",
        label=f"TP (n={tp_trials.shape[0]})",
    )
    ax.scatter(
        (fn_trials[features[1]] / df[features[1]].max()) * num_bins,
        (1 - (fn_trials[features[0]] / df[features[0]].max())) * num_bins,
        marker=".",
        color="darkorange",
        label=f"FN (n={fn_trials.shape[0]})",
    )
    ax.scatter(
        (inc_trials[features[1]] / df[features[1]].max()) * num_bins,
        (1 - (inc_trials[features[0]] / df[features[0]].max())) * num_bins,
        marker=".",
        color="grey",
        label=f"Inc (n={inc_trials.shape[0]})",
    )

    plt.title("Proportion of Missed IDs in Region\n(Expected Proportion in Parentheses)")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.xlabel(x_name)
    plt.ylabel(y_name)

    plt.savefig(output_fp, bbox_inches="tight")
    plt.close()

    return df


def plot_feature_dist(data, col, outcomes, xlabel, output_fp, bins=None, feature_ks=None):
    """
    Plots the mosaic plot for col by outcome.
    """
    df_copy = data.copy()

    plt.clf()
    fig, ax = plt.subplots(figsize=(7, 7))

    colors = {
        "FN": "darkorange",
        "IncMated": "darkgrey",
        "TP": "blue",
        "FP": "orange",
        "IncNonMated": "lightgrey",
        "TN": "lightblue",
    }

    # Get counts of col by outcome
    if col == "TempSeqNumLinks":  # special case
        grouping_counts = df_copy.groupby([col, "Outcome"]).size().reindex(outcomes, level=1)

        # calculate chi^2
        f_obs = grouping_counts.unstack(0).values
        f_exp = (f_obs.sum(axis=1) / f_obs.sum(None)).reshape(-1, 1).repeat(13, axis=1) * f_obs.sum(
            axis=0
        )
        chisq, p = chisquare(f_obs.T, f_exp.T)
        chisq, p = chisq[::-1], p[::-1]  # change ordering so it aligns with labels later
    else:
        df_copy["percentile"] = pd.cut(df_copy[col], bins=bins)
        if df_copy["percentile"].isna().any():
            df_copy["percentile"] = df_copy["percentile"].cat.add_categories("NaN").fillna("NaN")
        grouping_counts = (
            df_copy.groupby(["percentile", "Outcome"]).size().reindex(outcomes, level=1)
        )

    # Remove counts that are zeros for all outcomes
    for idx in set([i[0] for i in grouping_counts.index]):
        if grouping_counts.loc[idx].sum() == 0:
            grouping_counts = grouping_counts.drop(idx)

    # plot mosaic
    def props(key):
        return {"color": colors[key[1]]}

    def labelizer(key):
        return ""

    mosaic(data=grouping_counts.to_dict(), properties=props, labelizer=labelizer, ax=ax)
    handles = [Patch(facecolor=colors[outcome], label=outcome) for outcome in outcomes][::-1]
    labels = outcomes[::-1].copy()

    # Draw base rate reference line
    samples_1 = df_copy[df_copy["Outcome"] == outcomes[0]][col]
    samples_2 = df_copy[df_copy["Outcome"] == outcomes[1]][col]
    top_line = (len(samples_1) + len(samples_2)) / len(df_copy)
    bot_line = len(samples_1) / len(df_copy)
    plt.axhline(top_line, color=(0.3, 0, 0.6))
    plt.axhline(bot_line, color=(0.6, 0, 0))
    handles.append(Line2D([0], [0], color=(0.3, 0, 0.6), lw=2))
    handles.append(Line2D([0], [0], color=(0.6, 0, 0), lw=2))
    labels.append(f"{outcomes[2]} base rate ({(1 - top_line):.3f})")
    labels.append(f"{outcomes[0]} base rate ({bot_line:.3f})")

    # set x tick labels so that the boundary between bars shows the split value
    xtick_labels = [0]
    for i in grouping_counts.index.get_level_values(0).unique():
        if grouping_counts.loc[i].sum() > grouping_counts.sum().sum() * 0.02:
            if isinstance(i, (str, int)):
                xtick_labels.append(i)
            elif isinstance(i, pd._libs.interval.Interval):
                xtick_labels.append(f"{i.right:.2f}")
            else:
                raise TypeError(f"Unknown type for xtick label: {type(i)}")
    if not col == "TempSeqNumLinks":
        xticks = [patch.xy[0] for i, patch in enumerate(ax.patches) if i % 3 == 0]
        xticks.append(1)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation=45)
    else:
        xtick_labels.extend(["", "", f"{len(xtick_labels)-1}+"])
        ax.set_xticklabels(xtick_labels[1:])

    # Set y ticks and legend
    yticks = [i / 10 for i in range(11)]
    plt.yticks(yticks, yticks)
    plt.legend(
        handles=handles, labels=labels, bbox_to_anchor=(1.04, 1), loc="upper left",
    )

    ax.set_ylabel("Proportion of Trials")
    ax.set_xlabel(xlabel)

    # Add KS statistic to legend
    if feature_ks is not None:
        for i in range(len(labels)):
            if labels[i] in outcomes and "Inc" not in labels[i] and labels[i] != "FP":
                ks_stat = feature_ks[f"{labels[i]}_statistic"]
                pvalue = feature_ks[f"{labels[i]}_pvalue"]
                labels[i] = f"{labels[i]} [KS={ks_stat:.3f}, p={pvalue:.3f}]"
    elif col == "TempSeqNumLinks":  # special case, use chisq values
        for i in range(len(labels)):
            if labels[i] in outcomes and "Inc" not in labels[i] and labels[i] != "FP":
                labels[i] = f"{labels[i]} [$\u03A7^2$={chisq[i]:.3f}, p={p[i]:.3f}]"

    plt.legend(
        handles, labels, bbox_to_anchor=(1.04, 1), loc="upper left",
    )

    # Write n=[num samples] on the image
    plt.text(
        1.05, 0, f"n={grouping_counts.sum().sum():.0f}", transform=plt.gca().transAxes, ha="center",
    )

    plt.savefig(output_fp, bbox_inches="tight")
    plt.close()


def plot_cdf(df, col):
    plt.clf()
    fig, ax = plt.subplots(figsize=(7, 7))

    cdf1 = ECDF(df[df["Outcome"] == "TP"][col])
    cdf2 = ECDF(uniform.rvs(size=10000))

    x = np.concatenate([cdf1.x, cdf2.x])
    x = np.sort(x)
    x = np.unique(x)
    x = x[~np.isnan(x)]
    x[0] = 0

    cdf1_y = np.interp(x, cdf1.x[~np.isnan(cdf1.x)], cdf1.y[~np.isnan(cdf1.x)])
    cdf2_y = np.interp(x, cdf2.x, cdf2.y)

    cdf_diffs = cdf1_y - cdf2_y

    plt.plot(x, cdf1_y, color="blue", label="TP")
    # plt.plot(x, cdf1_y, color="darkorange", label="FN")
    plt.plot(x, x, color="grey", label="uniform distribution")

    idx = np.argmax(np.abs(cdf_diffs))
    ymin = min(cdf1_y[idx], cdf2_y[idx])
    ymax = max(cdf1_y[idx], cdf2_y[idx])
    plt.vlines(
        x=x[idx],
        ymin=ymin,
        ymax=ymax,
        linestyles="dashed",
        color="r",
        label=f"KS Statistic: {abs(cdf1_y[idx] - cdf2_y[idx]):.3f}",
    )

    plt.legend()
    plt.title(f"CDF for {col}")
    plt.xlabel("Percentile (Relative to Trials on Same Image Pair)")
    plt.ylabel(f"Proportion of trials with {col} < percentile")

    plt.savefig(
        os.path.join("visualizations", "cdfs", f"{col.replace(' ', '_')}.jpg"), bbox_inches="tight"
    )
    plt.close()


def save_scatter_plot(df, features, fill_kwargs, output_fp):
    assert len(features) == 2

    plt.clf()
    fig, ax = plt.subplots(figsize=(7, 7))
    if features[0] == "Phase C Time":
        ax.set(yscale="log")

    mates_idx = df["Mating"] == "Mates"
    not_nan = ~np.isnan(df[features[0]]) & ~np.isnan(df[features[1]])
    if (features[0] == "Phase C Time") & (features[1] == "Pct Deciding"):
        sns.scatterplot(
            x=features[1],
            y=features[0],
            hue="Outcome",
            data=df[mates_idx],
            palette={"TP": "blue", "FN": "darkorange", "IncMated": "grey"},
        )
        ax.fill_between(**fill_kwargs, facecolor=(1, 0, 0, 0.2), edgecolor="r", linewidth=0.0)
        ax.fill_between(
            x=[0.15, df[features[1]].max()],
            y1=0,
            y2=20,
            facecolor=(1, 0, 0, 0.2),
            edgecolor="r",
            linewidth=0.0,
        )

        yticks = [10, 20, 50, 100, 200, 500, 1000, 2000]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks)
    elif (features[0] == "TempSeqNumLinks") & (features[1] == "TempSeqAveDeviation"):
        max_links = df["TempSeqNumLinks"].max()
        max_dev = df["TempSeqAveDeviation"].max()

        df["TempSeqNumLinks"] = df["TempSeqNumLinks"].astype("category")
        df["TempSeqNumLinks"] = df["TempSeqNumLinks"].cat.set_categories(
            df["TempSeqNumLinks"].cat.categories[::-1]
        )

        # ax.fill_between(**fill_kwargs, facecolor=(1, 0, 0, 0.2), edgecolor="r", linewidth=0.0)
        ax.fill_between(
            x=[5, max_dev],
            y1=[0, 0],
            y2=[max_links, max_links],
            facecolor=(1, 0, 0, 0.2),
            edgecolor="r",
            linewidth=0.0,
        )
        sns.stripplot(
            x=features[1],
            y=features[0],
            jitter=0.25,
            hue="Outcome",
            data=df[mates_idx],
            palette={"TP": "blue", "FN": "darkorange", "IncMated": "grey"},
        )
    else:
        raise NotImplementedError(f"Scatter plot for features {features} not implemented.")

    for i, feature in enumerate(features):
        name, units = get_details(feature)
        label = f"{name} ({units})" if units is not None else name
        if i == 0:
            plt.ylabel(label)
        else:
            plt.xlabel(label)

    ax.text(
        0.95, -0.1, f"n={len(df[mates_idx & not_nan])}", transform=ax.transAxes,
    )

    plt.savefig(output_fp, bbox_inches="tight")
    plt.close()


def main():
    vis_dir = "visualizations"
    os.makedirs(vis_dir, exist_ok=True)

    df = pd.read_csv(os.path.join("data", "cleaned_data.csv"), usecols=STATS_COLS)
    df = df[df["Prefix"] != 0]

    # uncomment the next two lines to remove mated image pairs that had no TP outcomes
    # pairs_with_tp = df.groupby(["Outcome", "ImagePair"]).size()["TP"].index
    # df = df[df["ImagePair"].isin(pairs_with_tp) | df["Mating"] == "Nonmates"]

    # Rename long column names
    name_mapper = {
        "EMDDistanceToCorrect_C_Left": "EMDToCorrect_Left",
        "EMDDistanceToCorrect_C_Right": "EMDToCorrect_Right",
        "AveDeviationFromGroundTruthTempSeqFits_Detail_JstUnq_NormTransMat_NoClusterPrune_BW66_Thresh0.3InRidgeWidths": "TempSeqAveDeviation",
        "NumHighQualityLinksTempSeqFits_Detail_JstUnq_NormTransMat_NoClusterPrune_BW66_Thresh0.3": "TempSeqNumLinks",
        "AnalysisPercentVisitedCellsVisitedLatent": "A-PctCellsVisited",
        "ComparePercentVisitedCellsVisitedLatent": "C-PctCellsVisited",
        "AnalysisPropFixNearWBMinutiaK=22": "A-PctNearMinutiae",
        "ComparisonPropFixNearWBMinutiaK=22": "C-PctNearMinutiae",
    }
    df = df.rename(columns=name_mapper)
    for long_col_name in name_mapper:
        STATS_COLS.remove(long_col_name)
        STATS_COLS.append(name_mapper[long_col_name])

    discrete_cols = ["AllIndvClustLinksFoundBW60", "TempSeqNumLinks"]
    outcome_order = [
        "TP",
        "FN",
        "IncMated",
        "IncNonMated",
        "FP",
        "TN",
    ]

    # Split Inc into Incs on Mates vs Incs on Nonmates
    inc_trials = df["Outcome"] == "Inc"
    mated_pair_trials = df["Mating"] == "Mates"
    df.loc[inc_trials & mated_pair_trials, "Outcome"] = "IncMated"
    df.loc[inc_trials & ~mated_pair_trials, "Outcome"] = "IncNonMated"

    # Set PctClarBlue to nan if no one looked at blue on the image pair (assume no blue at all)
    mean_blue = df.groupby("ImagePair")["PctClarBlue"].mean()
    no_blue = mean_blue[mean_blue == 0].index

    # Create versions of the dataset
    df_ranked = ranked_df(df, by_image_pair=False)
    df_ranked_by_ip = ranked_df(df, by_image_pair=True)
    df_ranked_by_ip["PctClarBlue"] = df_ranked_by_ip["PctClarBlue"].mask(
        df_ranked_by_ip["ImagePair"].isin(no_blue)
    )
    dfs = {"normal": df, "ranked": df_ranked, "ranked_by_ip": df_ranked_by_ip}

    # Make output dirs
    for plot_type in ["boxplots", "hists", "heatmaps", "cdfs", "scatterplots"]:
        os.makedirs(os.path.join(vis_dir, plot_type), exist_ok=True)
    for mating in ["Mates", "Nonmates"]:
        for df_name in dfs:
            os.makedirs(os.path.join(vis_dir, "distplots", df_name, mating), exist_ok=True)

    # Plot difficulty
    plot_outcome_proportion_by_difficulty(
        df, os.path.join("visualizations", "hists", "Difficulty.jpg")
    )

    # Two feature heatmaps
    pairs = [
        ("Phase C Time", "Pct Deciding"),
        ("TempSeqNumLinks", "TempSeqAveDeviation"),
    ]
    highlight_regions = {
        "TempSeqNumLinks": [11.5, 8.5],
        "TempSeqAveDeviation": [5, df["TempSeqAveDeviation"].max()],
        "Phase C Time": [0, df["Phase C Time"].max()],
        "Pct Deciding": [0, 0.15],
    }
    ranked_all = ranked_df(df, by_image_pair=False)
    for pair in pairs:
        image_fp = f"{pair[0].replace(' ', '_')}_vs_{pair[1].replace(' ', '_')}.jpg"
        two_feature_heatmap(ranked_all, pair, os.path.join("visualizations", "heatmaps", image_fp))

        x = highlight_regions[pair[1]]
        y1 = highlight_regions[pair[0]][0]
        y2 = highlight_regions[pair[0]][1]
        fill_kwargs = dict(x=x, y1=y1, y2=y2,)
        save_scatter_plot(
            df, pair, fill_kwargs, os.path.join("visualizations", "scatterplots", image_fp)
        )

    # Run KS tests
    ks_features = [
        f for f in STATS_COLS if f not in NON_NUMERIC_FEATURES and f not in discrete_cols
    ]
    ks_results = dict()
    for df_name in dfs:
        if df_name == "normal":
            continue

        selected_df = dfs[df_name]
        ks_results[df_name] = dict()

        ks_results[df_name]["Mates"] = calculate_ks_test_results(
            df=selected_df[selected_df["Mating"] == "Mates"],
            features=ks_features,
            outcomes=["TP", "FN"],
        )
        ks_results[df_name]["Nonmates"] = calculate_ks_test_results(
            df=selected_df[selected_df["Mating"] == "Nonmates"],
            features=ks_features,
            outcomes=["TN"],
        )

    # Plot features
    for col in tqdm(df.columns):
        if col in NON_NUMERIC_FEATURES:
            continue
        col_filename = f"{col.replace(' ', '_')}.jpg"
        name, units = get_details(col)

        # Boxplots
        if col in discrete_cols:
            df[col] = df[col].astype(int).astype("category")
            for mating in ["Mates", "Nonmates"]:
                if mating == "Mates":
                    outcomes = ["FN", "IncMated", "TP"]
                else:
                    outcomes = ["FP", "IncNonMated", "TN"]
                plot_feature_dist(
                    data=df[df["Mating"] == mating],
                    col=col,
                    outcomes=outcomes,
                    xlabel=f"Number of Correspondence Attempts [{mating}]",
                    output_fp=os.path.join(vis_dir, "hists", f"{mating}_{col_filename}"),
                )
            continue
        else:
            plot_feature_boxplot(
                output_fp=os.path.join(vis_dir, "boxplots", col_filename),
                x=col,
                y="Outcome",
                data=df,
                outcome_order=outcome_order,
            )

        plot_cdf(df_ranked_by_ip, col)

        # Dist plots for mated and nonmated pairs
        for df_name in dfs:
            selected_df = dfs[df_name]
            subset_config = {
                "Mates": {
                    "df": selected_df[selected_df["Mating"] == "Mates"],
                    "outcomes": ["FN", "IncMated", "TP"],
                },
                "Nonmates": {
                    "df": selected_df[selected_df["Mating"] == "Nonmates"],
                    "outcomes": ["FP", "IncNonMated", "TN"],
                },
            }
            for matedness in subset_config:
                xlabel = name
                if df_name == "normal":
                    xlabel += (
                        f" ({units}) [{matedness}]" if units is not None else f" [{matedness}]"
                    )
                    _, bins = pd.qcut(
                        x=subset_config[matedness]["df"][col], q=10, retbins=True, duplicates="drop"
                    )
                    bins[0] -= 0.001  # correct so that lower bound is included in the bin
                    feature_ks = ks_results["ranked"][matedness].loc[col]
                else:
                    xlabel += (
                        f" (percentile) [{matedness}]" if units is not None else f" [{matedness}]"
                    )
                    if df_name == "ranked":
                        xlabel += "\nRanked Across All Trials"
                    elif df_name == "ranked_by_ip":
                        xlabel += "\nRanked Within Image Pair"
                    bins = [(i / 10) for i in range(11)]
                    bins[0] -= 0.001
                    feature_ks = ks_results[df_name][matedness].loc[col]

                plot_feature_dist(
                    data=subset_config[matedness]["df"],
                    col=col,
                    outcomes=subset_config[matedness]["outcomes"],
                    xlabel=xlabel,
                    output_fp=os.path.join(vis_dir, "distplots", df_name, matedness, col_filename),
                    bins=bins,
                    feature_ks=feature_ks,
                )


if __name__ == "__main__":
    main()
