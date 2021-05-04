import os

from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold


class Config:
    def __init__(
        self, experiment, trials, group_pairs, use_group_stats, use_clustering, separate=None
    ):
        """
        :param experiment: 'tn_vs_fn', 'tp_vs_inc', 'no_inc', '4_outcome', 'ex_vs_id', '1_vs_rest'
        :param trials: 'CE', 'CW', 'all', or 'no_prefix'
        :param use_group_stats: bool, use statistics that use information from other trials
        :param separate: only used if experiment is '1_vs_rest', the class to be considered separately
        """
        self.cv = StratifiedKFold(n_splits=5, shuffle=True)
        self.data_csv = os.path.join("data", "cleaned_data.csv")
        self.use_clustering = use_clustering
        self.use_group_stats = use_group_stats
        self.corr_threshold = 0.75
        self.scorer = make_scorer(score_func=f1_score, average="weighted")
        self.experiment = experiment
        self.group_pairs = group_pairs

        if experiment == "tn_vs_fn":
            self.target = "Outcome"
            self.drop_info = {
                "Outcome": ["TP", "Inc"],
                "Mating": "all",
                "Conclusion-Simple": "all",
            }
            self.target_mapping = {"TN": 0, "FN": 1}
        elif experiment == "tp_vs_inc":
            self.target = "Outcome"
            self.drop_info = {
                "Outcome": ["FN", "TN"],
                "Mating": ["Mates"],
                "Conclusion-Simple": "all",
            }
            self.target_mapping = {"TP": 0, "Inc": 1}
        elif experiment == "no_inc":
            self.target = "Outcome"
            self.drop_info = {
                "Outcome": ["Inc"],
                "Mating": "all",
                "Conclusion-Simple": "all",
            }
            self.target_mapping = {"FN": 0, "TN": 1, "TP": 2}
        elif experiment == "4_outcome":
            self.target = "Outcome"
            self.drop_info = {
                "Mating": "all",
                "Conclusion-Simple": "all",
            }
            self.target_mapping = {"FN": 0, "TN": 1, "TP": 2, "Inc": 3}
        elif experiment == "ex_vs_id":
            self.target = "Conclusion-Simple"
            self.drop_info = {
                "Conclusion-Simple": ["Inc"],
                "Mating": "all",
                "Outcome": "all",
            }
            self.target_mapping = {"Ex": 0, "ID": 1}
        elif experiment == "1_vs_rest":
            assert separate is not None
            self.target = "Outcome"
            self.drop_info = {
                "Mating": "all",
                "Conclusion-Simple": "all",
            }
            self.target_mapping = {"FN": 0, "TN": 0, "TP": 0, "Inc": 0}
            self.target_mapping[separate] = 1
            self.class_names = [f"Not {separate}", separate]
        elif experiment == "tp_vs_fn":
            self.target = "Outcome"
            self.drop_info = {
                "Outcome": ["TN", "Inc"],
                "Mating": "all",
                "Conclusion-Simple": "all",
            }
            self.target_mapping = {"TP": 0, "FN": 1}
        elif experiment == "inc":
            self.target = "Mating"
            self.drop_info = {"Outcome": ["FN", "TN", "TP"], "Conclusion-Simple": "all"}
            self.target_mapping = {"Nonmates": 0, "Mates": 1}
        elif experiment == "mating":
            self.target = "Mating"
            self.drop_info = {"Outcome": "all"}
            self.target_mapping = {"Nonmates": 0, "Mates": 1}
        else:
            raise NotImplementedError(f"experiment param '{experiment}' not implemented.")

        if trials == "CE":
            self.drop_info["Prefix"] = [1]
        elif trials == "CW":
            self.drop_info["Prefix"] = [0]
        elif trials == "no_prefix":
            self.drop_info["Prefix"] = "all"
        elif trials == "all":
            pass
        else:
            raise NotImplementedError(f"trials param '{trials}' not implemented")

        if not use_group_stats and experiment != "examiners":
            self.drop_info["EMDDistanceToCorrect_C_Left"] = "all"
            self.drop_info["EMDDistanceToCorrect_C_Right"] = "all"
            self.drop_info["EMDDistanceJustDecidingToCorrect_C_Left"] = "all"
            self.drop_info["EMDDistanceJustDecidingToCorrect_C_Right"] = "all"

        if not hasattr(self, "class_names"):
            self.class_names = [key for key in self.target_mapping]

        self.is_binary = False
        if len(self.class_names) == 2:
            self.is_binary = True

        self.decode_target = {}
        for i, class_name in enumerate(self.class_names):
            self.decode_target[i] = class_name


PLOT_DETAILS = {
    "A-Left Fix Stdev": {
        "name": "Latent Fixation Standard Deviation during Analysis",
        "units": "ridge widths",
    },
    "A-PctCellsVisited": {"name": "Proportion of Image Visited during Analysis"},
    "C-PctCellsVisited": {"name": "Proportion of Image Visited during Comparison"},
    "A-PctNearMinutiae": {
        "name": "Proportion of Fixations Near Minutiae during Analysis",
    },
    "C-PctNearMinutiae": {
        "name": "Proportion of Fixations Near Minutiae during Comparison",
    },
    "Phase C Time": {
        "name": "Comparison Time",
        "units": "seconds"
    },
    "Phase A Time": {
        "name": "Analysis Time",
        "units": "seconds"
    },
    "FixationsBeforeSwitch": {
        "name": "Average Number of Fixations Before Switching Images",
    },
    "Pct Deciding": {
        "name": "Proportion of Fixations in Detail Subphase"
    },
    "C-Left Fix Stdev": {
        "name": "Latent Fixation Standard Deviation",
        "units": "ridge widths"
    },
    "C-Right Fix Stdev": {
        "name": "Exemplar Fixation Standard Deviation",
        "units": "ridge widths"
    },
    "PctClarBlue": {
        "name": "Proportion of Fixations in Highest Clarity",
    },
    "PctClarRedYellow": {
        "name": "Proportion of Fixations in Low Clarity",
    },
    "TempSeqAveDeviation": {
        "name": "Average Distance of Correspondence Attempts From Ground Truth",
        "units": "ridge widths"
    },
    "TempSeqNumLinks": {
        "name": "Number of Correspondence Attempts"
    },
    "EMDToCorrect_Left": {
        "name": "Earth Mover Distance to TP Fixations on Latent",
        "units": "ridge widths"
    },
    "EMDToCorrect_Right": {
        "name": "Earth Mover Distance to TP Fixations on Exemplar",
        "units": "ridge widths"
    },
}

STATS_COLS = [
    "ImagePair",
    "Examiner",
    "Prefix",
    "Mating",
    "Outcome",
    "Difficulty",
    "Conclusion-Simple",
    "EMDLeftCToRightCSelfDeciding",
    "Pct Deciding",
    "Pct NA",
    "Pct A-Left Fixations",
    "Phase C Time",
    "Num Fixations",
    "Pct C-Left Fixations",
    "Pct C-Right Fixations",
    "Phase A Time",
    # "AllIndvClustLinksFoundBW60",
    "Switched",
    "FixationsBeforeSwitch",
    "C-Left Fix Stdev",
    "C-Right Fix Stdev",
    "A-Left Fix Stdev",
    "PctClarBlue",
    "EMDLeftCToRightCSelf",
    "Pct Scanning",
    "PctClarRedYellow",
    "PctClarGreen",
    "EMDDistanceToCorrect_C_Left",
    "EMDDistanceToCorrect_C_Right",
    "AveDeviationFromGroundTruthTempSeqFits_Detail_JstUnq_NormTransMat_NoClusterPrune_BW66_Thresh0.3InRidgeWidths",
    # "AveDeviationFromGroundTruthTempSeqFits_Detail_JstUnq_NormTransMat_NoClusterPrune_BW66_Thresh0.3",
    "NumHighQualityLinksTempSeqFits_Detail_JstUnq_NormTransMat_NoClusterPrune_BW66_Thresh0.3",
    "AnalysisPercentVisitedCellsVisitedLatent",
    "ComparePercentVisitedCellsVisitedLatent",
    "AnalysisPropFixNearWBMinutiaK=22",
    "ComparisonPropFixNearWBMinutiaK=22",
    "SpeedFast",
    "SpeedMedium",
    "SpeedSlow",
    "SaccadeLong",
    "SaccadeMedium",
    "SaccadeShort",
]

NON_NUMERIC_FEATURES = [
    "ImagePair",
    "Examiner",
    "Difficulty",
    "Prefix",
    "Mating",
    "Outcome",
    "Conclusion-Simple",
    "Borderline",
]
