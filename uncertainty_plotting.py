# %%
import matplotlib.patches as mpatches
import csv
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

import os
from pathlib import Path
from config import (
    BERT_LIKE_MODELS_DICT,
    BERT_LIKE_MODELS,
    WINOGENDER_SCHEMA_PATH,
    INFERENCE_FULL_PATH,
    PLOTS_FULL_PATH,
    STATS_FULL_PATH,
    VERY_VERBOSE,
    DATES, # TODO: make gender use of INDIE_VARS, note TESTING
)

# %%
NUM_AVE_UNCERTAIN = 2
STATIC_WINO_PARTICIPS = ["man", "woman", "someone"]
ALL_WINO_PARTICIPS = STATIC_WINO_PARTICIPS + ["other"]


# %%
# # # ######################################### PLOTTING PARAMS ############################################

COLORS = ["m", "g", "b", "r", "k"]
LEGEND_LABELS = ["Female", "Male"]



def get_uncertainty_metric(softmax_data):
    return abs(
        softmax_data[-NUM_AVE_UNCERTAIN:].values.mean()
        - softmax_data[:NUM_AVE_UNCERTAIN].values.mean()
    )


def get_df_per_model_from_all_files_in_dir(inference_path, indie_vars=None):
    files = Path(inference_path).glob("*.csv")  

    # Will init model_results as below for plotting:
    # {'bert-base-uncased': {'man': [], 'woman': [], 'someone': [], 'other': []},
    # ...
    # 'roberta-large': {'man': [], 'woman': [], 'someone': [], 'other': []}}
    model_results = {
        model_name: {participants: [] for participants in ALL_WINO_PARTICIPS}
        for model_name in BERT_LIKE_MODELS
    }
    # Will init model_results similarly but with defaultdict as below for plotting
    stats_model_results = {model_name: defaultdict() for model_name in BERT_LIKE_MODELS}

    for f in files:

        prog = re.compile(r".*(.)p_wikibio_([\w-]+)_(\w+)___(\w+)_(\w+)_([01])")
        m = prog.match(f.stem)
        try:
            (
                gender,
                model_name,
                indie_var_extracted,
                occupation,
                participant,
                version,
            ) = (
                m.group(1),
                m.group(2),
                m.group(3),
                m.group(4),
                m.group(5),
                m.group(6),
            )
        except AttributeError:
            if VERY_VERBOSE:
                print(f"Skipping {f.stem}. We don't plot test files")
            continue

        col_name = f"{gender}_{indie_var_extracted}_{occupation}_{version}"
        try:
            data = pd.read_csv(f)
            if indie_vars is None:
                indie_vars = data[indie_var_extracted]
            data.drop("Unnamed: 0", axis=1, inplace=True)

            # Dropping first row, that contains the input text, eg dropping:
            # "In DATE: The counselor disclosed to the patient that he was..."
            data.drop(indie_var_extracted, axis=1, inplace=True)
            data.columns = [col_name]

            if participant not in STATIC_WINO_PARTICIPS:
                participant = "other"

            model_results[model_name][participant].append(data)

            particip_version = f"{participant}_{version}"
            if not stats_model_results[model_name].keys().__contains__(occupation):
                stats_model_results[model_name][occupation] = defaultdict()
            if (
                not stats_model_results[model_name][occupation]
                .keys()
                .__contains__(particip_version)
            ):
                stats_model_results[model_name][occupation][
                    particip_version
                ] = defaultdict()
            stats_model_results[model_name][occupation][particip_version] = data

        except pd.errors.EmptyDataError as _e:
            print(f"No data for {col_name} in {f.stem}")

    return model_results, stats_model_results, indie_var_extracted


# %%


def load_examples(path=WINOGENDER_SCHEMA_PATH):
    bergsma_pct_female = {}
    bls_pct_female = {}
    with open(os.path.join(path, "occupations-stats.tsv")) as f:
        next(f, None)  # skip the headers
        for row in csv.reader(f, delimiter="\t"):
            occupation = row[0]
            bergsma_pct_female[occupation] = float(row[1])
            bls_pct_female[occupation] = float(row[2])
    return bergsma_pct_female, bls_pct_female


# %%
def get_wino_results_uncertainty(path, model_results, bls_pct_female_sorted_dict):

    wino_results = {}
    for model in BERT_LIKE_MODELS:
        wino_results[model] = {}
        for particip in ALL_WINO_PARTICIPS:
            wino_results[model][particip] = pd.concat(
                model_results[model][particip], axis=1
            )

    detail_filters = "f_0", "f_1", "m_0", "m_1"

    # %%
    wino_results_uncertainty = {model_name: dict() for model_name in BERT_LIKE_MODELS}

    for occ in bls_pct_female_sorted_dict.keys():
        for model in BERT_LIKE_MODELS:
            wino_results_uncertainty[model][occ] = dict()
            occ_filter = f"date_{occ}"

            for particip in ALL_WINO_PARTICIPS:
                wino_results_uncertainty[model][occ][particip] = dict()

                for filter_version in detail_filters:
                    f = filter_version.split("_")
                    filter = f"{f[0]}_{occ_filter}_{f[1]}"

                    filtered_results = wino_results[model][particip].filter(
                        like=filter, axis=1
                    )

                    wino_results_uncertainty[model][occ][particip][
                        filter_version
                    ] = get_uncertainty_metric(filtered_results)

    return wino_results, wino_results_uncertainty


def threshold_df(df, col, threshold, greater_than=False):
    if greater_than:
        return df[df[col] > threshold]
    else:
        return df[df[col] <= threshold]


# %%
def get_uc_model_stats(stats_model_results, model):

    uc_model_stats = {occ: defaultdict() for occ in stats_model_results[model].keys()}
    for occ in uc_model_stats.keys():
        for p_name, p_data in stats_model_results[model][occ].items():
            uc_model_stats[occ][p_name] = get_uncertainty_metric(p_data)

    uc_model_stats_df = pd.DataFrame.from_dict(uc_model_stats).transpose()

    uc_model_stats_df.to_csv(
        os.path.join(STATS_FULL_PATH, "roberta_large_delta_stats.csv")
    )
    return uc_model_stats_df


def get_true_positives(uc_model_stats_df, underspec_particip_list, threshold=1.0):
    tp = 0
    for p_v in underspec_particip_list:
        tp += len(threshold_df(uc_model_stats_df, p_v, threshold, greater_than=True))
    all_p = len(underspec_particip_list) * len(uc_model_stats_df)
    return tp / all_p


def get_true_negatives(uc_model_stats_df, wellspec_particip_list, threshold=1.0):
    tn = 0
    for p_v in wellspec_particip_list:
        tn += len(threshold_df(uc_model_stats_df, p_v, threshold))
    all_n = len(wellspec_particip_list) * len(uc_model_stats_df)
    return tn / all_n


# %%


def plot_all_occ_detail(wino_results, model="roberta-large", occ="Doctor"):
    fig, ax = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(6)
    sent_n = 0
    for i, particip in enumerate(ALL_WINO_PARTICIPS):
        sent_n += 1
        # Arbitrarily selecting female pronouns for this plot
        ax.plot(
            DATES,
            wino_results[model][particip][f"f_date_{occ.lower()}_0"],
            color=COLORS[i],
            marker="o",
            label=f"{sent_n}) Partcipant: '{particip}'; Coreferent: {occ}",
        )
    for i, particip in enumerate(ALL_WINO_PARTICIPS):
        sent_n += 1
        ax.plot(
            DATES,
            wino_results[model][particip][f"f_date_{occ.lower()}_1"],
            color=COLORS[i],
            marker="v",
            label=f"{sent_n}) Partcipant: '{particip}'; Coreferent: '{particip}'",
        )

    handles, labels = ax.get_legend_handles_labels()

    legend_loc = "center"
    legend_anchor = (0.0, 0.0, 1.0, 1.33)

    fig.legend(handles, labels, loc=legend_loc, bbox_to_anchor=legend_anchor)

    plt.ylabel("Softmax Probability over Gendered Predictions")

    plt.title(f"Softmax Probability for Female Pronouns vs Date in {occ} Sentence")
    file_path = os.path.join(PLOTS_FULL_PATH, f"female_{occ}_{model}")
    plt.savefig(file_path, dpi=300)
    print(f"Plot saved to {file_path}")


# %%
def plot_all_winogender_occs(
    bls_pct_female_sorted_dict, wino_results_uncertainty, occ_detail="doctor"
):
    # NOTE: Run with SNS poster setting for larger font
    VERSIONS = {"f_0": "Professional", "f_1": "Participant"}

    MARKERS = ["|", "_", "3", "4"]
    for model_name in BERT_LIKE_MODELS:  
        for version in VERSIONS.keys():
            fig, ax = plt.subplots() 
            fig.set_figheight(4)
            fig.set_figwidth(20)
            added_legend_items = False
            for occ in bls_pct_female_sorted_dict.keys():
                for i, particip in enumerate(ALL_WINO_PARTICIPS):
                    stats = wino_results_uncertainty[model_name][occ][particip][version]
                    if not added_legend_items:
                        ax.scatter(
                            occ,
                            stats,
                            color=COLORS[i],
                            marker=MARKERS[i],
                            alpha=0.95,
                            label=f"Participant identified as '{particip}'",
                        )
                    else:
                        ax.scatter(
                            occ, stats, color=COLORS[i], marker=MARKERS[i], alpha=0.95
                        )
                    if occ == occ_detail:
                        print(f"uncertainty: {model_name} {occ} {particip} {version}")
                        print(stats)
                added_legend_items = True

            ax.tick_params(axis="x", labelrotation=90)
            ax.set_ylim([-10, 60])
            ax.margins(x=0.01)
            fig.tight_layout()
            plt.legend()
            title = f" {BERT_LIKE_MODELS_DICT[model_name]} Uncertainty in Pronoun Prediction for the {VERSIONS[version]} in Winogender-like Texts"
            plt.ylabel("Uncertainty Metric")
            plt.title(title)
            plt.subplots_adjust(left=0.05, right=0.99, bottom=0.2, top=0.96)

            file_path = os.path.join(PLOTS_FULL_PATH, f"bls_{title.replace(' ', '_')}")
            print(f"Plot saved to {file_path}")
            plt.savefig(file_path, dpi=300)


if __name__ == "__main__":
    # Per the winogender_schema
    wellspec_particip_versions = ["man_1", "woman_1"]
    underspec_particip_versions = [
        "man_0",
        "woman_0",
        "someone_0",
        "other_0",
        "someone_1",
        "other_1",
    ]

    _, bls_pct_female = load_examples()
    bls_pct_female_sorted = sorted(bls_pct_female.items(), key=lambda item: item[1])
    bls_pct_female_sorted_dict = {
        occ: {"pct": pct} for occ, pct in bls_pct_female_sorted
    }

    (
        model_results,
        stats_model_results,
        indie_var_extracted,
    ) = get_df_per_model_from_all_files_in_dir(INFERENCE_FULL_PATH)

    # %%

    wino_results, wino_results_uncertainty = get_wino_results_uncertainty(
        INFERENCE_FULL_PATH, model_results, bls_pct_female_sorted_dict
    )

    plot_all_occ_detail(wino_results)
    plot_all_winogender_occs(bls_pct_female_sorted_dict, wino_results_uncertainty)

    # %%
    model_detail = "roberta-large"
    uc_model_stats = get_uc_model_stats(stats_model_results, "roberta-large")

    tp_all = get_true_positives(
        uc_model_stats,
        underspec_particip_versions,
        threshold=1.0,
    )

    tn_all = get_true_negatives(
        uc_model_stats,
        wellspec_particip_versions,
        threshold=1.0,
    )

    tp_gender = get_true_positives(
        uc_model_stats,
        ["man_0", "woman_0"],
        threshold=1.0,
    )

    print(
        f"For {model_detail}, true postive: {tp_all}, true negative: {tn_all}, and true postive gender co-occuring: {tp_gender} ",
    )

# %%
