import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import re
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from config import (
    BERT_LIKE_MODELS_DICT,
    INFERENCE_FULL_PATH,
    INDIE_VAR_NAME,
    DATASET_STYLE,
    PLOTS_FULL_PATH,
    STATS_FULL_PATH,
)

# # ######################################### PLOTTING PARAMS ############################################
LEGEND_LABELS = ["Female", "Male"]
DECIMAL_PLACES = 1

# Common paper layout
Y_LABEL = "Averaged softmax probability"
LABEL_ROTATION = 90
COLORS = ["m", "g", "b", "r", "k"]
LEGEND_LABELS = ["Female", "Male"]


# # ######################################### PLOTTING FUNCS ############################################

def get_results(filename, indie_var_name):
    results = pd.read_csv(filename, header=0)
    results.set_index(indie_var_name, inplace=True)
    results = results.drop("Unnamed: 0", axis=1)
    return results


def plot_subfig(filenames, title, axe, max_xlabels=10, dir=INFERENCE_FULL_PATH):
    for i, filename in enumerate(filenames):
        result = get_results(f"{dir}/{filename}.csv", INDIE_VAR_NAME)
        mean = result.mean(axis=1)
        # In absence of better gender-equity spectrum
        xs = list(range(len(mean)))
        coef = np.polyfit(xs, mean, 1)
        _poly1d_fn = np.poly1d(coef)
        axe.plot(mean, "o", color=COLORS[i]) 
        axe.set_title(title)
    axe.xaxis.set_major_locator(MaxNLocator(max_xlabels))


def get_results_all_models(
    filename, indie_var_name, baseline_included=False, dir=INFERENCE_FULL_PATH
):
    results = pd.read_csv(f"{dir}/{filename}", header=0) 
    results.set_index(indie_var_name, inplace=True)
    # dates should be strings like other indie vars
    results.index = results.index.astype("str")
    results = results.drop("Unnamed: 0", axis=1)
    if baseline_included:
        results = results.drop("baseline", axis=0)
    return results


def process_results_from_csvs(
    filenames,
    _y_label,
    legend_labels,
    title,
    axe,
    indie_var_name,
    max_xlabels=9,
    n_fit=1,
    no_plot=False,
    colors=COLORS,
):

    result_stats = []
    for i, filename in enumerate(filenames):
        result = get_results_all_models(filename, indie_var_name)
        ys = result.mean(axis=1)
        # In absence of better gender-equity spectrum
        xs = list(range(len(ys)))

        if not no_plot:
            # https://stackoverflow.com/questions/28505008/numpy-polyfit-how-to-get-1-sigma-uncertainty-around-the-estimated-curve
            p, C_p = np.polyfit(xs, ys, n_fit, cov=True)
            t = np.linspace(min(xs) - 1, max(xs) + 1, 10 * len(xs))  # This is x-axis
            TT = np.vstack([t ** (n_fit - i) for i in range(n_fit + 1)]).T

            # matrix multiplication calculates the polynomial values
            yi = np.dot(TT, p)
            C_yi = np.dot(TT, np.dot(C_p, TT.T))  # C_y = TT*C_z*TT.T
            sig_yi = 2 * np.sqrt(np.diag(C_yi))

            axe.fill_between(t, yi + sig_yi, yi - sig_yi, color=colors[i], alpha=0.25)
            axe.plot(t, yi, "-", color=colors[i])
            axe.plot(ys, "o", color=colors[i], label=legend_labels[i])

            axe.set_title(title)
        lin_stats = stats.linregress(xs, ys)
        result_stats.append(
            pd.DataFrame(
                {
                    "slope": [round(lin_stats.slope, DECIMAL_PLACES + 2)],
                    "r": [round(lin_stats.rvalue, DECIMAL_PLACES + 2)],
                },
                index=[filename],
            )
        )

    axe.xaxis.set_major_locator(MaxNLocator(max_xlabels))
    return result_stats


def plot_slope_and_r(
    fit_stats,
    model_dict,
    indie_var_name,
    figsize,
    dataset_name,
    filenames,
    no_legend=False,
    dir=PLOTS_FULL_PATH,
):
    w = "Year" if indie_var_name == "date" else "Country"

    fit_stats_df = pd.concat(fit_stats)
    fit_stats_df.to_csv(f"{STATS_FULL_PATH}/fit_stats_{filenames[0]}")

    f_stats = fit_stats_df.filter(like="fp", axis=0)
    named_f_stats = f_stats.rename(
        index=lambda x: model_dict[
            re.search(f".*_{dataset_name}_(.*)_{indie_var_name}.*", x).group(1)
        ]
    )
    m_stats = fit_stats_df.filter(like="mp", axis=0)
    named_m_stats = m_stats.rename(
        index=lambda x: model_dict[
            re.search(f".*_{dataset_name}_(.*)_{indie_var_name}.*", x).group(1)
        ]
    )

    title = f"Delta Slope and r Coefficients for {w} vs Gender Pronouns"
    y_label = "Delta Slope and r Coefficients"
    plt.figure(figsize=figsize)
    named_stats = named_f_stats - named_m_stats
    named_stats.plot(kind="bar", color=["blue", "skyblue"], figsize=figsize)
    plt.title(title)
    plt.ylabel(y_label)
    if no_legend:
        plt.legend("", frameon=False)
    else:
        plt.legend(["Slope", "r"], loc="best")
    file_path = os.path.join(dir, f"delta_fit_stats_plot{filenames[0].strip('.csv')}")
    plt.savefig(file_path, dpi=300)
    print(f"Saved plots to {file_path}")

###################################### PLOTS ############################################

iclr_v2_fig_cols = 2
iclr_v2_height = 6
iclr_v2_width = 6
legend_labels = LEGEND_LABELS
test_version = f"__normFalse"  # TODO: Move to config.py

fig, ax = plt.subplots(2, iclr_v2_fig_cols, sharex=True)
fig.set_figheight(iclr_v2_height)
fig.set_figwidth(iclr_v2_width)
fig.suptitle("Predicted Pronoun vs Year", fontsize=16)
y_label = Y_LABEL


fit_stats = []


axe = ax[0][0]
title = "Pre-trained BERT base"
filenames = [
    f"fp_{DATASET_STYLE}_bert-base-uncased_{INDIE_VAR_NAME}_{test_version}.csv",
    f"mp_{DATASET_STYLE}_bert-base-uncased_{INDIE_VAR_NAME}_{test_version}.csv",
]


fit_stats.extend(
    process_results_from_csvs(
        filenames, y_label, legend_labels, title, axe, INDIE_VAR_NAME
    )
)


axe = ax[1][0]
title = "Pre-trained BERT large"
filenames = [
    f"fp_{DATASET_STYLE}_bert-large-uncased_{INDIE_VAR_NAME}_{test_version}.csv",
    f"mp_{DATASET_STYLE}_bert-large-uncased_{INDIE_VAR_NAME}_{test_version}.csv",
]
fit_stats.extend(
    process_results_from_csvs(
        filenames, y_label, legend_labels, title, axe, INDIE_VAR_NAME
    )
)
axe.tick_params(axis="x", labelrotation=LABEL_ROTATION)


axe = ax[0][1]
title = "Pre-trained RoBERTa base"
filenames = [
    f"fp_{DATASET_STYLE}_roberta-base_{INDIE_VAR_NAME}_{test_version}.csv",
    f"mp_{DATASET_STYLE}_roberta-base_{INDIE_VAR_NAME}_{test_version}.csv",
]

fit_stats.extend(
    process_results_from_csvs(
        filenames, y_label, legend_labels, title, axe, INDIE_VAR_NAME
    )
)


axe = ax[1][1]
title = "Pre-trained RoBERTa large"
filenames = [
    f"fp_{DATASET_STYLE}_roberta-large_{INDIE_VAR_NAME}_{test_version}.csv",
    f"mp_{DATASET_STYLE}_roberta-large_{INDIE_VAR_NAME}_{test_version}.csv",
]
fit_stats.extend(
    process_results_from_csvs(
        filenames, y_label, legend_labels, title, axe, INDIE_VAR_NAME
    )
)
axe.tick_params(axis="x", labelrotation=LABEL_ROTATION)

print(pd.concat(fit_stats))
axe.tick_params(axis="x", labelrotation=LABEL_ROTATION)

handles, labels = axe.get_legend_handles_labels()
fig.legend(handles, labels) # TODO Add loc, bbox_to_anchor
# fig.tight_layout(rect=?)
# fig.subplots_adjust(top=?)

file_path = os.path.join(PLOTS_FULL_PATH, f"final_fig_{DATASET_STYLE}_{INDIE_VAR_NAME}_{test_version}")
plt.savefig(file_path, dpi=900)
print(f"Saved plots to {file_path}")


plot_slope_and_r(
    fit_stats,
    BERT_LIKE_MODELS_DICT,
    INDIE_VAR_NAME,
    (iclr_v2_width, iclr_v2_height/3),
    DATASET_STYLE,
    filenames,
)
