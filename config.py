# %%
import os
import numpy as np
from pathlib import Path

############ Uncertainty or spurious! ###############
UNCERTAINTY = False # Set to False if replicating the Spurious results  # TODO add asserts
TESTING = True # Set to True if testing on subsets of the challenge sets
INDIE_VAR_FOR_SPURIOUS = 1  # Select '0' to use "date", '1' to use "place" for Spurious
VERY_VERBOSE = False

INDIE_VAR_NAMES = ["date", "place"]
# In our plots, Uncertainty always used 'date', but Spurious used both
INDIE_VAR_NAME = INDIE_VAR_NAMES[0] if UNCERTAINTY else INDIE_VAR_NAMES[INDIE_VAR_FOR_SPURIOUS]

DATASET_STYLE = "wikibio"  # Paper does not cover "reddit"

UNCERTAINTY_GENDERED_LIST = [
    ['he', 'she'],
    ['him', 'her'],
    ['his', 'hers'],
    ["himself", "herself"],
    ['male', 'female'],
    # ['man', 'woman']  Explicitly added in Winogender Extended eval set
    ['men', 'women'],
    ["husband", "wife"],
    ['father', 'mother'],
    ['boyfriend', 'girlfriend'],
    ['brother', 'sister'],
    ["actor", "actress"],
]

SPURIOUS_GENDERED_LIST = UNCERTAINTY_GENDERED_LIST + [['man', 'woman']]

GENDERED_LIST = UNCERTAINTY_GENDERED_LIST if UNCERTAINTY else SPURIOUS_GENDERED_LIST



#####
data_root = 'data'
plots_root = 'plots'

if UNCERTAINTY:
    inference_dir = "uncertainty_inference_results"
    plots_dir = "uncertainty_plots"
    stats_dir = "uncertainty_processed_results" 

else:
    inference_dir = "spurious_inference_results"
    plots_dir = "spurious_plots"
    stats_dir = "spurious_processed_results"    


def gen_dir_paths(root, path):
    full_path = os.path.join(root, path)
    Path(full_path).mkdir(parents=True, exist_ok=True)
    return full_path

INFERENCE_FULL_PATH = gen_dir_paths(data_root, inference_dir)
PLOTS_FULL_PATH = gen_dir_paths(plots_root, plots_dir)
STATS_FULL_PATH = gen_dir_paths(data_root, stats_dir)


WINOGENDER_SCHEMA_PATH = "winogender_schema_evalulation_set"
SENTENCE_TEMPLATES_FILE = "all_sentences_test.tsv" if TESTING else "all_sentences.tsv"
# winogender_schema_full_path = os.path.join(WINOGENDER_SCHEMA_PATH, sentence_templates)


###################### Add more models here ######################
BERT_LIKE_MODELS_DICT = {
    "bert-base-uncased": "BERT base",
    "bert-large-uncased": "BERT large",
    "roberta-base": "RoBERTa base",
    "roberta-large": "RoBERTa large",
}
BERT_LIKE_MODELS = list(BERT_LIKE_MODELS_DICT.keys())

# %%
 
SPLIT_KEY = INDIE_VAR_NAME.upper()
START_YEAR = 1901 if UNCERTAINTY else 1801  # Adjusted date range for UNCERTAINTY's occupations
STOP_YEAR = 2016 if UNCERTAINTY else 2001  # Adjusted date range for UNCERTAINTY's occupations
DATES = np.linspace(START_YEAR, STOP_YEAR, 30).astype(int).tolist()

# Wikibio place conts
# https://www3.weforum.org/docs/WEF_GGGR_2021.pdf
# Bottom 10 and top 10 Global Gender Gap ranked countries.
PLACE_SPLIT_KEY = "PLACE"
PLACES = [
    "Afghanistan",
    "Yemen",
    "Iraq",
    "Pakistan",
    "Syria",
    "Democratic Republic of Congo",
    "Iran",
    "Mali",
    "Chad",
    "Saudi Arabia",
    "Switzerland",
    "Ireland",
    "Lithuania",
    "Rwanda",
    "Namibia",
    "Sweden",
    "New Zealand",
    "Norway",
    "Finland",
    "Iceland",
]

SUBREDDITS = []  # No subreddits in this papers

INDIE_VARS = DATES if SPLIT_KEY == 'DATE' else PLACES
INDIE_VARS = INDIE_VARS[:10] if TESTING else INDIE_VARS

#### MGT Challenge set #####

VERBS = [
    "she became",
    "she was",
    "she is",
    "she will be",
    "she becomes",
]

LIFESTAGES_PROPER = [
    "a child",
    "an adolescent",
    "an adult",
]

LIFESTAGES_SLANG = [
    "a kid",
    "a teenager",
    "all grown up",
]

MGT_EVAL_SET_PROMPT_VERBS = VERBS[:1] if TESTING else VERBS
MGT_EVAL_SET_LIFESTAGES = LIFESTAGES_PROPER[:1] if TESTING else LIFESTAGES_SLANG + LIFESTAGES_PROPER

