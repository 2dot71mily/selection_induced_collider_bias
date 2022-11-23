# %%
import re
from transformers import pipeline
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
from config import (
    BERT_LIKE_MODELS,
    UNCERTAINTY,
    INFERENCE_FULL_PATH,
    WINOGENDER_SCHEMA_PATH,
    SENTENCE_TEMPLATES_FILE,
    GENDERED_LIST,
    MGT_EVAL_SET_PROMPT_VERBS,
    MGT_EVAL_SET_LIFESTAGES,
    SPLIT_KEY,
    INDIE_VARS,
    INDIE_VAR_NAME,
    DATASET_STYLE,
    TESTING,
)

##### If using finetuned models, otherwise skip #####
MAX_TOKEN_LENGTH = 32

SUBREDDIT_FINETUNED_DICT = {}
# Not using finetuned models in this paper
# {"r_none": "Finetuned\nno Metadata", "subreddit": "Finetuned\nw Subreddit"}
SUBREDDIT_FINETUNED_VARIABLES = list(SUBREDDIT_FINETUNED_DICT.keys())

WIKIBIO_FINETUNED_DICT = {}
# Not using finetuned models in this paper
# {"w_none": "Finetuned\nno Metadata","birth_date": "Finetuned\nw Birthdate",
# "birth_place": "Finetuned\nw Birthplace"}
WIKIBIO_FINETUNED_VARIABLES = list(WIKIBIO_FINETUNED_DICT.keys())

GENDER_OPTIONS = ["female", "male"]
DECIMAL_PLACES = 1
# Picked ints that will pop out visually during debug
NON_GENDERED_TOKEN_ID = 30
LABEL_DICT = {GENDER_OPTIONS[0]: 9, GENDER_OPTIONS[1]: -9}
CLASSES = list(LABEL_DICT.keys())
NON_LOSS_TOKEN_ID = -100
EPS = 1e-5  # to avoid /0 errors
#######################################################
# %%
#### Fire up the models for inference, this may take awhile ####
models_paths = dict()
models = dict()
tokenizers = dict()
base_path = ""  # TODO: Fill in here huggingface user name if using finetuned models

for finetuned_model_name in SUBREDDIT_FINETUNED_VARIABLES:
    # To distinguish between each dataset's none-variants
    saved_model_name = (
        "none" if finetuned_model_name == "r_none" else finetuned_model_name
    )
    models_paths[finetuned_model_name] = (
        base_path
        + f"cond_ft_{saved_model_name}_on_reddit__prcnt_100__test_run_False__roberta-base"
    )
    models[finetuned_model_name] = AutoModelForTokenClassification.from_pretrained(
        models_paths[finetuned_model_name]
    )
    tokenizers[finetuned_model_name] = AutoTokenizer.from_pretrained(
        models_paths[finetuned_model_name]
    )

# wikibio finetuned models:
for finetuned_model_name in WIKIBIO_FINETUNED_VARIABLES:
    # To distinguish between each dataset's none-variants
    saved_model_name = (
        "none" if finetuned_model_name == "w_none" else finetuned_model_name
    )
    models_paths[finetuned_model_name] = (
        base_path + f"cond_ft_{saved_model_name}_on_wiki_bio__prcnt_100__test_run_False"
    )
    models[finetuned_model_name] = AutoModelForTokenClassification.from_pretrained(
        models_paths[finetuned_model_name]
    )
    tokenizers[finetuned_model_name] = AutoTokenizer.from_pretrained(
        models_paths[finetuned_model_name]
    )

# BERT-like models:
for model_name in BERT_LIKE_MODELS:
    hf_model_key = model_name.replace("_", "/")
    models_paths[model_name] = model_name
    models[model_name] = pipeline(
        "fill-mask", model=models_paths[model_name].replace("_", "/")
    )
    tokenizers[model_name] = models[model_name].tokenizer


# %%


def get_gendered_token_ids(tokenizer, gendered_lists):
    # Set up gendered token constants

    male_gendered_tokens = [list[0] for list in gendered_lists]
    female_gendered_tokens = [list[1] for list in gendered_lists]

    male_gendered_token_ids = tokenizer.encode(
        " ".join(male_gendered_tokens), add_special_tokens=False
    )
    female_gendered_token_ids = tokenizer.encode(
        " ".join(female_gendered_tokens), add_special_tokens=False
    )

    # Assert all single token words for ease of processing
    assert len(male_gendered_tokens) == len(male_gendered_token_ids)
    assert len(female_gendered_tokens) == len(female_gendered_token_ids)

    return (
        male_gendered_tokens,
        male_gendered_token_ids,
        female_gendered_tokens,
        female_gendered_token_ids,
    )


# %%


def tokenize_and_append_metadata(
    text, tokenizer, female_gendered_token_ids, male_gendered_token_ids
):
    """Tokenize text and mask/flag 'gendered_tokens_ids' in token_ids and labels."""

    label_list = list(LABEL_DICT.values())
    assert label_list[0] == LABEL_DICT["female"], "LABEL_DICT not an ordered dict"
    label2id = {label: idx for idx, label in enumerate(label_list)}

    tokenized = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_TOKEN_LENGTH,
    )

    # Finding the gender pronouns in the tokens
    token_ids = tokenized["input_ids"]
    female_tags = torch.tensor(
        [
            LABEL_DICT["female"]
            if id in female_gendered_token_ids
            else NON_GENDERED_TOKEN_ID
            for id in token_ids
        ]
    )
    male_tags = torch.tensor(
        [
            LABEL_DICT["male"]
            if id in male_gendered_token_ids
            else NON_GENDERED_TOKEN_ID
            for id in token_ids
        ]
    )

    # Labeling and masking out occurrences of gendered pronouns
    labels = torch.tensor([NON_LOSS_TOKEN_ID] * len(token_ids))
    labels = torch.where(
        female_tags == LABEL_DICT["female"],
        label2id[LABEL_DICT["female"]],
        NON_LOSS_TOKEN_ID,
    )
    labels = torch.where(
        male_tags == LABEL_DICT["male"], label2id[LABEL_DICT["male"]], labels
    )
    masked_token_ids = torch.where(
        female_tags == LABEL_DICT["female"],
        tokenizer.mask_token_id,
        torch.tensor(token_ids),
    )
    masked_token_ids = torch.where(
        male_tags == LABEL_DICT["male"], tokenizer.mask_token_id, masked_token_ids
    )

    tokenized["input_ids"] = masked_token_ids
    tokenized["labels"] = labels

    return tokenized


def prepare_text_for_masking(input_text, mask_token, gendered_tokens, split_key):
    text_w_masks_list = [
        mask_token if word.lower() in gendered_tokens else word
        for word in input_text.split()
    ]
    num_masks = len([m for m in text_w_masks_list if m == mask_token])

    masked_text_portions = " ".join(text_w_masks_list).split(split_key)
    return masked_text_portions, num_masks


def get_tokenized_text_with_metadata(
    tokenizer,
    text_portions,
    indie_var_name,
    indie_vars,
    male_gendered_token_ids,
    female_gendered_token_ids,
    subreddit_prepend_text,
):
    """Construct dict of tokenized texts with each year injected into the text."""

    tokenized_w_metadata = {"ids": [], "atten_mask": [], "toks": [], "labels": []}

    for indie_var in indie_vars:

        target_text = f"{indie_var}".join(text_portions)

        if indie_var_name == "subreddit":
            target_text = f" ".join(text_portions) + subreddit_prepend_text + indie_var
        else:
            target_text = f"{indie_var}".join(text_portions)

        print(f"fine_tuned: {target_text}")
        tokenized_sample = tokenize_and_append_metadata(
            target_text, tokenizer, male_gendered_token_ids, female_gendered_token_ids
        )

        tokenized_w_metadata["ids"].append(tokenized_sample["input_ids"])
        tokenized_w_metadata["atten_mask"].append(
            torch.tensor(tokenized_sample["attention_mask"])
        )
        tokenized_w_metadata["toks"].append(
            tokenizer.convert_ids_to_tokens(tokenized_sample["input_ids"])
        )
        tokenized_w_metadata["labels"].append(tokenized_sample["labels"])

    return tokenized_w_metadata


def get_avg_prob_from_finetuned_outputs(outputs, is_masked, num_preds, gender):
    preds = torch.softmax(outputs[0][0].cpu(), dim=1, dtype=torch.double)
    pronoun_preds = torch.where(is_masked, preds[:, CLASSES.index(gender)], 0.0)
    return round(
        torch.sum(pronoun_preds).item() / (EPS + num_preds) * 100, DECIMAL_PLACES
    )


def get_avg_prob_from_pipeline_outputs(mask_filled_text, gendered_token, num_preds):
    pronoun_preds = [
        sum(
            [
                pronoun["score"]
                if pronoun["token_str"].strip().lower() in gendered_token
                else 0.0
                for pronoun in top_preds
            ]
        )
        for top_preds in mask_filled_text
    ]
    return round(sum(pronoun_preds) / (EPS + num_preds) * 100, DECIMAL_PLACES)


def save_results(results_dict, indie_var_name, filename, dir=INFERENCE_FULL_PATH):
    first_df = results_dict.popitem()[1]  # 2nd element is values
    rest_dfs = [df.drop(indie_var_name, axis=1) for df in results_dict.values()]
    all_dfs = pd.concat([first_df] + rest_dfs, axis=1)
    all_dfs.set_index(indie_var_name)
    file_path = os.path.join(dir, f"{filename}.csv")
    all_dfs.to_csv(file_path)
    print(f"Saved inference results to {file_path}")


# %%
def predict_gender_pronouns(
    model_name,
    input_text,
    normalizing,
    indie_vars=INDIE_VARS,
    indie_var_name=INDIE_VAR_NAME,
    split_key=SPLIT_KEY,
):
    """Run inference on input_text for each model type, returning df and plots of precentage
    of gender pronouns predicted as female and male in each target text.
    """

    model = models[model_name]

    tokenizer = tokenizers[model_name]
    mask_token = tokenizer.mask_token

    (
        male_gendered_tokens,
        male_gendered_token_ids,
        female_gendered_tokens,
        female_gendered_token_ids,
    ) = get_gendered_token_ids(tokenizer, GENDERED_LIST)

    female_dfs = []
    male_dfs = []
    female_dfs.append(pd.DataFrame({indie_var_name: indie_vars}))
    male_dfs.append(pd.DataFrame({indie_var_name: indie_vars}))

    female_pronoun_preds = []
    male_pronoun_preds = []

    masked_text_portions, num_preds = prepare_text_for_masking(
        input_text, mask_token, male_gendered_tokens + female_gendered_tokens, split_key
    )

    if model_name not in BERT_LIKE_MODELS:

        tokenized = get_tokenized_text_with_metadata(
            tokenizer,
            masked_text_portions,
            indie_var_name,
            indie_vars,
            male_gendered_token_ids,
            female_gendered_token_ids,
        )

        toks = tokenized["toks"][1]
        target_text = " ".join(toks[1:-1])  # Removing [CLS] and [SEP]
        initial_is_masked = tokenized["ids"][0] == tokenizer.mask_token_id

        for indie_var_idx in range(len(indie_vars)):
            if indie_var_name == "date":
                is_masked = initial_is_masked  # injected text all same token length
            else:
                is_masked = tokenized["ids"][indie_var_idx] == tokenizer.mask_token_id

            ids = tokenized["ids"][indie_var_idx]
            atten_mask = tokenized["atten_mask"][indie_var_idx]
            labels = tokenized["labels"][indie_var_idx]

            with torch.no_grad():
                outputs = model(ids.unsqueeze(dim=0), atten_mask.unsqueeze(dim=0))

                female_pronoun_preds.append(
                    get_avg_prob_from_finetuned_outputs(
                        outputs, is_masked, num_preds, "female"
                    )
                )
                male_pronoun_preds.append(
                    get_avg_prob_from_finetuned_outputs(
                        outputs, is_masked, num_preds, "male"
                    )
                )

    else:  # BERT-like base model
        for indie_var in indie_vars:

            target_text = str(indie_var).join(masked_text_portions)
            if UNCERTAINTY:
                target_text = target_text.replace("MASK", mask_token)

            print(target_text)

            mask_filled_text = model(target_text)
            # Quick hack as realized return type based on how many MASKs in text.
            if type(mask_filled_text[0]) is not list:
                mask_filled_text = [mask_filled_text]

            female_pronoun_preds.append(
                get_avg_prob_from_pipeline_outputs(
                    mask_filled_text, female_gendered_tokens, num_preds
                )
            )
            male_pronoun_preds.append(
                get_avg_prob_from_pipeline_outputs(
                    mask_filled_text, male_gendered_tokens, num_preds
                )
            )

        if normalizing:
            total_gendered_probs = np.add(female_pronoun_preds, male_pronoun_preds)
            female_pronoun_preds = np.around(
                np.divide(female_pronoun_preds, total_gendered_probs + EPS) * 100,
                decimals=DECIMAL_PLACES,
            )
            male_pronoun_preds = np.around(
                np.divide(male_pronoun_preds, total_gendered_probs + EPS) * 100,
                decimals=DECIMAL_PLACES,
            )

    female_dfs.append(pd.DataFrame({input_text: female_pronoun_preds}))
    male_dfs.append(pd.DataFrame({input_text: male_pronoun_preds}))

    female_results = pd.concat(female_dfs, axis=1)
    male_results = pd.concat(male_dfs, axis=1)

    return (
        target_text,
        female_results,
        male_results,
    )


# %%
def prep_inference(
    special_id="",
    freeform_text="",
    normalizing=False,
):
    # TODO: Pull into config.py. Make consistent, rem under underscores
    test_version = (
        f"test{TESTING}_{special_id}_norm{normalizing}"
        if TESTING
        else f"__{special_id}_norm{normalizing}"
    )

    input_texts = []
    if freeform_text:
        input_texts = [freeform_text]

    else:
        for verb in MGT_EVAL_SET_PROMPT_VERBS:
            for stage in MGT_EVAL_SET_LIFESTAGES:
                input_texts.append(f"{verb} {stage} in {SPLIT_KEY}.")
                input_texts.append(f"In {SPLIT_KEY}, {verb} {stage}.")

    return {
        "input_texts": input_texts,
        "test_version": test_version,
    }


def run_inference(
    model_names,
    indie_var_name,
    special_id,
    freeform_text,
    results_dir=INFERENCE_FULL_PATH,
):

    infer_params = prep_inference(special_id, freeform_text, normalizing)

    input_texts = infer_params["input_texts"]
    test_version = infer_params["test_version"]

    for model_name in model_names:
        all_female_results = {}
        all_male_results = {}

        for input_text in input_texts:
            _target_text, female_results, male_results = predict_gender_pronouns(
                model_name,
                input_text,
                int(normalizing),
            )
            all_female_results[input_text] = female_results
            all_male_results[input_text] = male_results

        filename = f"{DATASET_STYLE}_{model_name}_{indie_var_name}_{test_version}"
        f_filename = f"fp_{filename}"
        m_filename = f"mp_{filename}"

        save_results(
            all_female_results.copy(), indie_var_name, f_filename, dir=results_dir
        )
        save_results(
            all_male_results.copy(), indie_var_name, m_filename, dir=results_dir
        )


if UNCERTAINTY:
    freeform_text = "wino_gender"
    visualization = False
    normalizing = True

    fp = open(os.path.join(WINOGENDER_SCHEMA_PATH, SENTENCE_TEMPLATES_FILE), "r")
    next(fp)  # First line is headers
    for line in fp:
        line = line.strip().split("\t")
        special_id, freeform_text = (
            line[0],
            f"In {INDIE_VAR_NAME.upper()}: {line[1]}",
        )  # Note, always 'DATE' in our plots
        run_inference(
            BERT_LIKE_MODELS,
            INDIE_VAR_NAME,
            special_id,
            freeform_text,
        )
else:
    freeform_text = ""
    visualization = True
    normalizing = False
    special_id = ""

    run_inference(
        BERT_LIKE_MODELS,
        INDIE_VAR_NAME,
        special_id,
        freeform_text,
    )
