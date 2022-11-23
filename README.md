# Reproducing Selection Induced Collider Bias in LLMs: A Gender Pronoun Uncertainty Case Study

## Interact with open source demos!

Note: If you would rather checkout running and flexible demos, to reproduce the methods and measurements in this paper, please see below, otherwise, skip to the `setup`.:
- Spurious Correlations Open Source Hugging Face Space: https://huggingface.co/spaces/paper5186/spurious.
- Uncertainty Measurement Open Source Hugging Face Space: https://huggingface.co/spaces/paper5186/uncertainty.
- More General Setting Toy SCM: https://tinyurl.com/2ub4xyjs.

# Setup
```
git clone https://github.com/anon-anon-anony/sicb_paper.git
cd sicb_paper
python3 -m venv ~/venv_sicb
source ~/venv_sicb/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```


# Reproducing plots with existing data
## Spurious plots
In `config.py`:
```
UNCERTAINTY = True # Set to False if replicating the Spurious results
```
Then run in terminal:
`python spurious_plotting.py`

You should see a printout of slope and r coefficients and file save updates print out before successful completion, e.g.:

```
% python spurious_plotting.py  
                                                    slope      r
fp_wikibio_bert-base-uncased_place___normFalse.csv  0.280  0.467
mp_wikibio_bert-base-uncased_place___normFalse.csv -0.398 -0.579
fp_wikibio_bert-large-uncased_place___normFalse...  0.381  0.527
mp_wikibio_bert-large-uncased_place___normFalse... -0.523 -0.697
fp_wikibio_roberta-base_place___normFalse.csv       0.132  0.359
mp_wikibio_roberta-base_place___normFalse.csv      -0.174 -0.578
fp_wikibio_roberta-large_place___normFalse.csv      0.268  0.676
mp_wikibio_roberta-large_place___normFalse.csv     -0.266 -0.618
Saved plots to plots/spurious_plots/final_fig_wikibio_place___normFalse
Saved plots to plots/spurious_plots/delta_fit_stats_plotfp_wikibio_roberta-large_place___normFalse
```

## Uncertainty plots
In `config.py`:
```
UNCERTAINTY = False # Set to False if replicating the Spurious results
TESTING = False # Set to True if testing on subsets of the challenge sets
```

Then run in terminal:
`python inference.py`

You should see a printout of some uncertainty metrics, TP and TN rates, and file save updates print out before successful completion, e.g.:

```
% python uncertainty_plotting.py 
Plot saved to plots/uncertainty_plots/female_Doctor_roberta-large
....
... 
uncertainty: roberta-large doctor man f_1
0.09999999999999998
uncertainty: roberta-large doctor woman f_1
0.5
uncertainty: roberta-large doctor someone f_1
16.35
uncertainty: roberta-large doctor other f_1
9.25
Plot saved to plots/uncertainty_plots/bls__RoBERTa_large_Uncertainty_in_Pronoun_Prediction_for_the_Participant_in_Winogender-like_Texts
For roberta-large, true postive: 0.9388888888888889, true negative: 0.9166666666666666, and true postive gender co-occuring: 0.9 
```

# Reproducing the data used in the plots
First test the setup as shown below.


## Testing your setup

Note: The LLM weights will be downloaded / cached from Hugging Face. Throughout, it *is* expected that `Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertForMaskedLM: ['cls.seq_relationship.bias', 'cls.seq_relationship.weight']`


### Spurious data
In `config.py`:
```
UNCERTAINTY = False # Set to False if replicating the Spurious results
TESTING = True # Set to True if testing on subsets of the challenge sets
INDIE_VAR_FOR_SPURIOUS = 1  # Select '0' to use "date", '1' to use "place" for Spurious
```
Then run in terminal:
`python inference.py`

You should see dozens of test sentences and file save updates print out before successful completion, e.g.:
```
% python inference.py  
...
In Chad, <mask> became a child.
In Saudi Arabia, <mask> became a child.
Saved inference results to data/spurious_inference_results/fp_wikibio_roberta-base_place_testTrue__normFalse.csv
Saved inference results to data/spurious_inference_results/mp_wikibio_roberta-base_place_testTrue__normFalse.csv
<mask> became a child in Afghanistan.
<mask> became a child in Yemen.
...
```

### Uncertainty data
In `config.py`:
```
UNCERTAINTY = True # Set to False if replicating the Spurious results
TESTING = True # Set to True if testing on subsets of the challenge sets
```
Then run in terminal:
`python inference.py`

You should see dozens of test sentences and file save updates print out before successful completion, e.g.:
```
% python inference.py  
...
In 1932: The doctor told the woman that [MASK] would be on vacation next week.
In 1936: The doctor told the woman that [MASK] would be on vacation next week.
Saved inference results to data/uncertainty_inference_results/fp_wikibio_bert-large-uncased_date_testTrue_doctor_woman_0_normTrue.csv
Saved inference results to data/uncertainty_inference_results/mp_wikibio_bert-large-uncased_date_testTrue_doctor_woman_0_normTrue.csv
In 1901: The doctor told the woman that <mask> would be on vacation next week.
In 1904: The doctor told the woman that <mask> would be on vacation next week
...
```


## Reproducing all the data
### Spurious data (can take ~30 minutes)
In `config.py`:
```
UNCERTAINTY = False # Set to False if replicating the Spurious results
TESTING = False # Set to True if testing on subsets of the challenge sets
INDIE_VAR_FOR_SPURIOUS = 1  # Select '0' to use "date", '1' to use "place" for Spurious
```

Then run in terminal:
`python inference.py`

Similar to before but you should see **thousands** of test sentences and file save updates print out before successful completion, e.g.:
```
% python inference.py  
...
In Saudi Arabia, <mask> became a child.
Saved inference results to data/spurious_inference_results/fp_wikibio_roberta-base_place___normFalse.csv
Saved inference results to data/spurious_inference_results/mp_wikibio_roberta-base_place___normFalse.csv
<mask> became a child in Afghanistan.
...
```


### Uncertainty data
In `config.py`:
```
UNCERTAINTY = True # Set to False if replicating the Spurious results
TESTING = False # Set to True if testing on subsets of the challenge sets
```
Then run in terminal:
`python inference.py`

Similar to before but you should see **thousands** of test sentences and file save updates print out before successful completion, e.g.:
```
% python inference.py  
...
In 1936: The doctor told the woman that [MASK] would be on vacation next week.
Saved inference results to data/uncertainty_inference_results/fp_wikibio_bert-large-uncased_date_testTrue_doctor_woman_0_normTrue.csv
Saved inference results to data/uncertainty_inference_results/mp_wikibio_bert-large-uncased_date_testTrue_doctor_woman_0_normTrue.csv
In 1901: The doctor told the woman that <mask> would be on vacation next week.
...
```

