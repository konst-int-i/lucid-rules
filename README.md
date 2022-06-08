
# LUCID-RULES: Rule extraction for model explainability

This repository implements the computational experiments for the MPhil in Advanced Computer Science
project of the title "**Global and local interpretability in ML-enabled
clinical decision-making tools**". 

Broadly speaking, we build in multiple **extensions of the ECLAIRE rule extraction 
algorithm** (`lucid/extract_rules/eclaire_extensions.py`) and implement a discrete **column generation 
solver** to build optimal Boolean Rulesets for a given objective function (Hamming Loss) (`lucid/extract_rules/cg_extract.py`).

We also implement an explainability module, which implements multiple local and global explanations of the original model
and allows to combine (aggregate) the output of multiple different explanations, which we use as an extended baseline 
for our computational experiments (`lucid/explainability/explainer.py`).

This repository builds on the [remix](https://github.com/mateoespinosa/remix), an implementation of the ECLAIRE 
[publication](https://arxiv.org/pdf/2111.12628.pdf) and implements multiple extensions to improve the faithfulness, 
stability, and complexity of the resulting rule-based surrogate models. 

Additionally, we implement a Column Generation Rule Extractor as a novel type of surrogate model, the 
implementation of which is heavily based on the [AIX360 implementation](https://github.com/Trusted-AI/AIX360). 

## Setup
In order to install this library, you will need the following requirements
first:
- `python` 3.5 – 3.8
- `pip` 19.0 or later
- `R 4.*` needs to be installed and accessible in your machine. This is required
- as we use R's implementation of C5.0 with a `rpy2` wrapper.

Note that you only require the R dependencies if you want to use the C5.0 algorithm 
as the intermediate rule extractor for the ECLAIRE algorithm. Most of our experiments 
use CART instead, which does not require the R dependencies. 

Once you have installed R, you will also need to have the following packages
installed in R:
- `C50`
- `Cubist`
- `reshape2`
- `plyr`
- `Rcpp`
- `stringr`
- `stringi`
- `magrittr`
- `partykit`
- `Formula`
- `libcoin`
- `mvtnorm`
- `inum`

If you have all of these, then you can install our code as a Python package
using pip as follows:
```python
python setup.py install --user
```


## Recreating Results

For a full results run, you can also run the full experiment:  
```bash
python run_all_experiments.py
```
Where the results will be written to the `experiments_results.yml` file
Note that the configurations of which model and datasets should be run can be updated in `experiments/all_experiments.yml`

To recreate any of the results reported this dissertation, you can call
```bash
python run_experiment.py --config experiment_configs/<dataset_name>/<method_name>_best_config.yaml
```
One can run the tool by manually inputting the dataset information as
command-line arguments or by providing a YAML config containing the experiment
parameterization as the following example:

```yaml
# The directory of our training data. Can be a path relative to the caller or
# an absolute path.
dataset_file: "datasets/XOR/data.csv"

# The name of our dataset. Must be one of the data sets supported by our
# experiment runners.
dataset_name: 'XOR'

# Whether or not we want our experiment runner to overwrite results from
# previous runs found in the given output directory or we want to reuse them
# instead.
# If not provided, or set to null, then we will NOT overwrite any
# previous results and use them as checkpoints to avoid double work in this
# experiment.
# Otherwise, it must be one of
#    - "all"
#    - "data_split"
#    - "fold_split"
#    - "grid_search"
#    - "nn_train"
#    - "rule_extraction"
# to indicate the stage in which we will start rewriting previous results. If
# such a specific stage is provided, then all subsequent stages will be also
# overwritten (following the same order as the list above)
force_rerun: null

# Number of split folds for our training. If not provided, then it will default
# to a single fold.
n_folds: 5

# Our neural network training hyper-parameters
hyperparameters:
    # The batch size we will use for training.
    batch_size: 16
    # Number of epochs to run our model for
    epochs: 150
    # Now many hidden layers we will use and how many activations in each
    layer_units: [64, 32, 16]
    # The activation use in between hidden layers. Can be any valid Keras
    # activation function. If not provided, it defaults to "tanh"
    activation: "tanh"
    # The last activation used in our model. Used to define the type of
    # categorical loss we will use. If not provided, it defaults to the
    # corresponding last activation for the given loss function.
    last_activation: "softmax"
    # The type of loss we will use. Must be one of
    # ["softmax_xentr", "sigmoid_xentr"]. If not provided, we will use the
    # given last layer's activation to obtain the corresponding loss if it was
    # provided. Otherwise, we will default to softmax xentropy.
    loss_function: "softmax_xentr"
    # The learning rate we will use for our Adam optimizer. If not provided,
    # then it will be defaulted to 0.001
    learning_rate: 0.001
    # Dropout rate to use in layer in between last hidden layer and last layer
    # If 0, then no dropout is done. This is the probability that a given
    # activation will be dropped.
    dropout_rate: 0
    # Frequency of skip connections in the form of additions. If zero, then
    # no skip connections are made
    skip_freq: 0


    ###########################################
    # Early Stopping Parameters
    ###########################################

    # Early stopping parameters. Only valid if patience is greater than 0.
    early_stopping_params:
        validation_percent: 0.1
        patience: 0
        monitor: "loss"
        min_delta: 0.001
        mode: 'min'

    # Optimizer parameters
    optimizer_params:
        decay_rate: 1
        decay_steps: 1

# How many subprocesses to use to evaluate our ruleset in the testing data
evaluate_num_workers: 6

# The rule extractor we will use. If not provided, it defaults to ECLAIRE.
# Must be one of [
#   "REM-D",
#   "ECLAIRE", (or equivalently "eREM-D")
#   "cREM-D",
#   "Pedagogical",
#   "Clause-REM-D",
#   "DeepRED",
#   "REM-T",
#   "sREM-D"
# ]
rule_extractor: "ECLAIRE"

# And any parameters we want to provide to the extractor for further
# tuning/experimentation. This is dependent on the used extractor
extractor_params:
    # An integer indicating how many decimals should we truncate our thresholds
    # to. If null, then no truncation will happen.
    # For original REM-D: set to null
    threshold_decimals: null

    # The winnow parameter to use for C5 for intermediate rulesets.
    # Must be a boolean.
    # For original REM-D: set to True
    winnow_intermediate: True

    # The winnow parameter to use for C5 for the last ruleset generation (which
    # depends on the actual features)
    # Must be a boolean.
    # For original REM-D: set to True
    winnow_features: True

    # The minimum number of cases for a split in C5. Must be a positive integer
    min_cases: 2

    # What is the maximum number of training samples to use when building trees
    # If a float, then this is seen as a fraction of the dataset
    # If 0 or null, then no restriction is applied.
    # For original REM-D: set to null
    max_number_of_samples: null
    
# Where are we dumping our results. If not provided, it will default to the same
# directory as the one containing the dataset.
output_dir: "experiment_results"

# Parameters to be used during our grid search
grid_search_params:
    # Whether or not to perform grid-search
    enable: False
    # The metric we will optimize over during our grid-search. Must be one of
    # ["accuracy", "auc"]
    metric_name: "accuracy"
    # Batch sizes to be used during training
    batch_sizes: [16, 32]
    # Training epochs to use for our DNNs
    epochs: [50, 100, 150]
    # Learning rates to try in our optimizer
    learning_rates: [0.001, 0.0001]
    # The sizes to try for each hidden layer
    layer_sizes: [[128, 64, 32], [64, 32]]
    # Activations to use between hidden layers. Must be valid Keras activations
    activations: ["tanh", "elu"]
    # The amount of dropout to use between hidden layers and the last layer
    dropout_rates: [0, 0.2]
    # Finally, the loss function to use for training
    loss_functions: ["softmax_xentr", "sigmoid_xentr"]

```