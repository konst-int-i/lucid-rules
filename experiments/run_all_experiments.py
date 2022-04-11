"""
File which serves as wrapper to run full sets of experiments as well as multiple Eclaire variations
that test for faithfulness, and stability
"""

import os
from remix.utils.config import Config
from sklearn.model_selection import ParameterGrid
from run_experiment import run_experiment
import logging
import pandas as pd
from datetime import datetime
import click
from typing import *
from experiments.model_training.train import load_model
from experiments.experiment_runners.cross_validation import _deserialize_rules
from remix.explainability.rules import PipelineExplainer
from pathlib import Path
import tensorflow as tf

@click.command()
# @click.option(
#     "--config",
#     "-c",
#     default=None,
#     help="initial configuration YAML file for our experiment's setup.",
#     metavar="file.yaml"
# )
@click.option(
    '--n_folds',
    '-n',
    default=None,
    help='how many folds to use for our data partitioning.',
    metavar='N',
    type=int,
)
@click.option(
    '--dataset_name',
    default=None,
    help="name of the dataset to be used for training.",
    metavar="name",
)
@click.option(
    '--dataset_file',
    default=None,
    help="comma-separated-valued file containing our training data.",
    metavar="data.cvs",
)
@click.option(
    '--rule_extractor',
    default=None,
    help=(
        "name of the extraction algorithm to be used to generate our "
        "rule set."),
    metavar="name",
    type=click.Choice([
        "CART",
        "Clause-REM-D",
        "cREM-D",
        "DeepRED",
        "ECLAIRE",
        "eREM-D",
        "random_forest",
        "RandomForest",
        "REM-T",
        "sREM-D",
        'Pedagogical',
        'REM-D',
    ], case_sensitive=False)
)
@click.option(
    '--grid_search',
    # action="store_true",
    default=None,
    help=(
            "whether we want to do a grid search over our model's "
            "hyperparameters. If the results of a previous grid search are "
            "found in the provided output directory, then we will use those "
            "rather than starting a grid search from scratch. This means that "
            "turning this flag on will overwrite any hyperparameters you "
            "provide as part of your configuration (if any)."
    ),
)
# @click.option(
#     '--output_dir',
#     '-o',
#     default=None,
#     help=(
#             "directory where we will dump our experiment's results. If not "
#             "given, then we will use the same directory as our dataset."
#     ),
#     metavar="path",
# )
@click.option(
    '--randomize',
    '-r',
    default=False,
    # action="store_true",
    help=(
            "If set, then the random seeds used in our execution will not "
            "be fixed and the experiment will be randomized. By default, "
            "otherwise, all experiments are run with the same seed for "
            "reproducibility."
    ),
)
# @click.option(
#     '--force_rerun',
#     '-f',
#     type=click.Choice((["all"] + EXPERIMENT_STAGES), case_sensitive=False),
#     default=None,
#     help=(
#             "If set and we are given as output directory the directory of a "
#             "previous run, then we will overwrite any previous work starting "
#             "from the provided stage (and all subsequent stages of the "
#             "experiment) and redo all computations. Otherwise, we will "
#             "attempt to use as much as we can from the previous run."
#     ),
# )
@click.option(
    '--profile',
    help=(
            "prints out profiling statistics of the rule-extraction in terms "
            "of low-level calls used for the extraction method."
    ),
    # action="store_true",
    default=False,
)
@click.option(
    "-d",
    "--debug",
    # action="store_true",
    default=False,
    help="starts debug mode in our program.",
)
@click.option(
    '-p',
    '--param',
    metavar=('param_name=value'),
    nargs=2,
    help=(
            'Allows the passing of a config param that will overwrite '
            'anything passed as part of the config file itself.'
    ),
    default=["param", "value"],
)
def main(**kwargs):
    config = Config("all_experiments.yml").read()

    # set up logging
    logging.getLogger().setLevel(
        logging.INFO
    )
    logging.basicConfig(
        format='[%(levelname)s] %(message)s'
    )

    grid = ParameterGrid({
        "dataset": config.datasets,
        "method": config.extraction_methods
    })

    new_run_dir = Path(f"experiment_results/{datetime.now().strftime('%m_%d_%H_%M_%S')}")

    for experiment in grid:
        logging.info(f"Running experiment - Dataset: {experiment['dataset']} - Method: {experiment['method']}")

        #########################################################
        # STEP 1 - TRAIN DNN AND RUN RULE EXTRACTION (BASELINE) #
        #########################################################
        if "train_and_extract" in config.pipeline:

            # create new run directory when training and extraction is rerun
            output_dir = new_run_dir.joinpath(f"{experiment['dataset']}_{experiment['method']}")
            if "force_rerun" in config.keys():
                if len(config.force_rerun) == 5:
                    rerun = "all"
                else:
                    rerun = config.force_rerun
            else:
                rerun = None
            logging.info(f"Rerunning steps: {rerun}")
            run_experiment(config=f"experiment_configs/{experiment['dataset']}/{experiment['method']}_best_config.yaml",
                           force_rerun=rerun, output_dir=output_dir, **kwargs)
        else:
            # if we aren't rerunning the training and extraction, load latest from experiment outputs
            exp_dir = sorted(os.listdir("experiment_results/"))[-1]
            run_dir = Path("experiment_results").joinpath(exp_dir)
            output_dir = run_dir.joinpath(f"{experiment['dataset']}_{experiment['method']}")
            assert os.path.exists(output_dir), f"Directory {output_dir} does not exist - check that you " \
                                               f"ran the model and extraction method combination"

        models, rulesets = _load_model_and_ruleset(output_dir)
        x_train_folds, y_train_folds, x_test_folds, y_test_folds = _load_data_splits(dataset=experiment['dataset'], output_dir=output_dir)
        # read in experiment logging
        result_path = output_dir.joinpath(f"cross_validation/5_folds/rule_extraction/"
                                                    f"{experiment['method'].upper()}/results.csv")
        result_df = pd.read_csv(result_path)

        ######################################
        # STEP 2 - RUN EXPLAINABILITY MODULE #
        ######################################
        if "explain" in config.pipeline:
            n_folds = len(models)
            for fold, (model, ruleset) in enumerate(zip(models, rulesets)):

                # print(model, ruleset[0])
                explainer = PipelineExplainer(config, model, ruleset[0])
                rule_ranking = explainer.rule_feature_ranking(top_n=20)
                shap_ranking = explainer.shap_feat_ranking(top_n=20, x_train=x_train_folds[fold],
                                                           x_test=x_test_folds[fold])
                # plot both rankings
                plot_dir = result_path.parent.joinpath("figures")
                if not os.path.exists(plot_dir):
                    os.mkdir(plot_dir)
                plot_file = plot_dir.joinpath(f"importances_fold_{fold+1}.png")
                explainer.plot_rankings(rule_ranking=rule_ranking, original_ranking=shap_ranking, save_path=plot_file)
                tau, p_value = explainer.ranking_similarity(rule_ranking, shap_ranking)
                # log in results
                result_df.loc[result_df["fold"] == fold + 1, "tau"] = tau
                result_df.loc[result_df["fold"] == fold + 1, "ranking_p_value"] = p_value
        result_df.to_csv(result_path)

    return None


def _load_data_splits(dataset: str, output_dir: Path):
    split_path= output_dir.joinpath(f"cross_validation/5_folds/data_split_indices.txt")
    with open(split_path, "r") as f:
        indices = f.read().split("\n")

    data_mapping = {"xor": "XOR", "metabric-er": "MB-GE-ER", "metabric-hist": "MB-1004-GE-2Hist"}

    df = pd.read_csv(f"datasets/{data_mapping[dataset]}/data.csv")
    # note: last column is the target
    target_col = df.columns[-1]
    y = df[target_col]
    x = df.drop([target_col], axis=1)
    train_indices = [e.split(" ")[1:] for e in indices[0::2]]
    test_indices = [e.split(" ")[1:] for e in indices[1::2]]

    x_train_folds  = []
    y_train_folds = []
    x_test_folds = []
    y_test_folds = []

    for train_index, test_index in zip(train_indices, test_indices):
        x_train_folds.append(x.iloc[train_index])
        y_train_folds.append(y.iloc[train_index])
        x_test_folds.append(x.iloc[test_index])
        y_test_folds.append(y.iloc[test_index])
    return x_train_folds, y_train_folds, x_test_folds, y_test_folds




def _load_model_and_ruleset(output_dir: Path) -> Tuple[List]:
    """
    Takes in the output directory from a completed experiment run (single dataset, single method)
    and loads in the DNN models and extracted rule sets for all folds
    Args:
        output_dir (Path): relative path to output directory (typically ``experiments/experiment_results/...``)
    Returns:
        Tuple[List]: List of tf models, and list of corresponding rulesets

    """
    models = []
    rulesets = []
    # get the number of folds based on number of models
    n_folds = int(sorted(os.listdir(output_dir.joinpath("cross_validation")))[0].split("_")[0])
    model_dir = output_dir.joinpath(f"cross_validation/{n_folds}_folds/trained_models")
    rule_dir = output_dir.joinpath(f"cross_validation/{n_folds}_folds/rule_extraction")
    logging.info(f"Reading in models and rules from {model_dir.parent}")

    # load model and rule for each fold (1-indexed)
    for fold in range(1, n_folds+1):
        model_path = model_dir.joinpath(f"fold_{1}_model.h5")
        rule_path = rule_dir.joinpath(f"ECLAIRE/rules_extracted/fold_{fold}.rules")
        rulesets.append(_deserialize_rules(rule_path))
        models.append(load_model(model_path))

    return models, rulesets



def _check_main_config(config):
     valid_methods = ["eclaire", "deep_red", "pedc5.0", "rem_d", "c5.0"]
     valid_datasets = ["xor", "metabric-er", "metabric_hist"]

     for dataset in config.datasets:
         assert dataset in valid_datasets, f"Invalid dataset in 'all_experiments.yml' - valid values: {valid_datasets}"

    # for method in config.extraction_methods:
    #     assert method in valid_methods







if __name__ == "__main__":
    main()