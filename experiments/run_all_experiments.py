"""
File which serves as wrapper to run full sets of experiments as well as multiple Eclaire variations
that test for faithfulness, and stability
"""

import os
from lucid.utils.config import Config
from sklearn.model_selection import ParameterGrid
from run_experiment import run_experiment
import logging
import pandas as pd
from datetime import datetime
import click
from typing import *
from experiments.model_training.train import load_model
from experiments.experiment_runners.cross_validation import _deserialize_rules
from lucid.explainability.explainer import PipelineExplainer
from pathlib import Path
from lucid.rules.ruleset import Ruleset
from lucid.rules.column_generation import BooleanRuleCG
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

    run_dir = Path(f"experiment_results/{datetime.now().strftime('%m_%d_%H_%M_%S')}")

    for experiment in grid:
        logging.info(f"Running experiment - Dataset: {experiment['dataset']} - Method: {experiment['method']}")

        #########################################################
        # STEP 1 - TRAIN DNN AND RUN RULE EXTRACTION (BASELINE) #
        #########################################################
        if "train_and_extract" in config.pipeline:

            # create new run directory when training and extraction is rerun
            output_dir = run_dir.joinpath(f"{experiment['dataset']}_{experiment['method']}")
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
            # exp_dir = "05_26_16_14_08_c50_agg_vs_shap"
            run_dir = Path("experiment_results").joinpath(exp_dir)
            output_dir = run_dir.joinpath(f"{experiment['dataset']}_{experiment['method']}")
            assert os.path.exists(output_dir), f"Directory {output_dir} does not exist - check that you " \
                                               f"ran the model and extraction method combination"

        models, rulesets = _load_model_and_ruleset(output_dir)
        x_train_folds, y_train_folds, x_test_folds, y_test_folds = _load_data_splits(dataset=experiment['dataset'], output_dir=output_dir)
        # read in experiment logging
        results_subdir = os.listdir(output_dir.joinpath("cross_validation/n_folds/rule_extraction"))[0]
        result_path = output_dir.joinpath(f"cross_validation/n_folds/rule_extraction/"
                                                    f"{results_subdir}/results.csv")
        result_df = pd.read_csv(result_path)

        ######################################
        # STEP 2 - RUN EXPLAINABILITY MODULE #
        ######################################
        if "explain" in config.pipeline:
            for fold, (model, ruleset) in enumerate(zip(models, rulesets)):

                if isinstance(ruleset[0], Ruleset):
                    explainer = PipelineExplainer(config, model, ruleset[0], experiment['method'])
                elif isinstance(ruleset[0], BooleanRuleCG):
                    explainer = PipelineExplainer(config, model, ruleset[0].ruleset, experiment['method'])
                rule_ranking = explainer.rule_feature_ranking(top_n=None)
                if experiment['method'] in ["eclaire", "eclaire-cart"]:
                    original_ranking = explainer.original_feat_ranking(top_n=None, x_train=x_train_folds[fold],
                                                               x_test=x_test_folds[fold], method="ig")
                else: # gold_standard otherwise
                    original_ranking = explainer.gold_standard_ranking(top_n=None, x_train=x_train_folds[fold],
                                                                        x_test=x_test_folds[fold])
                # plot both rankings
                plot_dir = result_path.parent.joinpath("figures")
                if not os.path.exists(plot_dir):
                    os.mkdir(plot_dir)
                plot_file = plot_dir.joinpath(f"importances_fold_{fold+1}.png")
                explainer.plot_rankings(rule_ranking=rule_ranking, original_ranking=original_ranking,
                                        save_path=plot_file, top_n=20)
                tau, p_value = explainer.rank_correlation(rule_ranking, original_ranking, method="tau")
                weighted_tau, _ = explainer.rank_correlation(rule_ranking, original_ranking, method="weighted_tau")
                # rho, p_value = explainer.rank_correlation(rule_ranking, original_ranking, method="rho")
                rbo_score = explainer.rank_order(rule_ranking, original_ranking, method="rbo")
                # log in results
                result_df.loc[result_df["fold"] == fold + 1, "tau"] = tau
                result_df.loc[result_df["fold"] == fold + 1, "weighted_tau"] = weighted_tau
                result_df.loc[result_df["fold"] == fold + 1, "rbo_score"] = rbo_score
        result_df.to_csv(result_path)

    summary_df = _write_joint_results(run_folder=Path(run_dir))
    agg_df = aggregate_experiments(summary_df=summary_df, run_folder=Path(run_dir))

    return None


def _write_joint_results(run_folder: Path) -> pd.DataFrame:
    """
    Method to write summary table from all experiments
    Args:
        run_folder:

    Returns:
        None: writes the table as table ``results_summary.csv`` into results folder
    """
    exp_runs = [exp for exp in os.listdir(run_folder) if exp not in ["results_summary.csv", "results_aggregated.csv", "results_agg.csv"]]

    # iterate through all runs
    dfs = []
    for single_exp in exp_runs:
        subdir = os.listdir(f"{run_folder}/{single_exp}/cross_validation/n_folds/rule_extraction")[0]
        result_path = f"{run_folder}/{single_exp}/cross_validation/n_folds/rule_extraction/{subdir}/results.csv"
        exp_df = pd.read_csv(result_path)
        exp_df  = exp_df.loc[:, ~exp_df.columns.str.contains('^Unnamed')]

        exp_df.insert(0, "dataset", single_exp.split("_")[0])
        exp_df.insert(0, "model", single_exp.split("_")[1])
        exp_df["fold"] = exp_df["fold"].astype("int")

        dfs.append(exp_df)
    summary_df = pd.concat(dfs, sort=False)
    # drop unnamed columns:

    summary_df.to_csv(f"{run_folder}/results_summary.csv")
    return summary_df

def aggregate_experiments(summary_df: pd.DataFrame, run_folder: Path) -> None:
    """
    Calculates relevant aggregate metrics for experiment results across seeds
    """
    # summary_df["re_n_rules_per_class"] = summary_df["re_n_rules_per_class"].astype("str")
    try:
        summary_df.loc[:, "total_rules"] = summary_df["re_n_rules_per_class"].apply(lambda x: x.split(",")[0][1:]).astype("int")
    except:
        summary_df.loc[:, "total_rules"] = 0

    result_agg = summary_df.groupby(["model", "dataset"], as_index=False).agg({
        "nn_loss": ["mean", "std"],
        "nn_auc": ["mean", "std"],
        "majority_class": ["mean"],
        "re_time (sec)": ["mean", "std"],
        "re_memory (MB)": ["mean", "std"],
        "re_acc": ["mean", "std"],
        "re_fid": ["mean", "std"],
        "re_auc": ["mean", "std"],
        "total_rules": ["mean", "std"],
        "tau": ["mean", "std"],
        "weighted_tau": ["mean", "std"],
        "rbo_score": ["mean", "std"]
    })

    result_agg.to_csv(f"{run_folder}/results_agg.csv")
    return result_agg

def _load_data_splits(dataset: str, output_dir: Path):
    split_path= output_dir.joinpath(f"cross_validation/n_folds/data_split_indices.txt")
    with open(split_path, "r") as f:
        indices = f.read().split("\n")

    data_mapping = {"xor": "XOR", "metabric-er": "MB-GE-ER", "metabric-hist": "MB-1004-GE-2Hist",
                    "synthetic": "SYNTHETIC", "magic": "MAGIC", "diabetes": "DIABETES"}

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
    # n_folds = int(sorted(os.listdir(output_dir.joinpath("cross_validation")))[0].split("_")[0])
    model_dir = output_dir.joinpath(f"cross_validation/n_folds/trained_models")
    rule_dir = output_dir.joinpath(f"cross_validation/n_folds/rule_extraction")
    logging.info(f"Reading in models and rules from {model_dir.parent}")

    # load model and rule for each fold (1-indexed)
    for fold in range(1, len(os.listdir(model_dir)) + 1):
        model_path = model_dir.joinpath(f"fold_{1}_model.h5")
        method_subdir = os.listdir(rule_dir)[0]
        rule_path = rule_dir.joinpath(f"{method_subdir}/rules_extracted/fold_{fold}.rules")
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