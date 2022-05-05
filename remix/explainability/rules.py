import logging

import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
import shap
from scipy import stats
from box import Box
from collections import Counter



class PipelineExplainer(object):
    """
    Object containing multiple explainability utils for the ruleset and the
    original model
    """
    def __init__(self, config, dnn_model, ruleset):

        self.model = dnn_model
        self.ruleset = ruleset


    def rule_feature_ranking(self, as_pct:bool=True, top_n: int = None) -> pd.Series:
        """
        Function which calculates the feature importance ranking based on the frequency
        of features used in the antecedents of extracted rules

        Args:
            as_pct(bool): returns the ranking as relative % of total importance. Doing this
                helps making the method

        Returns:
            pd.Series: Series with feature names in index

        """
        logging.info(f"Getting rule feature ranking")
        decision_vars = [elem.variable for elem in list(self.ruleset.get_terms_from_rule_premises())]
        var_count = Counter(decision_vars)
        rule_count = pd.Series(data=list(var_count.values()),
                               index=list(var_count.keys())).sort_values(ascending=False)

        if as_pct:
            rule_count /= rule_count.sum()

        if top_n is not None:
            rule_count = rule_count[:top_n]

        rule_count.name = "rules"
        return rule_count


    def shap_feat_ranking(self, x_train, x_test, as_pct: bool=True, top_n: int = None,
                          ) -> pd.Series:
        """
        Returns feature importance ranking of the original model based on shapley values
        Args:
            as_pct:
            top_n:

        Returns:
        """
        logging.info("Fitting Shap Deep Explainer")
        # e = shap.DeepExplainer(self.model, x_train[:1000].to_numpy())
        e = shap.DeepExplainer(self.model, x_train.to_numpy())
        shap_values = e.shap_values(x_test.to_numpy())
        shap_rank = pd.Series(data=np.abs(shap_values[0]).sum(axis=0), index=x_test.columns)
        shap_rank = shap_rank.sort_values(ascending=False)
        if as_pct:
            shap_rank /= shap_rank.sum()

        shap_rank.name = "original"
        return shap_rank

    def plot_rankings(self, rule_ranking: pd.Series, original_ranking: pd.Series, save_path: str, plot_top: int = 20,
                      ):
        """
        Given two feature rankings, we write a bar plot of both to the output
        Args:
            rule_ranking:
            original_ranking:
            plot_top:

        Returns:

        """
        joint_df = rule_ranking.to_frame().join(original_ranking)
        joint_df = joint_df.reset_index().rename({"index": "Feature"}, axis=1)
        plot_df = pd.melt(joint_df, value_vars=["original", "rules"], id_vars=["Feature"])
        if plot_top is not None:
            plot_df = plot_df.iloc[:, :plot_top]
        plot_df = plot_df.rename({"value": "% Rank", "variable": "Extraction"}, axis=1)
        fig, ax = plt.subplots()
        sns.barplot(data=plot_df, x="% Rank", y="Feature", hue="Extraction", ax=ax)
        fold = int(str(save_path)[-5])
        plt.title(f"Feature Importances - Fold {fold}")
        plt.savefig(save_path)
        # plt.show()

    def aggregate_rankings(self):
        """

        Returns:

        """
        pass

    def euclidean_importance(self, a, b):
        """
        Calculates the squared L2 norm of two ranked lists. Takes in two feature importance
        rankings and returns joint importance metric
        Args:
            a:
            b:

        Returns:
            Squared L2 norm of rankings

        """
        return np.square(np.linalg.norm((a-b), ord=2))


    def ranking_similarity(self, rule_ranking: pd.Series, original_ranking: pd.Series) -> float:
        """
        Given two pandas series of the same size, this method calculates different ranking
        similarity metrics (default: Kendall's Tau)
        Returns:

        """
        # merge such that indeces are aligned
        merge_rank = rule_ranking.to_frame().join(original_ranking)

        tau, p_value = stats.kendalltau(merge_rank["rules"], merge_rank["original"])
        logging.info(f"Kendall's tau: {tau}; p-value: {p_value}")

        assert np.isnan(tau) == False, "Invalid tau value - check the ranking series"

        return tau, p_value
