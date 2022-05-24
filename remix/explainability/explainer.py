import logging

import seaborn as sns
import matplotlib.pyplot as plt
import math
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

        term_confidence = self.ruleset.get_terms_with_conf_from_rule_premises()
        terms = [elem.variable for elem in list(self.ruleset.get_terms_from_rule_premises())]
        confidences = [term[1] for term in term_confidence.items()]
        confidence_df = pd.DataFrame(data={"term": terms, "conf": confidences})
        confidence = confidence_df.groupby("term", as_index=True).median()
        term_count = pd.Series(data=list(Counter(terms).values()), index=list(Counter(terms).keys()), name="terms")
        score_df = pd.merge(confidence, term_count, left_index=True, right_index=True)
        features = list(set(terms))
        clause_counter = {} # count clauses that contain a term for each feature
        all_clauses = flatten([list(rule.premise) for rule in self.ruleset.rules])
        nested_clauses = [[term.variable for term in clause.terms] for clause in all_clauses]
        # for feat in features:
        #     clause_count = 0
        #     # iterate through all terms in clauses
        #     for clause in nested_clauses:
        #         if feat in clause:
        #             clause_count += 1
        #     clause_counter[feat] = clause_count
        # clause_df = pd.Series(data=list(clause_counter.values()), index=list(clause_counter.keys()), name="clauses")
        #
        # score_df = pd.merge(score_df, clause_df, left_index=True, right_index=True)

        score_df["term_ratio"] = score_df.terms / score_df.terms.sum()
        # score_df["clause_ratio"] = score_df.clauses / len(all_clauses)
        # # weight the term ratio by confidence
        score_df["term_ratio_weighted"] = score_df["term_ratio"] * score_df["conf"]
        score_df["feat_importance"] = score_df["term_ratio"] * score_df["conf"]
        # # calculate as pct
        # score_df["term_ratio_pct"] = score_df["term_ratio"] / score_df["term_ratio"].sum()
        score_df["feat_importance_pct"] = score_df["feat_importance"] /score_df["feat_importance"].sum()
        score_df.sort_values(by="feat_importance", ascending=False, inplace=True)

        if top_n is not None:
            score_df = score_df[:top_n]
        if as_pct:
            importance = score_df["feat_importance_pct"]
        else:
            importance = score_df["feat_importance"]
        importance.name = "rules"
        return importance

        # get total number of terms in ruleset


        # decision_vars = [elem.variable for elem idn list(self.ruleset.get_terms_from_rule_premises())]
        # var_count = Counter(decision_vars)
        # rule_count = pd.Series(data=list(var_count.values()),
        #                        index=list(var_count.keys())).sort_values(ascending=False)
        #
        # if as_pct:
        #     rule_count /= rule_count.sum()
        #
        # if top_n is not None:
        #     rule_count = rule_count[:top_n]
        #
        # rule_count.name = "rules"
        # return rule_count


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
        plt.show()

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


    def rank_correlation(self, rule_ranking: pd.Series, original_ranking: pd.Series, method="tau") -> float:
        """
        Given two pandas series of the same size, this method calculates different ranking
        similarity metrics (default: Kendall's Tau)
        Returns:

        """
        # merge such that indeces are aligned
        valid_methods = ["tau", "rho"]
        assert method in valid_methods, f"Invalid method, select one of {valid_methods}"

        merge_rank = rule_ranking.to_frame().join(original_ranking)

        if method == "tau":
            corr, p_value = stats.kendalltau(merge_rank["rules"], merge_rank["original"])
            logging.info(f"Kendall's tau: {corr}; p-value: {p_value}")
        elif method == "rho":
            corr, p_value = stats.spearmanr(merge_rank["rules"], merge_rank["original"])
            logging.info(f"Spearman Rho: {corr}; p-value: {p_value}")
        corr = np.round(corr, 5)
        p_value = np.round(p_value, 5)

        assert np.isnan(corr) == False, "Invalid tau value - check the ranking series"

        return corr, p_value

    def rank_order(self, rule_ranking: pd.Series, original_ranking: pd.Series, method="rbo") -> float:
        """
        Calculates an evaluation metric for two independent ranked lists, only looking at the
        ranked order of features, not the magnitude of their respective ranking metrics
        Args:
            rule_ranking:
            original_ranking:
            method:

        Returns:

        """
        if method == "rbo":
            rank_score = np.round(self.rbo(rule_ranking.index.to_list(), original_ranking.index.to_list()), 5)
            logging.info(f"Ranked Biased Overlap: {rank_score}")

        return rank_score

    def rbo(self, list1, list2, p=0.5):
        """
        Rank biased overlap (RBO) implementation - heavily based on the following implementation
        on GitHub: https://gist.github.com/pjoshi30/cbf51bb9ec73da7040062591a030ceaa#file-rbo_ext-py
        """
        def rec_summation_set(ret, i, d):
            l1 = set(list1[:i]) if i < len(list1) else set(list1)
            l2 = set(list2[:i]) if i < len(list2) else set(list2)
            a_d = len(l1.intersection(l2)) / i
            term = math.pow(p, i) * a_d
            if d == i:
                return ret + term
            return rec_summation_set(ret + term, i + 1, d)

        k = max(len(list1), len(list2))
        x_k = len(set(list1).intersection(set(list2)))
        summation = rec_summation_set(0, 1, k)
        return ((float(x_k) / k) * math.pow(p, k)) + ((1 - p) / p * summation)


def flatten(t):
    return [item for sublist in t for item in sublist]