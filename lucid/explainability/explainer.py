import logging

import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import shap
from scipy import stats
from lucid.explainability.ig import IGExplainer
from collections import Counter



class PipelineExplainer(object):
    """
    Object containing multiple explainability utils for the ruleset and the
    original model
    """
    def __init__(self, config, dnn_model, ruleset, method):
        self.config = config
        self.model = dnn_model
        self.ruleset = ruleset
        self.method = method



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

        score_df["term_ratio"] = score_df.terms / score_df.terms.sum()
        # # weight the term ratio by confidence
        score_df["term_ratio_weighted"] = score_df["term_ratio"] * score_df["conf"]
        score_df["feat_importance"] = score_df["term_ratio"] * score_df["conf"]
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

    def calc_shap_values(self, x_train: pd.DataFrame, x_test: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculates SHAP values based on training data and model
        Args:
            x_train(pd.DataFrame): train data used to train `self.model`
            x_test(pd.DataFrame): test data used to calculate shap values
            **kwargs: Other keywords

        Returns:
            pd.DataFrame: dataframe with shap values of shape ``(n_features x n_test_samples)``

        """
        e = shap.DeepExplainer(self.model, x_train.to_numpy())
        shap_values = e.shap_values(x_test.to_numpy(), check_additivity=False) # note - this changed
        shap_value_df = pd.DataFrame(data=shap_values[0], columns=x_train.columns)
        return shap_value_df


    def calc_ig_values(self, x_test: pd.DataFrame, **kwargs):
        """
        Calculates the integrated gradient matrix for feature contributions with tabular data
        Args:
            x_test(pd.DataFrame): test data used to calculate integrated gradients

        Returns:
            pd.DataFrame: dataframe with IG contributions of shape ``(n_features x n_test_samples)``
        """
        ig_deep = IGExplainer(model=self.model, type=None)
        ig_values = ig_deep.ig_values(x_test.to_numpy())
        ig_value_df = pd.DataFrame(data=ig_values, columns=x_test.columns)
        # scale such that the feature importance adds up to 1
        scale_factor = ig_value_df.abs().mean().sum()
        ig_value_df /= scale_factor
        return ig_value_df


    def gold_standard_ranking(self, x_train: pd.DataFrame, x_test: pd.DataFrame, as_pct: bool=True, top_n: int = None) -> pd.Series:
        """
        Calculates the "gold standard", which is tested in hypotheses 1 of the dissertation and used as an extended
        reference in the remaining hypotheses to calculate faithfulness.
        """
        shap_value_df = self.calc_shap_values(x_train, x_test)
        ig_value_df = self.calc_ig_values(x_test)
        gold_ranking = self.aggregate_rankings(ig_value_df, shap_value_df, x_train).squeeze()
        gold_ranking = gold_ranking.sort_values(ascending=False)
        gold_ranking.name = "original"

        if as_pct:
            gold_ranking /= gold_ranking.sum()
        if top_n is not None:
            gold_ranking = gold_ranking[:top_n]
        return gold_ranking

    def original_feat_ranking(self, x_train: pd.DataFrame, x_test: pd.DataFrame, method: str = "shap", as_pct: bool=True, top_n: int = None,
                              ) -> pd.Series:
        """
        Returns feature importance ranking of the original model based on shapley values
        """
        logging.info("Fitting Shap Deep Explainer")
        if method == "shap":
            attr_values = self.calc_shap_values(x_train=x_train, x_test=x_test)
        elif method == "ig":
            attr_values = self.calc_ig_values(x_test=x_test)
        rank = attr_values.abs().mean(axis=0).sort_values(ascending=False)
        if as_pct:
            rank /= rank.sum()

        rank.name = "original"
        return rank

    def plot_rankings(self, rule_ranking: pd.Series, original_ranking: pd.Series, save_path: str, top_n: int = 20,
                      ):
        """
        Given two feature rankings, we write a bar plot of both to the output
        Returns:
            None: saves plots to given `save_path`, which is a relative path from the repository root
        """
        joint_df = self.join_rankings(original_ranking, rule_ranking, top_n)

        plot_df = pd.melt(joint_df, value_vars=["original", "rules"], id_vars=["Feature"])
        plot_df = plot_df.rename({"value": "% Rank", "variable": "Extraction"}, axis=1)
        fig, ax = plt.subplots()
        sns.set_theme(style="whitegrid", palette="Paired", font_scale=1)
        sns.barplot(data=plot_df, x="% Rank", y="Feature", hue="Extraction", ax=ax)
        fold = int(str(save_path)[-5])
        plt.title(f"{self.method} - Feature Importances - Fold {fold}")
        plt.savefig(save_path)
        plt.show()

    def join_rankings(self, original_ranking: pd.Series, rule_ranking: pd.Series, top_n: int=20):
        """
        Helper function join two separate rankings from pandas series into a single dataset.
        This is mainly required to plot the comparison of the feature importance ranking from the original
        and the rule-based model
        """
        rule_plot = rule_ranking.sort_values(ascending=False).to_frame().iloc[:top_n]
        original_plot = original_ranking.sort_values(ascending=False).to_frame().iloc[:top_n]
        joint_df = rule_plot.merge(original_plot, how="outer", left_index=True, right_index=True, validate="1:1")
        joint_df = joint_df.fillna(0).sort_values(by=["rules", "original"], ascending=False)
        scale_pct = lambda x,col :  x[col]/x[col].sum()
        joint_df["rules"] = scale_pct(joint_df, "rules")
        joint_df["original"] = scale_pct(joint_df, "original")
        joint_df = joint_df.reset_index().rename({"index": "Feature"}, axis=1)
        return joint_df


    def aggregate_rankings(self, rank_1: pd.DataFrame, rank_2: pd.DataFrame, x_train):
        """
        Calculates the l2
        Returns:

        """
        # express both rankings in terms of absolute % contribution for each sample
        row_scale_matrix = lambda df: df.abs().divide(df.abs().sum(axis=1), axis="rows")
        rank_1 = row_scale_matrix(rank_1)
        rank_2 = row_scale_matrix(rank_2)
        importances = []
        for feat_idx in range(rank_1.to_numpy().shape[1]):
            a = rank_1.iloc[:, feat_idx]
            b = rank_2.iloc[:, feat_idx]
            importances.append(self.euclidean_importance(a,b))

        agg_df = pd.DataFrame(data=importances,
                              index=x_train.columns).rename({0: "importance"}, axis=1)

        # scale importances to sum to 1
        agg_df["importance"] /= agg_df["importance"].sum()
        return agg_df

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
        valid_methods = ["tau", "rho", "weighted_tau"]
        assert method in valid_methods, f"Invalid method, select one of {valid_methods}"

        merge_rank = self.join_rankings(original_ranking, rule_ranking)

        if method == "tau":
            corr, p_value = stats.kendalltau(merge_rank["rules"], merge_rank["original"])
            logging.info(f"Kendall's tau: {np.round(corr, 4)}")
        elif method == "weighted_tau":
            corr, p_value = stats.weightedtau(merge_rank["rules"], merge_rank["original"])
            logging.info(f"Weighted Tau: {np.round(corr, 4)}")
        elif method == "rho":
            corr, p_value = stats.spearmanr(merge_rank["rules"], merge_rank["original"])
            logging.info(f"Spearman Rho: {np.round(corr, 4)}")
        corr = np.round(corr, 5)
        p_value = np.round(p_value, 5)

        assert np.isnan(corr) == False, "Invalid tau value - check the ranking series"

        return corr, p_value

    def rank_order(self, rule_ranking: pd.Series, original_ranking: pd.Series, method="rbo", top_n: int=15) -> float:
        """
        Calculates an evaluation metric for two independent ranked lists, only looking at the
        ranked order of features, not the magnitude of their respective ranking metrics
        Args:
            rule_ranking:
            original_ranking:
            method:

        Returns:

        """
        # rankings = self.join_rankings(rule_ranking=rule_ranking, original_ranking=original_ranking, top_n=15)
        rules = rule_ranking.index.to_list()
        original = original_ranking.index.to_list()
        original_filtered = [elem for elem in original if elem in rules]

        if method == "rbo":
            rank_score = np.round(self.rbo(rules, original_filtered), 5)
            logging.info(f"Ranked Biased Overlap: {rank_score}")

        return rank_score

    def rbo(self, list1, list2, p=0.8):
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
    """
    Helper function to flatten a nested list
    """
    return [item for sublist in t for item in sublist]