import logging

import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
import shap
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

        return rule_count


    def shap_feat_ranking(self, as_pct: bool=True, top_n: int = None, ) -> pd.Series:
        """
        Returns feature importance ranking of the original model based on shapley values
        Args:
            as_pct:
            top_n:

        Returns:
        """
        logging.info("Fitting Shap Deep Explainer")
        e = shap.DeepExplainer(self.model, x_train[:1000])


    def kendall_tau(self):
        pass
