import dill
import logging
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from typing import *

from .utils import ModelCache
from remix.logic_manipulator.merge import merge
from remix.logic_manipulator.substitute_rules import clausewise_substitute
from remix.rules.C5 import C5
from remix.rules.cart import cart_rules, random_forest_rules, hist_boosting_rules
from remix.rules.rule import Rule
from remix.rules.ruleset import Ruleset, RuleScoreMechanism
from remix.utils.data_handling import stratified_k_fold_split
from remix.utils.parallelism import serialized_function_execute
from remix.extract_rules.eclaire_exp_base import EclaireBase
from sklearn.feature_selection import mutual_info_classif
from multiprocessing import Pool
from tqdm import tqdm  # Loading bar for rule generation


class EclaireCart(EclaireBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.final_algorithm_name = "cart"
        self.intermediate_algorithm_name = "cart"

class EclaireCartSampleWeighted(EclaireCart):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.case_weighting = True

class EclaireHistCart(EclaireCart):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _preprocess_train(self):
        self.train_data = bin_dataset(self.train_data, n_bins=255)


class EclaireCartPrune(EclaireCart):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _preprocess_train(self, threshold: float = 0.05) -> None:
        """
        Filter out features with highest information score w.r.t the target
        """
        mis = mutual_info_classif(X=self.train_data, y=self.train_labels, random_state=42)
        indices = np.where(mis>threshold)
        self.train_data[:, [indices]]
        print(self.train_data.shape)
        return None



def bin_dataset(data: pd.DataFrame, n_bins: int = 255, exclude_cols: List = []) -> pd.DataFrame:
    if type(data) == np.ndarray:
        data = pd.DataFrame(data=data)
    bin_columns = [col for col in data.columns if col not in exclude_cols]

    for col in bin_columns:
        # check that unique valeus don't exceed n_bins
        if data[col].nunique() <= n_bins:
            print(f"Skipping column {col}; {data[col].nunique()} unique values")
            continue

        #         data[col] = pd.cut(x=data[col], bins=n_bins, precision=4)
        data[col] = pd.qcut(x=data[col], q=n_bins, precision=4, duplicates="drop")

        # get lower bound of bin as float - more elegant solution than this?
        data[col] = data[col].astype("str").apply(lambda x: x[1:].split(",")[0]).astype("float")

    return data.to_numpy()