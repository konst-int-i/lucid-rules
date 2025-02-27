"""
Implementation of ECLAIRE algorithm. This algorithm extracts intermediate rules
for each hidden layer and then performs a change of variables in all of these
rule sets by using a clause-wise level rather than at a term-wise level.
This helps the model avoiding the exponential explosion of terms that arises
from distribution term-wise clauses during substitution. It also helps reducing
the variance in the ruleset sizes while also capturing correlations between
terms when extracting a ruleset for the overall clause.
"""

import dill
import logging
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from typing import *

from .utils import ModelCache
from lucid.logic_manipulator.merge import merge
from lucid.logic_manipulator.substitute_rules import clausewise_substitute
from lucid.rules.C5 import C5
from lucid.rules.cart import cart_rules, random_forest_rules, hist_boosting_rules
from lucid.rules.column_generation import column_generation_rules
from lucid.rules.rule import Rule
from lucid.rules.ruleset import Ruleset, RuleScoreMechanism
from lucid.utils.data_handling import stratified_k_fold_split
from lucid.utils.parallelism import serialized_function_execute
from multiprocessing import Pool
from tqdm import tqdm  # Loading bar for rule generation


################################################################################
## Exposed Methods
################################################################################

class EclaireBase(object):
    def __init__(self,
                 model,
                 train_data,
                 train_labels,
                 # test_data,
                 verbosity=logging.INFO,
                 last_activation=None,
                 threshold_decimals=None,
                 winnow_intermediate=True,
                 winnow_features=True,
                 min_cases=15,
                 intermediate_end_min_cases=None,
                 initial_min_cases=None,
                 ecclectic_min_cases=None,
                 num_workers=1,
                 feature_names=None,
                 output_class_names=None,
                 trials=1,
                 block_size=1,
                 max_number_of_samples=None,
                 min_confidence=0,
                 final_algorithm_name="C5.0",
                 intermediate_algorithm_name="C5.0",
                 estimators=30,
                 ccp_prune=True,
                 regression=False,
                 balance_classes=False,
                 intermediate_tree_max_depth=None,
                 final_tree_max_depth=None,
                 ecclectic=False,
                 max_intermediate_rules=float("inf"),
                 intermediate_drop_percent=0,
                 rule_score_mechanism=RuleScoreMechanism.Accuracy,
                 per_class_elimination=True,
                 case_weighting=False,
                 feature_weighting=False,
                 interaction_prune=True,
                 expl_baseline="shap",
                 **kwargs,
                 ):

        self.model = model
        self.train_data = train_data
        self.train_labels = train_labels
        # self.test_data = test_data
        self.verbosity = verbosity
        self.last_activation = last_activation
        self.threshold_decimals = threshold_decimals
        self.winnow_intermediate = winnow_intermediate
        self.winnow_features = winnow_features
        self.min_cases = min_cases
        self.intermediate_end_min_cases = intermediate_end_min_cases
        self.initial_min_cases = initial_min_cases
        self.ecclectic_min_cases = ecclectic_min_cases
        self.num_workers = num_workers
        self.feature_names = feature_names
        self.output_class_names = output_class_names
        self.trials = trials
        self.block_size = block_size
        self.max_number_of_samples = max_number_of_samples
        self.min_confidence = min_confidence
        self.final_algorithm_name = final_algorithm_name
        self.intermediate_algorithm_name = intermediate_algorithm_name
        self.estimators = estimators
        self.ccp_prune = ccp_prune
        self.regression = regression
        self.balance_classes = balance_classes
        self.intermediate_tree_max_depth = intermediate_tree_max_depth
        self.final_tree_max_depth = final_tree_max_depth
        self.ecclectic = ecclectic
        self.max_intermediate_rules = max_intermediate_rules
        self.intermediate_drop_percent = intermediate_drop_percent
        self.rule_score_mechanism = rule_score_mechanism
        self.per_class_elimination = per_class_elimination
        self.case_weighting = case_weighting
        self.feature_weighting = feature_weighting
        self.interaction_prune = interaction_prune
        self.case_weights = 1
        self.expl_baseline = expl_baseline


    def extract_rules(
        self,
        **kwargs,
    ):
        """
        Extracts a ruleset model that approximates the given Keras model.

        This algorithm extracts intermediate rules for each hidden layer and then
        performs a change of variables in all of these rule sets by using a
        clause-wise level rather than at a term-wise level. This helps the model
        avoiding the exponential explosion of terms that arises from distribution
        term-wise clauses during substitution. It also helps reducing the variance
        in the ruleset sizes while also capturing correlations between terms when
        extracting a ruleset for the overall clause.

        :param keras.Model model: An input instantiated Keras Model object from
            which we will extract rules from.
        :param np.ndarray train_data: A tensor of shape [N, m] with N training
            samples which have m features each.
        :param logging.VerbosityLevel verbosity: The verbosity level to use for this
            function.
        :param str last_activation: Either "softmax" or "sigmoid" indicating which
            activation function should be applied to the last layer of the given
            model if last function is fused with loss. If None, then no activation
            function is applied.
        :param int threshold_decimals: The maximum number of decimals a threshold in
            the generated ruleset may have. If None, then we impose no limit.
        :param bool winnow_intermediate: Whether or not we use winnowing to
            extract intermediate rule sets when using C5.0 for intermediate rule
            set extraction.
        :param bool winnow_features: Whether or not we use winnowing when extracting
            rules to approximate intermediate clauses when using C5.0 for
            clause-approximating.
        :param int min_cases: The minimum number of samples we must have to perform
            a split in a decision tree when approximating intermediate clauses.
        :param int intermediate_end_min_cases:  The minimum number of samples we
            must have to perform a split in a decision tree when extracting
            intermediate rules from the first layer. It is annealed together with
            initial_min_cases such that intermediate rule sets from the last hidden
            layer are extracted using initial_min_cases minimum samples per split
            and intermediate rule sets from the first hidden layer are extracted
            using intermediate_end_min_cases min samples per split. If None, then
            this defaults to initial_min_cases
        :param int initial_min_cases: Initial minimum number of samples required for
            a split when generating intermediate rule sets for hidden layers (see
            description of intermediate_end_min_cases to understand how it is
            annealed). If None, then it defaults to min_cases.
        :param int num_workers: Maximum number of working processes to be spanned
            when extracting rules.
        :param List[str] feature_names: List of feature names to be used for
            generating our rule set. If None, then we will assume all input features
            are named `h_0_0`, `h_0_1`, `h_0_2`, etc.
        :param List[str] output_class_names: List of output class names to be used
            for generating our rule set. If None, then we will assume all output
            are named `h_{d+1}_0`, `h_{d+1}_1`, `h_{d+1}_2`, etc where `d` is the
            number of hidden layers in the network.
        :param int trials: The number of sampling trials to use when using bagging
            for C5.0 in intermediate and clause-wise rule extraction.
        :param int block_size: The hidden layer sampling frequency. That is, how
            often will we use a hidden layer in the input network to extract an
            intermediate rule set from it.
        :param Or[int, float] max_number_of_samples: The maximum number of samples
            to use from the training data. This corresponds to how much we will
            subsample the input training data before using it to construct
            intermediate and clause-wise rules. If given as a number in [0, 1], then
            this represents the fraction of the input set which will be used during
            rule extraction. If None, then we will use the entire training set as
            given.
        :param float min_confidence: The minimum confidence we will require each
            rule in an intermediate rule set to have for us to extract rules from
            its clauses later on (i.e., we will drop all intermediate rules that
            have a confidence less than this value).
        :param str final_algorithm_name: One of ["C5.0", "CART", "random_forest"]
            indicating which rule extraction algorithm to use for extracting rules
            to approximate clauses in intermediate rules.
        :param str intermediate_algorithm_name: One of ["C5.0", "CART",
            "random_forest"] indicating which rule extraction algorithm to use for
            extracting intermediate rules.
        :param int estimators: If using random_forest for any rule extraction,
            this value represents the number of trees we will grow in the forest.
        :param bool ccp_prune: If using CART for any rule extraction,
            this value indicate whether or not we perform CCP post-hoc pruning
            in the trees we extract with CART before rule induction.
        :param bool regression: Whether or not we are working in a regression task
            rather than a classification task. If True, then CART or random_forest
            must be used to extract intermediate rule sets (set by parameter
            `intermediate_algorithm_name`).
        :param bool balance_classes: Whether or not we will use class weights when
            using C5.0 to approximate intermediate clauses using input activations.
        :param int intermediate_tree_max_depth: max tree depth when using CART or
            random_forest for intermediate rule set extraction.
        :param int final_tree_max_depth: max tree depth when using CART or
            random_forest for approximating clauses in intermediate rules using
            input activations only.
        :param bool eclectic: Whether or not ECLAIRE will use the input features
            as a set of activations from which it can extract intermediate rule
            sets from.
        :param int max_intermediate_rules: If given, then we will constraint any
            extracted intermediate rule set for any hidden layer to have at most
            `max_intermediate_rules` in it. If an intermediate rule sets has more,
            then we will drop as many rules as needed to get the rule set to this
            limit. The dropping will be done based on the rule rankings as specified
            by ranking mechanism `rule_score_mechanism`.
        :param float intermediate_drop_percent: The fraction of rules in
            intermediate rule sets we will drop based on their rankings as specified
            by ranking mechanism `rule_score_mechanism`. We will drop the lowest
            `intermediate_drop_percent` of all intermediate rule sets independently
            of each other.
        :param Or[str, RuleScoreMechanism] rule_score_mechanism: The name or enum
            type of the rule scoring mechanism to be used when dropping rules in
            intermediate rule sets.
        :param bool per_class_elimination: If True, and one also requested to drop
            intermediate rules either by setting `max_intermediate_rules` or by
            using `intermediate_drop_percent`, then we will drop rules in a
            per-class basis rather than in a global fashion. That means that all
            rules are ranked only against rules that share their same conclusion.
        :param Dict[str, Any] kwargs: The keywords arguments used for easier
            integration with other rule extraction methods.

        :returns Ruleset: the set of rules extracted from the given model.
        """

        # Preprocess train data if required by experimental child class
        self._preprocess_train()

        # Other experiment setup (to be overridden by child classes)
        self._experiment_setup()

        if self.final_algorithm_name.lower() in ["c5.0", "c5", "see5"]:
            final_algo_call = C5
            final_algo_kwargs = dict(
                winnow=self.winnow_features,
                threshold_decimals=self.threshold_decimals,
                trials=self.trials,
                case_weights=self.case_weights
            )
            # If case weighting is true, determine case weights
            # Note - case weighting only supported by C50 so far
        elif self.final_algorithm_name.lower() in ["cart", "cart_hist"]:
            final_algo_call = cart_rules
            final_algo_kwargs = dict(
                threshold_decimals=self.threshold_decimals,
                ccp_prune=self.ccp_prune,
                max_depth=self.final_tree_max_depth,
                sample_weight=self.case_weights,
            )
            if self.balance_classes:
                final_algo_kwargs["class_weight"] = "balanced"
        elif self.final_algorithm_name.lower() == "random_forest":
            final_algo_call = random_forest_rules
            final_algo_kwargs = dict(
                threshold_decimals=self.threshold_decimals,
                estimators=self.estimators,
                max_depth=self.final_tree_max_depth,
                sample_weight=self.case_weights
            )
        elif self.final_algorithm_name.lower() == "column_generation":
            final_algo_call = column_generation_rules
            final_algo_kwargs = dict(
                cnf = True,
                lambda0 = 0.001,
                lambda1 = 0.001
            )
        elif self.final_algorithm_name.lower() == "hist_boosting":
            final_algo_call = hist_boosting_rules
            final_algo_kwargs = dict(
                regression=self.regression,
                max_depth=self.intermediate_tree_max_depth,
            )
        else:
            raise ValueError(
                f'Unsupported tree extraction algorithm '
                f'{self.final_algorithm_name}. Supported algorithms are '
                '"C5.0", "CART", and "random_forest".'
            )

        if self.intermediate_algorithm_name.lower() in ["c5.0", "c5", "see5"]:
            intermediate_algo_call = C5
            intermediate_algo_kwargs = dict(
                winnow=self.winnow_intermediate,
                threshold_decimals=self.threshold_decimals,
                trials=self.trials,
                case_weights=self.case_weights
            )
            if self.regression:
                raise ValueError(
                    f"One can only use either CART or random_forest as an "
                    f"intermediate tree construction algorithm if the task in "
                    f"hand if a regression task."
                )
        elif self.intermediate_algorithm_name.lower() in ["cart", "cart_hist"]:
            intermediate_algo_call = cart_rules
            intermediate_algo_kwargs = dict(
                threshold_decimals=self.threshold_decimals,
                ccp_prune=self.ccp_prune,
                regression=self.regression,
                max_depth=self.intermediate_tree_max_depth,
                sample_weight=self.case_weights
            )
            if self.balance_classes:
                intermediate_algo_kwargs["class_weight"] = "balanced"
        elif self.intermediate_algorithm_name.lower() == "random_forest":
            intermediate_algo_call = random_forest_rules
            intermediate_algo_kwargs = dict(
                threshold_decimals=self.threshold_decimals,
                estimators=self.estimators,
                regression=self.regression,
                max_depth=self.intermediate_tree_max_depth,
                sample_weight=self.case_weights
            )
        elif self.intermediate_algorithm_name.lower() == "column_generation":
            intermediate_algo_call = column_generation_rules
            intermediate_algo_kwargs = dict(
                cnf = True,
                lambda0 = 0.001,
                lambda1 = 0.001
            )
        elif self.intermediate_algorithm_name.lower() == "hist_boosting":
            intermediate_algo_call = hist_boosting_rules
            intermediate_algo_kwargs = dict(
                regression=self.regression,
                max_depth=self.intermediate_tree_max_depth,
            )
        else:
            raise ValueError(
                f'Unsupported tree extraction algorithm '
                f'{self.intermediate_algorithm_name}. Supported algorithms are '
                '"C5.0", "CART", and "random_forest".'
            )

        if isinstance(self.rule_score_mechanism, str):
            # Then let's turn it into its corresponding enum
            rule_score_mechanism = RuleScoreMechanism.from_string(
                self.rule_score_mechanism
            )

        if (
            self.max_intermediate_rules is not None
        ) and (
            not self.intermediate_drop_percent
        ) and (
            self.max_intermediate_rules != float("inf")
        ):
            self.intermediate_drop_percent = 1

        # Determine whether we want to subsample our training dataset to make it
        # more scalable or not
        sample_fraction = 0
        if self.max_number_of_samples is not None:
            if self.max_number_of_samples < 1:
                sample_fraction = self.max_number_of_samples
            elif self.max_number_of_samples < self.train_data.shape[0]:
                sample_fraction = self.max_number_of_samples / self.train_data.shape[0]

        if sample_fraction:
            [(new_indices, _)] = stratified_k_fold_split(
                X=self.train_data,
                n_folds=1,
                test_size=(1 - sample_fraction),
                random_state=42,
            )
            train_data = train_data[new_indices, :]

        cache_model = ModelCache(
            keras_model=self.model,
            train_data=self.train_data,
            last_activation=self.last_activation,
            feature_names=self.feature_names,
            output_class_names=self.output_class_names,
        )

        if self.initial_min_cases is None:
            # Then we do a constant min cases through the entire network
            initial_min_cases = self.min_cases
        if self.intermediate_end_min_cases is None:
            intermediate_end_min_cases = self.min_cases

        # Compute our total looping space for purposes of logging our progress
        output_layer = len(self.model.layers) - 1
        input_hidden_acts = list(range(1, output_layer, self.block_size))

        if self.regression:
            class_rule_conclusion_map = None
        else:
            # Else this is a classification task
            num_classes = self.model.layers[-1].output_shape[-1]
            class_rule_conclusion_map = {}
            for i in range(num_classes):
                if self.output_class_names is not None:
                    class_rule_conclusion_map[i] = self.output_class_names[i]
                else:
                    class_rule_conclusion_map[i] = i

        if self.regression:
            y_predicted = np.squeeze(self.model.predict(self.train_data), axis=-1)
        else:
            nn_model_predictions = np.argmax(self.model.predict(self.train_data), axis=-1)
            # C5 requires y to be a pd.Series
            y_predicted = pd.Series(nn_model_predictions)

        # First extract rulesets out of every intermediate block
        with tqdm(
            total=len(input_hidden_acts),
            disable=(self.verbosity == logging.WARNING),
        ) as pbar:
            # Extract layer-wise rules

            # Now compute the effective number of workers we've got as
            # it can be less than the provided ones if we have less terms
            effective_workers = min(self.num_workers, len(input_hidden_acts))
            if effective_workers > 1:
                # Them time to do this the multi-process way
                pbar.set_description(
                    f"Extracting rules for all layers using "
                    f"{effective_workers} new processes for "
                    f"{len(input_hidden_acts)} activation blocks"
                )

                with Pool(processes=effective_workers) as pool:
                    serialized_indices = [
                        None for _ in range(len(input_hidden_acts))
                    ]
                    for block_idx, layer_idx in enumerate(
                        input_hidden_acts
                    ):
                        # Let's serialize our (function, args) tuple
                        serialized_indices[block_idx] = dill.dumps((
                            _extract_rules_from_layer,
                            (
                                cache_model.get_layer_activations(
                                    layer_index=layer_idx
                                ),
                                layer_idx,
                                block_idx,
                                input_hidden_acts,
                                self.intermediate_end_min_cases,
                                self.initial_min_cases,
                                intermediate_algo_call,
                                intermediate_algo_kwargs,
                                class_rule_conclusion_map,
                                y_predicted,
                                self.min_confidence,
                            )
                        ))

                    # And do the multi-process pooling call
                    intermediate_rulesets = pool.map(
                        serialized_function_execute,
                        serialized_indices,
                    )
                pbar.update(len(input_hidden_acts))
            else:
                # Else we will do it in this same process in one jump
                intermediate_rulesets = list(map(
                    lambda x: _extract_rules_from_layer(
                        activation=cache_model.get_layer_activations(
                            layer_index=x[1],
                        ),
                        block_idx=x[0],
                        layer_idx=x[1],
                        input_hidden_acts=input_hidden_acts,
                        intermediate_end_min_cases=intermediate_end_min_cases,
                        initial_min_cases=initial_min_cases,
                        intermediate_algo_call=intermediate_algo_call,
                        intermediate_algo_kwargs=intermediate_algo_kwargs,
                        class_rule_conclusion_map=class_rule_conclusion_map,
                        y_predicted=y_predicted,
                        min_confidence=self.min_confidence,
                        pbar=pbar
                    ),
                    enumerate(input_hidden_acts),
                ))
            pbar.set_description("Done extracting intermediate rulesets")

        for (block_idx, layer_idx), rules in zip(
            enumerate(input_hidden_acts),
            intermediate_rulesets,
        ):
            new_ruleset = Ruleset(
                rules=rules,
                feature_names=[
                    f'h_{layer_idx}_{i}'
                    for i in range(cache_model.get_num_activations(layer_idx))
                ]
            )
            if (
                (self.intermediate_drop_percent) and
                (new_ruleset.num_clauses() > 1)
            ):
                # Then let's do some rule dropping for compressing our
                # generated ruleset and improving the complexity of the
                # resulting algorithm
                logging.debug(
                    f"Eliminating rules for ruleset of block {block_idx} using "
                    f"rule ranking mechanism {self.rule_score_mechanism}, drop percent "
                    f"{self.intermediate_drop_percent}, and max number of rules "
                    f"{self.max_intermediate_rules or float('inf')}."
                )
                new_ruleset.rank_rules(
                    X=cache_model.get_layer_activations(
                        layer_index=layer_idx
                    ).to_numpy(),
                    y=y_predicted,
                    score_mechanism=self.rule_score_mechanism,
                    use_label_names=True,
                )
                before_elimination = new_ruleset.num_clauses()
                new_ruleset.eliminate_rules(
                    percent=self.intermediate_drop_percent,
                    per_class=self.per_class_elimination,
                    max_num=(self.max_intermediate_rules or float("inf")),
                )
                after_elimination = new_ruleset.num_clauses()
                logging.debug(
                    f"\tRule elimination generated a rule set with "
                    f"{after_elimination} rules, removing a total of "
                    f"{before_elimination - after_elimination} rules."
                )

            intermediate_rulesets[block_idx] = new_ruleset

        # Now time to replace all intermediate clauses with clauses that only
        # depend on the input activations
        if self.ecclectic:
            end_rules = intermediate_algo_call(
                x=cache_model.get_layer_activations(
                    layer_index=0,
                ),
                y=y_predicted,
                min_cases=(self.ecclectic_min_cases or self.min_cases),
                prior_rule_confidence=1,
                rule_conclusion_map=class_rule_conclusion_map,
                **intermediate_algo_kwargs
            )
        else:
            end_rules = set()
        with tqdm(
            total=len(input_hidden_acts),
            disable=(self.verbosity == logging.WARNING),
        ) as pbar:
            input_acts = cache_model.get_layer_activations(layer_index=0)
            for block_idx, layer_idx in enumerate(input_hidden_acts):
                # Obtain our cached predictions for this block's tensor
                block_out_activations = cache_model.get_layer_activations(
                    layer_index=layer_idx,
                )

                # We will accumulate all rules extracted for this intermediate layer
                # into a ruleset that depends only on our input activations
                extracted_ruleset = Ruleset(feature_names=self.feature_names)
                layer_ruleset = intermediate_rulesets[block_idx]
                clauses = sorted(layer_ruleset.all_clauses(),  key=str)
                num_clauses = len(clauses)

                # Now compute the effective number of workers we've got as
                # it can be less than the provided ones if we have less terms
                effective_workers = min(self.num_workers, num_clauses)
                if effective_workers > 1:
                    # Them time to do this the multi-process way
                    pbar.set_description(
                        f"Extracting rules for layer {layer_idx} using "
                        f"{effective_workers} new processes for {num_clauses} "
                        f"clauses"
                    )
                    with Pool(processes=effective_workers) as pool:
                        # Now time to do a multiprocess map call. Because this
                        # needs to operate only on serializable objects, what
                        # we will do is the following: we will serialize each
                        # partition bound and the function we are applying
                        # into a tuple using dill and then the map operation
                        # will deserialize each entry using dill and execute
                        # the provided method
                        serialized_clauses = [None for _ in range(len(clauses))]
                        for j, clause in enumerate(clauses):
                            # Let's serialize our (function, args) tuple
                            serialized_clauses[j] = dill.dumps(
                                (_extract_rules_from_clause,
                                 (clause,
                                  j,
                                  num_clauses,
                                  layer_idx,
                                  self.min_cases,
                                  input_acts,
                                  block_out_activations,
                                  self.balance_classes,
                                  self.final_algorithm_name,
                                  final_algo_kwargs,
                                  final_algo_call,
                                  ))
                            )

                        # And do the multi-process pooling call
                        new_rulesets = pool.map(
                            serialized_function_execute,
                            serialized_clauses,
                        )
                    # And update our bar with only one step as we do not have
                        # the granularity we do in the non-multi-process way
                    pbar.update(1)
                    # new_rulesets = list(map(
                    #     lambda x: _extract_rules_from_clause(
                    #         clause=x[1],
                    #         i=x[0],
                    #         num_clauses=num_clauses,
                    #         layer_idx=layer_idx,
                    #         min_cases=self.min_cases,
                    #         input_acts=input_acts,
                    #         block_out_activations=block_out_activations,
                    #         balance_classes=self.balance_classes,
                    #         final_algorithm_name=self.final_algorithm_name,
                    #         final_algo_kwargs=final_algo_kwargs,
                    #         final_algo_call=final_algo_call,
                    #         pbar=pbar
                    #     ),
                    #     enumerate(clauses),
                    # ))
                else:
                    # Else we will do it in this same process in one jump
                    new_rulesets = list(map(
                        lambda x: _extract_rules_from_clause(
                            clause=x[1],
                            i=x[0],
                            num_clauses=num_clauses,
                            layer_idx=layer_idx,
                            min_cases=self.min_cases,
                            input_acts=input_acts,
                            block_out_activations=block_out_activations,
                            balance_classes=self.balance_classes,
                            final_algorithm_name=self.final_algorithm_name,
                            final_algo_kwargs=final_algo_kwargs,
                            final_algo_call=final_algo_call,
                            pbar=pbar
                        ),
                        enumerate(clauses),
                    ))

                # Time to do our simple reduction from our map above by
                # accumulating all the generated rules into a single ruleset
                for ruleset in new_rulesets:
                    extracted_ruleset.add_rules(ruleset)

                extracted_ruleset.rules = merge(extracted_ruleset.rules)

                # Merge rules with current accumulation
                pbar.set_description(
                    f"Substituting rules for layer {layer_idx}"
                )
                for i, intermediate_rule in enumerate(
                    intermediate_rulesets[block_idx].rules
                ):
                    new_rule = clausewise_substitute(
                        total_rule=intermediate_rule,
                        intermediate_rules=extracted_ruleset,
                    )
                    end_rules.add(new_rule)

        return Ruleset(
            rules=merge(end_rules),
            feature_names=self.feature_names,
            output_class_names=self.output_class_names,
            regression=self.regression,
        )

    def _experiment_setup(self) -> None:
        pass

    def _preprocess_train(self) -> None:
        pass


def _extract_rules_from_clause(clause,
                               i,
                               num_clauses,
                               layer_idx,
                               min_cases,
                               input_acts,
                               block_out_activations,
                               balance_classes,
                               final_algorithm_name,
                               final_algo_kwargs,
                               final_algo_call,
                               pbar=None):
    if pbar and (i is not None):
        pbar.set_description(
            f'Extracting ruleset for clause {i + 1}/'
            f'{num_clauses} of layer {layer_idx + 1} for '
            f'(min_cases is {min_cases})'
        )

    target = pd.Series(
        data=[True for _ in range(input_acts.shape[0])]
    )
    for term in clause.terms:
        target = np.logical_and(
            target,
            term.apply(
                block_out_activations[str(term.variable)]
            )
        )
    logging.debug(
        f"\tA total of {np.count_nonzero(target)}/"
        f"{len(target)} training samples satisfied {clause}."
    )
    if balance_classes and (
            final_algorithm_name.lower() in ["c5.0", "c5", "see5"]
    ):
        # Then let's extend it so that it supports unbalanced
        # cases in here
        class_weights = \
            sklearn.utils.class_weight.compute_class_weight(
                'balanced',
                np.unique(target),
                target
            )
        case_weights = [1 for _ in target]
        for i, label in enumerate(target):
            case_weights[i] = class_weights[int(label)]

        case_weights = pd.Series(data=case_weights)
        final_algo_kwargs['case_weights'] = case_weights

    new_rules = final_algo_call(
        x=input_acts,
        y=target,
        rule_conclusion_map={
            True: clause,
            False: f"not_{clause}",
        },
        prior_rule_confidence=clause.confidence,
        min_cases=min_cases,
        **final_algo_kwargs
    )
    if pbar:
        pbar.update(1 / num_clauses)

    return new_rules


def _extract_rules_from_layer(
        activation,
        layer_idx,
        block_idx,
        input_hidden_acts,
        intermediate_end_min_cases,
        initial_min_cases,
        intermediate_algo_call,
        intermediate_algo_kwargs,
        class_rule_conclusion_map,
        y_predicted,
        min_confidence,
        pbar=None,
):
    """
    Helper function to extract rules from each layer. Note that this is a separate
    method, since object member functions cannot be serialized
    Returns:

    """
    if len(input_hidden_acts) > 1:
        slope = (intermediate_end_min_cases - initial_min_cases)
        slope = slope / (len(input_hidden_acts) - 1)
        eff_min_cases = intermediate_end_min_cases - (
                slope * block_idx
        )
    else:
        eff_min_cases = intermediate_end_min_cases
    if intermediate_end_min_cases >= 1:
        # Then let's make sure we pass an int
        eff_min_cases = int(np.ceil(eff_min_cases))

    if pbar:
        pbar.set_description(
            f'Extracting ruleset for block output tensor '
            f'{block_idx + 1}/{len(input_hidden_acts)} (min_cases is '
            f'{eff_min_cases})'
        )

    new_rules = intermediate_algo_call(
        x=activation,
        y=y_predicted,
        min_cases=eff_min_cases,
        prior_rule_confidence=1,
        rule_conclusion_map=class_rule_conclusion_map,
        **intermediate_algo_kwargs
    )
    if pbar:
        pbar.update(1)

    if min_confidence:
        real_rules = set()
        for rule in new_rules:
            new_clauses = []
            for clause in rule.premise:
                if clause.confidence >= min_confidence:
                    new_clauses.append(clause)

            if new_clauses:
                rule.premise = set(new_clauses)
                real_rules.add(rule)
        new_rules = real_rules
    return new_rules

