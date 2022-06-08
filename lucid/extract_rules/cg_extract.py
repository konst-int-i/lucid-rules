import numpy as np
from lucid.rules.ruleset import Ruleset
from lucid.rules.column_generation import column_generation_rules
import pandas as pd


def extract_rules(
        model,
        train_data,
        cnf=True,
        lambda0=0.001,
        lambda1=0.0001,
        num_thresh: int = 9,
        negations=True,
        iter_max: int = 25,
        verbose: bool = False,
        silent: bool = False,
        output_class_names=None,
        feature_names=None,
        **kwargs
):
    """
    Uses column generation to extract rules given the model's prediction
    Args:
        model:
        train_data:

    Returns:

    """
    train_data = pd.DataFrame(train_data, columns=feature_names)
    model_pred_prob = model.predict(train_data)
    y_model_pred = np.argmax(model_pred_prob, axis=1)

    cg_rules, cg_rule_objects = column_generation_rules(
        x=pd.DataFrame(train_data, columns=feature_names),
        y=y_model_pred,
        cnf=cnf,
        lambda0=lambda0,
        lambda1=lambda1,
        num_thresh=num_thresh,
        negations=negations,
        iter_max=iter_max,
        verbose=verbose,
        silent=silent
    )

    ruleset = Ruleset(
        cg_rule_objects,
        feature_names=feature_names,
        output_class_names=output_class_names
    )
    cg_rule_predictions = cg_rules.predict(train_data)
    cg_rules.ruleset = ruleset
    return cg_rules
