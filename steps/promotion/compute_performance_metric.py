from typing import Tuple

import pandas as pd
from sklearn.metrics import accuracy_score
from typing_extensions import Annotated

from zenml import Model, get_step_context, step
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def compute_performance_metric_on_current_data(
        dataset_tst: pd.DataFrame,
        target_env: str
) -> Tuple[Annotated[float, "latest_metric"], Annotated[float, "Current_metric"]]:
    """Get metrics for comparison during promotion on fresh dataset.

        A metrics calculation step. It computes metric
        on recent test dataset.
    """
    X = dataset_tst.drop(columns=["target"])
    y = dataset_tst["target"]
    logger.info("Evaluating model metrics.....")

    latest_version = get_step_context().model
    current_version = Model(name=latest_version.name, version=target_env)

    latest_version_number = latest_version.number
    try:
        current_version_number = current_version.number
    except KeyError:
        current_version_number = None

    if current_version_number is None:
        current_version_number = -1
        metrics = {latest_version_number: 1.0, current_version_number: 0.0}
    else:
        #get predictor
        predictors = {
            latest_version_number: latest_version.load_artifact("model"),
            current_version_number: current_version.load_artifact("model"),
        }
        metrics = {}
        for version in [latest_version_number, current_version_number]:
            #predict and ealuate
            predictions = predictors[version].predict(X)
            metrics[version] = accuracy_score(y, predictions)
    return metrics[latest_version_number], metrics[current_version_number]
