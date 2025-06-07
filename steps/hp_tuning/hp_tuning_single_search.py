from typing import Any, Dict
import pandas as pd

from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

from typing_extensions import Annotated
from utils import get_model_from_config

from zenml import log_metadata, step
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def hp_tuning_single_search(
        model_package: str,
        model_class: str,
        search_grid: Dict[str, Any],
        dataset_tr: pd.DataFrame,
        dataset_tst: pd.DataFrame,
        target: str
) -> Annotated[ClassifierMixin, "hp_result"]:
    """Evaluate a trained model
    A model hyperparameter tuning step that takes in train and test datasets to perform a randomized search for best model
    in configured space.
    """

    model_class = get_model_from_config(model_package, model_class)

    for search_key in search_grid:
        if "range" in search_grid[search_key]:
            search_grid[search_key] = range(
                search_grid[search_key]["range"]["start"],
                search_grid[search_key]["range"]["end"],
                search_grid[search_key]["range"].get("step", 1),
            )

    x_trn = dataset_tr.drop(columns=[target])
    y_trn = dataset_tr[target]
    x_tst = dataset_tst.drop(columns=[target])
    y_tst = dataset_tst[target]

    logger.info("Running hyperparameter tuning with search randomized search")
    cv = RandomizedSearchCV(
        estimator=model_class(),
        param_distributions=search_grid,
        cv=3,
        n_jobs=-1,
        n_iter=10,
        random_state=42,
        scoring="accuracy",
        refit=True,
    )

    cv.fit(x_trn, y_trn)
    y_pred = cv.predict(x_tst)
    score = accuracy_score(y_tst, y_pred)

    log_metadata(
        metadata={
            "metric": float(score)},
        artifact_name="hp_result",
        infer_artifact=True
    )

    return cv.best_estimator_
