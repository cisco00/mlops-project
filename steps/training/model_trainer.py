import mlflow
import pandas as pd
from sklearn.base import ClassifierMixin
from typing_extensions import Annotated

from zenml import step, ArtifactConfig, get_step_context
from zenml.client import Client
from zenml.logger import get_logger
from zenml.integrations.mlflow.experiment_trackers import (
MLFlowExperimentTracker
)
from zenml.integrations.mlflow.steps.mlflow_registry import (
mlflow_register_model_step
)
from zenml.logger import get_logger

logger = get_logger(__name__)
experiment_tracker = Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(experiment_tracker,
                                            MLFlowExperimentTracker):
    raise RuntimeError(
        "Your active stack needs to contain a MLFlow experiemnt "
        "tracker for this example to work"
    )

@step
def model_trainer(
        dataset_trn: pd.DataFrame,
        model: ClassifierMixin,
        target: str,
        name: str
) -> Annotated[
    ClassifierMixin, ArtifactConfig(name="model", is_model_artifact = True)
]:
    logger.info(f"Train Model: {model}....")
    mlflow.sklearn.autolog()
    model.fit(dataset_trn.drop(columns=[target]),
              dataset_trn[target])

    #register mlflow model
    mlflow_register_model_step.entrypoint(
        model, name=name
    )

    #keep_track of mlflow model

    model_registry = Client().active_stack.model_registry
    if model_registry:
        version = model_registry.get_latest_model_version(
            name=name, stage=None
        )
        if version:
            model_ = get_step_context().model
            model_.log_metadata(
                {"model_registry_version": version.version}
            )
    return model