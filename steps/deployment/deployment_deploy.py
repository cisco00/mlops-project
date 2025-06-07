from typing import Optional

from typing_extensions import Annotated
from zenml import ArtifactConfig, get_step_context, step
from zenml.client import Client
from zenml.integrations.mlflow.services.mlflow_deployment import (
MLFlowDeploymentService,
)
from zenml.integrations.mlflow.steps.mlflow_deployer import (
mlflow_model_registry_deployer_step
)
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def deployment_deploy() -> Annotated[Optional[MLFlowDeploymentService], ArtifactConfig(name="mlflow_deployment", is_deployment_artifact=True),]:
    """Prediction step.
    This is an example of a predictions step that takes the data in and returns
    predicted values."""

    if Client().active_stack.orchestrator.flavor == "local":
        model = get_step_context()

        #deploy predictor service
        deployment_service = mlflow_model_registry_deployer_step.entrypoint(
            registry_model_name=model.name,
            registry_model_version=model.run_metadata[
                "model_registry_version"
            ],
            replace_existing=True,
        )
    else:
        logger.warning("Skipping deployment as the orchestrator is not local.")
        deployment_service = None
    return deployment_service

