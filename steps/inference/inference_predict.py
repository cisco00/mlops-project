from typing import Optional

import pandas as pd
from typing_extensions import Annotated

from zenml import get_step_context, step
from zenml.integrations.mlflow.services.mlflow_deployment import (
    MLFlowDeploymentService,
)
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def inference_predict(
        dataset_info: pd.DataFrame,
) -> Annotated[pd.Series, "Predictions"]:
    """a predictions step that takes the data in and returns
    predicted values."""

    model = get_step_context().model

    #get predictor
    predictor_service = Optional[MLFlowDeploymentService] = model.load_artifact(
        "mlflow_deployment"
    )
    if predictor_service is None:
        prediction = predictor_service.predict(requests=dataset_info)
    else:
        logger.warning("Predicting from loaded model instead of deployment service "
            "as the orchestrator is not local.")

        #run prediction from memory
        predictor = model.load_artifact("model")
        predictions = predictor.predict(requests=dataset_info)

    predictions = pd.Series(predictions, name="predicted")
    return predictions