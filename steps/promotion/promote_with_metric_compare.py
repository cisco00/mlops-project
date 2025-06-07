from utils import promote_in_model_registry

from zenml import Model, get_step_context, step
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def promote_with_metric_compare(
        latest_metric: float,
        current_metric: float,
        mlflow_model_name: str,
        target_env: str
) -> None:
    """Try to promote trained model.
    a model promotion step. It gets precomputed
    metrics for 2 model version: latest and currently promoted to target environment
    (Production, Staging, etc) and compare than in order to define
    if newly trained model is performing better or not. If new model
    version is better by metric - it will get relevant
    tag, otherwise previously promoted model version will remain.

    If the latest version is the only one - it will get promoted automatically.
    """

    should_promote = True

    latest_version = get_step_context().model
    current_version = Model(name=latest_version.name, version=target_env)

    try:
        current_version_number = current_version.number
    except KeyError:
        current_version_number = None

    if current_version_number is None:
        logger.info("No current version found - promoting latest")

    else:
        logger.info(
            f"Latest model metric={latest_metric:.6f}\n"
            f"Current model metric={current_metric:.6f}"
        )
        if latest_metric >= current_metric:
            logger.info("Latest model version outperforms current version - promoting latest")
        else:
            logger.info("Current model version outperforms latest version - keeping current")
            should_promote = False

    if should_promote:
        #promote in model control plane
        model = get_step_context().model
        model.set_stage(
            stage=target_env, force=True
        )
        logger.info(f"Current model version was promoted to {target_env},")

