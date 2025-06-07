from steps.deployment import deployment_deploy
from steps.alerts.notify_on import notify_on_success, notify_on_failure

from zenml import pipeline

@pipeline(on_failure=notify_on_failure)
def e2e_use_case_deployment():
    """
        Model deployment pipeline.

        This is a pipeline deploys trained model for future inference.
    """

    ### Deployment Stage####
    deployment_deploy()

    notify_on_success(after=["deployment_deploy"])