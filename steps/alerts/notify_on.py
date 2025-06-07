from zenml import get_step_context, step
from zenml.client import Client
from zenml.utils.dashboard_utils import get_run_url

alerter = Client().active_stack.alerter

def build_message(status: str) -> str:
    """Builds a message to post.
    Args
    status: status to be set in text

    Returns
        str: prepare message.
        """

    step_context = get_step_context()
    run_url = get_run_url(step_context.pipeline_run)

    return (
        f"Pipeline `{step_context.pipeline.name}` [{str(step_context.pipeline.id)}] {status}!\n"
        f"Run `{step_context.pipeline_run.name}` [{str(step_context.pipeline_run.id)}]\n"
        f"URL: {run_url}\n"
    )

def notify_on_failure() -> None:
    """Notify user on step failure. Used in hook"""
    step_context = get_step_context()
    if alerter and step_context.pipeline_run.config.extra["notify_on_failure"]:
        alerter.post(message=build_message(status="FAILED"))

@step(enable_cache=False)
def notify_on_success(notify_on_success: bool) -> None:
    """Notify user on pipeline success"""

    if alerter and notify_on_success:
        alerter.post(message=build_message(status="succeeded"))