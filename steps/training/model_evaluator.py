import mlflow
import pandas as pd
from sklearn.base import ClassifierMixin

from zenml import step
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def model_evaluator(
        model: ClassifierMixin,
        dataset_trn: pd.DataFrame,
        dataset_tst: pd.DataFrame,
        target: str,
        min_train_accuracy: float=0.0,
        min_test_accuracy: float=0.0,
        fail_on_accuracy_quality_gates: bool = False,
) -> None:
    """Evaluate a trained model"""
    trn_acc = model.score(
        dataset_trn.drop(columns=[target]),
        dataset_tst[target],
    )
    logger.info(f"Train Accuracy: {trn_acc * 100:.2f}%")

    tst_acc = model.score(
        dataset_tst.drop(columns=[target]),
        dataset_tst[target],
    )
    logger.info(f"Test Accuracy: {tst_acc * 100:.2f}%")
    mlflow.log_metric("testing_accuracy", tst_acc)

    messages = []
    if trn_acc < min_train_accuracy:
        messages.append(f"Train Accuracy {trn_acc *100:.2f}% is below {min_train_accuracy * 100:.2f}% !")
    if tst_acc < min_test_accuracy:
        messages.append(
            f"Test Accuracy {tst_acc * 100:.2f}% is below {min_test_accuracy * 100:.2f}% !"
        )
    if fail_on_accuracy_quality_gates and messages:
        raise RuntimeError(
            "Model performance did not meet the minimum criteria:\n "
            "\n".join(messages)
        )
    else:
        for message in messages:
            logger.info(message)