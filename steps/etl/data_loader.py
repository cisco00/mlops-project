from typing import Tuple
import pandas as pd
from sklearn.datasets import load_breast_cancer
from typing_extensions import Annotated

from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

@step
def data_loader(random_state: int,
                is_inference: bool = False) -> Tuple[
    Annotated[pd.DataFrame, "dataset"],
    Annotated[str, "target_column"],
    Annotated[int, "random_state"]]:

    """Dataset reader step."""

    dataset = load_breast_cancer(as_frame=True)
    inference_size = int(len(dataset.target) * 0.05)
    target = "target"
    dataset: pd.DataFrame = dataset.frame
    inference_subset = dataset.sample(
        inference_size, random_state=random_state
    )

    if is_inference:
        dataset = inference_subset
        dataset.drop(columns=["target"], inplace=True)
    else:
        dataset.drop(columns=["id"], inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    logger.info(f"Dataset with  {len(dataset)} record is loaded.")
    return dataset, target, random_state


