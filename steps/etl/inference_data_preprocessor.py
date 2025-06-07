import pandas as pd
from pandas.conftest import datetime_series
from sklearn.pipeline import Pipeline
from typing_extensions import Annotated

from zenml import step

@step
def inference_data_processing(dataset_inf: pd.DataFrame,
                              preprocessing_pipeline: Pipeline,
                              target: str) -> Annotated[pd.DataFrame, "inference_date"]:

    """Data preprocessing step"""

    dataset_inf[target] = pd.Series([1] * dataset_inf.shape[0])
    dataset_inf = preprocessing_pipeline.transform(dataset_inf)
    dataset_inf.drop(columns=["target"], inplace=True)

    return dataset_inf
