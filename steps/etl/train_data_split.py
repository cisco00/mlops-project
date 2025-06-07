from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated

from zenml import step

@step
def train_data_splitter(dataset: pd.DataFrame, test_size=0.2) -> Tuple[Annotated[pd.DataFrame, "raw_dataset_tr"], Annotated[pd.DataFrame, "raw_dataset_tst"]]:
    """Dataset splitter step."""

    dataset_tr, dataset_tst = train_test_split(dataset,
                                               test_size=test_size,
                                               random_state=42,
                                               shuffle=True)

    dataset_tr = pd.DataFrame(dataset_tr, columns=dataset_tr.columns)
    dataset_tst = pd.DataFrame(dataset_tst, columns=dataset_tst.columns)

    return dataset_tr, dataset_tst
