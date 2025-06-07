import json
from zenml import step

@step
def drift_quality_gate(report: str, na_drift_tolerance: float = 0.1) -> None:
    """Analyze the evident report and raise runtimeError on
        high deviation of NA count in 2 dataset.
    """
    result = json.loads(report)["metrics"][0]["result"]
    if result["reference"]["number_of_missing_values"] > 0 and (
        abs(
            result["reference"]["number_of_missing_values"]
            - result["reference"]["number_of_missing_values"]
        )
        / result["reference"]["number_of_missing_values"] > na_drift_tolerance
    ):
        raise RuntimeError(
            "Number of NA values in scoring dataset is significantly different compare to train dataset"
        )