from steps.etl.data_loader import data_loader
from steps.etl.inference_data_preprocessor import inference_data_processing
from steps.data_quality.drift_quality_gate import drift_quality_gate
from steps.inference.inference_predict import inference_predict
from steps.alerts.notify_on import notify_on_failure, notify_on_success

from zenml import get_pipeline_context, pipeline
from zenml.integrations.evidently.metrics import EvidentlyMetricConfig
from zenml.integrations.evidently.steps import evidently_report_step
from zenml.logger import get_logger

logger = get_logger(__name__)

@pipeline(on_failure=notify_on_failure)
def e2e_use_case_batch_inference():
    """
        Model batch inference pipeline.

        This is a pipeline that loads the inference data, processes
        it, analyze for data drift and run inference.
    """

    model = get_pipeline_context().model
    ##### ETL STAGE ####
    df_inference, target = data_loader(random_state=model.get_artifact("random_state"),
                                       is_inference=True)
    df_inference = inference_data_processing(
        dataset_inf = df_inference, preprocess_pipeline = model.get_artifact("preprocess_pipeline"),
        target = target,
    )

    #### DataQuality stage ####
    report, _ = evidently_report_step(
        reference_dataset = model.get_artifact("dataset_trn"),
        comparison_dataset = df_inference,
        ignored_columns = ["target"],
        metrics = [
            EvidentlyMetricConfig.metric("DataQualityPreset"),
        ],
    )
    drift_quality_gate(report)

    #### Inference stage #####
    inference_predict(
        dataset_inf = df_inference,
        after=["drift_quality_gate"],
    )

    notify_on_success(
        after=["inference_predict"],
    )
