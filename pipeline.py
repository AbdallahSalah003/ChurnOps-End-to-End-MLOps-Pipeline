from pathlib import Path

from src.data_pkg.collection import collect_data
from src.data_pkg.validation import validate_data
from src.data_pkg.cleaning import clean_data
from src.features.features import feature_engineering
from src.train.train import train_model
from src.eval.eval import evaluate_model
from src.logger import get_logger

logger = get_logger(__name__)

DATA_DIR = Path("data")
RAW_PATH = DATA_DIR / "raw.csv"
VALIDATION_REPORT_PATH = DATA_DIR / "validation_report.json"
CLEANED_PATH = DATA_DIR / "cleaned.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
MODEL_PATH = DATA_DIR / "model.joblib"


def telco_churn_pipeline():
    logger.info("Starting telco churn pipeline")

    collect_data(output_path=str(RAW_PATH))

    validate_data(
        input_path=str(RAW_PATH),
        validation_report_path=str(VALIDATION_REPORT_PATH),
    )

    clean_data(
        input_path=str(RAW_PATH),
        validation_report_path=str(VALIDATION_REPORT_PATH),
        output_path=str(CLEANED_PATH),
    )

    feature_engineering(
        input_path=str(CLEANED_PATH),
        train_output_path=str(TRAIN_PATH),
        validation_output_path=str(VALIDATION_PATH),
    )

    model_meta = train_model(
        train_path=str(TRAIN_PATH),
        model_output_path=str(MODEL_PATH),
    )

    metrics = evaluate_model(
        model_path=str(MODEL_PATH),
        validation_path=str(VALIDATION_PATH),
    )

    logger.info("Pipeline completed")
    logger.info("Model metadata: %s", model_meta)
    logger.info("Evaluation metrics: %s", metrics)

    return metrics


if __name__ == "__main__":
    telco_churn_pipeline()
