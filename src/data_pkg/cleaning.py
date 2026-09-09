import json
from pathlib import Path

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)

NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

YES_NO_COLUMNS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "MultipleLines",
]

ONE_HOT_COLUMNS = [
    "gender",
    "InternetService",
    "Contract",
    "PaymentMethod",
]


def clean_data(input_path: str, validation_report_path: str, output_path: str):
    """
    Clean the Telco Customer Churn dataset.

    Args:
        input_path: Path to the input CSV dataset.
        validation_report_path: Path to the JSON validation report.
        output_path: Path where the cleaned CSV will be saved.
    """
    report = json.loads(Path(validation_report_path).read_text())

    if not report["is_valid"]:
        raise ValueError(
            "Cannot clean invalid dataset. " f"Validation errors: {report['errors']}"
        )

    logger.info("Validation passed. Starting cleaning.")

    df = pd.read_csv(Path(input_path))

    logger.info(
        "Raw shape: %s",
        df.shape,
    )

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    duplicate_count = int(df.duplicated().sum())

    if duplicate_count:
        df = df.drop_duplicates()

        logger.info(
            "Dropped %d duplicate rows",
            duplicate_count,
        )

    for column in NUMERIC_COLUMNS:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    before = len(df)

    df = df.dropna(
        subset=[column for column in NUMERIC_COLUMNS if column in df.columns]
    )

    dropped = before - len(df)

    if dropped:
        logger.info(
            "Dropped %d rows with missing numeric values",
            dropped,
        )

    for column in YES_NO_COLUMNS:

        if column not in df.columns:
            continue

        mapping = {
            "Yes": 1,
            "No": 0,
            "No internet service": 0,
            "No phone service": 0,
        }

        df[column] = df[column].map(mapping).astype(int)

    one_hot_columns = [column for column in ONE_HOT_COLUMNS if column in df.columns]

    df = pd.get_dummies(
        df,
        columns=one_hot_columns,
        dtype=int,
    )

    logger.info(
        "One-hot encoded: %s",
        one_hot_columns,
    )

    if df.isna().any().any():
        raise ValueError("Cleaned dataset still contains missing values.")

    output = Path(output_path)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output,
        index=False,
    )

    logger.info(
        "Cleaned dataset saved to: %s",
        output,
    )

    logger.info(
        "Cleaned shape: %s",
        df.shape,
    )
