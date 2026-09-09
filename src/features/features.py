from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.logger import get_logger

logger = get_logger(__name__)

TARGET = "Churn"

SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def feature_engineering(
    input_path: str, train_output_path: str, validation_output_path: str
):
    """
    Feature engineering and train/validation split
    """
    df = pd.read_csv(input_path)

    logger.info(
        "Loaded dataset: %s",
        df.shape,
    )

    tenure_bins = [
        0,
        12,
        24,
        48,
        60,
        72,
    ]

    tenure_labels = [
        0,
        1,
        2,
        3,
        4,
    ]

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=tenure_bins,
        labels=tenure_labels,
        include_lowest=True,
    ).astype(int)

    df["avg_monthly_charge"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    active_services = [column for column in SERVICE_COLUMNS if column in df.columns]

    df["service_count"] = df[active_services].sum(axis=1)

    df["monthly_charge_x_tenure"] = df["MonthlyCharges"] * df["tenure"]

    y = df[TARGET]

    X = df.drop(columns=[TARGET])

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    logger.info(
        "Train: %s | Validation: %s",
        X_train.shape,
        X_val.shape,
    )

    scaler = StandardScaler()

    numeric_columns = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    X_train = X_train.copy()
    X_val = X_val.copy()

    X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])

    X_val[numeric_columns] = scaler.transform(X_val[numeric_columns])

    train_df = pd.concat(
        [X_train, y_train],
        axis=1,
    )

    val_df = pd.concat(
        [X_val, y_val],
        axis=1,
    )

    Path(train_output_path).parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(
        train_output_path,
        index=False,
    )

    Path(validation_output_path).parent.mkdir(parents=True, exist_ok=True)

    val_df.to_csv(
        validation_output_path,
        index=False,
    )

    logger.info(
        "Training dataset saved to: %s",
        train_output_path,
    )

    logger.info(
        "Validation dataset saved to: %s",
        validation_output_path,
    )
