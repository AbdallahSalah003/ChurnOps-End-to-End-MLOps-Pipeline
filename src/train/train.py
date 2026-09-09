from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.logger import get_logger

logger = get_logger(__name__)

TARGET = "Churn"


def train_model(train_path: str, model_output_path: str):
    """
    Train a Logistic Regression model
    for Telco Customer Churn classification.

    Args:
        train_path: Path to the training CSV dataset.
        model_output_path: Path where the trained model will be saved.
    """
    train_df = pd.read_csv(train_path)

    X_train = train_df.drop(columns=[TARGET])

    y_train = train_df[TARGET]

    logger.info(
        "Training data: %s",
        X_train.shape,
    )

    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    classifier.fit(
        X_train,
        y_train,
    )

    logger.info("Logistic Regression trained successfully")

    output_path = Path(model_output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        classifier,
        output_path,
    )

    logger.info(
        "Model saved to: %s",
        output_path,
    )

    return {
        "framework": "scikit-learn",
        "model_type": "LogisticRegression",
        "features": X_train.shape[1],
    }
