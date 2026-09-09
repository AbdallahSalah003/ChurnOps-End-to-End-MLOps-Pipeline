import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from src.logger import get_logger

logger = get_logger(__name__)

TARGET = "Churn"


def evaluate_model(model_path: str, validation_path: str) -> dict:
    """
    Evaluate the trained churn classification model
    """
    classifier = joblib.load(model_path)

    val_df = pd.read_csv(validation_path)

    X_val = val_df.drop(columns=[TARGET])

    y_val = val_df[TARGET]

    logger.info(
        "Validation dataset: %s",
        X_val.shape,
    )

    y_pred = classifier.predict(X_val)

    y_prob = classifier.predict_proba(X_val)[:, 1]

    accuracy = float(
        accuracy_score(
            y_val,
            y_pred,
        )
    )

    precision = float(
        precision_score(
            y_val,
            y_pred,
            zero_division=0,
        )
    )

    recall = float(
        recall_score(
            y_val,
            y_pred,
            zero_division=0,
        )
    )

    f1 = float(
        f1_score(
            y_val,
            y_pred,
            zero_division=0,
        )
    )

    roc_auc = float(
        roc_auc_score(
            y_val,
            y_prob,
        )
    )

    logger.info(
        "Accuracy: %.4f",
        accuracy,
    )

    logger.info(
        "Precision: %.4f",
        precision,
    )

    logger.info(
        "Recall: %.4f",
        recall,
    )

    logger.info(
        "F1: %.4f",
        f1,
    )

    logger.info(
        "ROC-AUC: %.4f",
        roc_auc,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }
