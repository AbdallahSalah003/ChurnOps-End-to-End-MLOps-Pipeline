import shutil
from pathlib import Path

import kagglehub

from src.logger import get_logger

logger = get_logger(__name__)


def collect_data(output_path: str):
    """
    Download the Telco Customer Churn dataset from Kaggle
    """
    DATASET_HANDLE = "blastchar/telco-customer-churn"
    DATASET_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    download_dir = Path(output_path).parent / "kaggle_download"
    download_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading dataset: %s", DATASET_HANDLE)

    downloaded_path = kagglehub.dataset_download(
        DATASET_HANDLE,
        output_dir=str(download_dir),
    )

    downloaded_path = Path(downloaded_path)

    logger.info(
        "Kaggle download location: %s",
        downloaded_path,
    )

    csv_candidates = list(downloaded_path.rglob(DATASET_FILENAME))

    if not csv_candidates:
        raise FileNotFoundError(
            f"Could not find {DATASET_FILENAME} after downloading "
            f"{DATASET_HANDLE}. Downloaded path: {downloaded_path}"
        )

    source_csv = csv_candidates[0]

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_csv, destination)

    logger.info(
        "Raw dataset saved to: %s",
        destination,
    )
