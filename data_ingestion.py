import tarfile
from pathlib import Path

import boto3
import xgboost as xgb


def download_and_extract_from_s3(bucket, key, dest_dir):
    """
    Downloads a tar.gz file from an AWS S3 bucket and extracts its contents to
    a specified directory.
    Args:
        bucket (str): Name of the S3 bucket.
        key (str): Key (path) to the tar.gz file in the S3 bucket.
        dest_dir (Path or str): Destination directory to save and
            extract the contents.
    Raises:
        Exception: If there is an error downloading the file from S3
            or extracting the tar file.
    Prints:
        Status messages indicating download and extraction progress.
    """
    print(f"Downloading model from s3://{bucket}/{key}")
    s3 = boto3.client("s3")
    local_tar = dest_dir / "model.tar.gz"
    try:
        s3.download_file(bucket, key, str(local_tar))
        print(f"Downloaded to {local_tar}")
    except Exception as e:
        print(f"Error downloading file from S3: {e}")
        raise

    try:
        with tarfile.open(str(local_tar), "r:gz") as tar:
            tar.extractall(path=str(dest_dir))
        print("Extracted files:", list(dest_dir.rglob("*")))
    except Exception as e:
        print(f"Error extracting tar file: {e}")
        raise


def load_bst_model(model_dir: Path):
    """
    Loads an XGBoost Booster model from a specified directory.
    Searches recursively for the first `.bst` model file within
    the given directory, loads the model using XGBoost's Booster,
    and returns the loaded model.
    Args:
        model_dir (Path): Path to the directory containing the
            `.bst` model file.
    Returns:
        xgb.Booster: The loaded XGBoost Booster model.
    Raises:
        FileNotFoundError: If the specified model directory does not exist.
        RuntimeError: If no `.bst` model file is found in the directory.
    """
    print("Searching for .bst model file in", model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    candidates = list(model_dir.rglob("*.bst"))
    if not candidates:
        raise RuntimeError("No .bst model file found in " + str(model_dir))

    model_file = candidates[0]
    print(f"Loading XGBoost model from {model_file}")

    booster = xgb.Booster()
    booster.load_model(str(model_file))
    print("Model loaded successfully")
    return booster
