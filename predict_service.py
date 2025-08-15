import os
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import xgboost as xgb
from data_ingestion import download_and_extract_from_s3, load_bst_model
from data_processing import extract_url_features
from fastapi import FastAPI, HTTPException
from services import publish_prediction

# This functions when utilizing an AWS EC2 cluster with the provisioned roles
MODEL_S3_BUCKET = os.environ.get("MODEL_S3_BUCKET", "malicious-url-project")
MODEL_S3_KEY = os.environ.get(
    "MODEL_S3_KEY",
    (
        "url-model-xgboost/output/"
        "sagemaker-xgboost-2025-08-09-22-58-52-766/output/model.tar.gz"
    )
)
LOCAL_MODEL_DIR = Path("/tmp/modeldir")
LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_COLUMNS = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "num_dots",
    "num_hyphens",
    "num_at",
    "num_question_marks",
    "num_equals",
    "num_underscores",
    "num_ampersands",
    "num_digits",
    "has_https",
    "uses_ip",
    "num_subdomains",
    "has_login",
    "has_secure",
    "has_account",
    "has_update",
    "has_free",
    "has_lucky",
    "has_banking",
    "has_confirm",
    "has_port",
    "url_entropy"
]
LABEL_MAPPING = {
    0: "benign",
    1: "phishing",
    2: "malware",
    3: "defacement"
}

app = FastAPI(title="Malicious URL Detection Service")
MODEL = None


@app.on_event("startup")
def load_model_startup():
    """
    Event handler for FastAPI application startup.

    This function downloads and extracts a model file from an S3 bucket,
    then loads the model into the global variable `MODEL`. If any error occurs
    during the process, it prints the error and raises the exception.

    Raises:
        Exception: If downloading, extracting, or loading the model fails.
    """
    global MODEL
    try:
        download_and_extract_from_s3(
            MODEL_S3_BUCKET, MODEL_S3_KEY, LOCAL_MODEL_DIR
        )
        MODEL = load_bst_model(LOCAL_MODEL_DIR)
    except Exception as e:
        print("Error loading model:", e)
        raise


@app.get("/predict/{url:path}")
def predict_url(url: str):
    """
    Predicts the class of a given URL using a pre-trained XGBoost model.
    Args:
        url (str): The URL path parameter, which will be decoded and processed.
    Returns:
        dict: A dictionary containing the decoded URL and its predicted
            class label.
    Raises:
        HTTPException: If the model is not loaded or an error occurs
            during prediction.
    """
    global MODEL
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        decoded_url = unquote(url)
        features = extract_url_features(decoded_url)

        # Enforce feature column order
        features = features[TRAIN_COLUMNS]

        dmatrix = xgb.DMatrix(pd.DataFrame([features]))
        pred_class = int(MODEL.predict(dmatrix)[0])

        result = {
            "url": decoded_url,
            "predicted_class": LABEL_MAPPING.get(pred_class, "unknown")
        }

        # Publish result to Kafka topic (non-blocking, errors ignored)
        try:
            publish_prediction(result)
        except Exception as pub_exc:
            print(f"Kafka publish error: {pub_exc}")

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
