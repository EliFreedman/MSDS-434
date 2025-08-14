import os
import re
import math
import tarfile
import pickle
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse, unquote

import boto3
import pandas as pd
import tldextract
import xgboost as xgb
from fastapi import FastAPI, HTTPException

# =====================
# ENV CONFIG
# =====================
MODEL_S3_BUCKET = os.environ.get("MODEL_S3_BUCKET", "malicious-url-project")
MODEL_S3_KEY = os.environ.get(
    "MODEL_S3_KEY",
    "url-model-xgboost/output/sagemaker-xgboost-2025-08-09-22-58-52-766/output/model.tar.gz"
)
LOCAL_MODEL_DIR = Path("/tmp/modeldir")
LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Define your feature columns in EXACT order as model was trained on
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

# Map your numeric class indices back to label names
LABEL_MAPPING = {
    0: "benign",
    1: "phishing",
    2: "malware",
    3: "defacement"
}

# =====================
# FEATURE EXTRACTION
# =====================
def shannon_entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum(count/lns * math.log2(count/lns) for count in p.values())

def extract_url_features(url):
    parsed = urlparse(url)
    ext = tldextract.extract(url)

    hostname = parsed.netloc
    path = parsed.path
    query = parsed.query

    features = {}
    features['url_length'] = len(url)
    features['hostname_length'] = len(hostname)
    features['path_length'] = len(path)
    features['query_length'] = len(query)

    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['num_at'] = url.count('@')
    features['num_question_marks'] = url.count('?')
    features['num_equals'] = url.count('=')
    features['num_underscores'] = url.count('_')
    features['num_ampersands'] = url.count('&')
    features['num_digits'] = sum(c.isdigit() for c in url)

    features['has_https'] = int(url.lower().startswith('https'))
    features['uses_ip'] = int(
        bool(re.search(r'http[s]?://(?:\d{1,3}\.){3}\d{1,3}', url))
    )

    features['num_subdomains'] = len(ext.subdomain.split('.')) if ext.subdomain else 0

    suspicious_keywords = [
        'login',
        'secure',
        'account',
        'update',
        'free',
        'lucky',
        'banking',
        'confirm'
    ]
    for keyword in suspicious_keywords:
        features[f'has_{keyword}'] = int(keyword in url.lower())

    features['has_port'] = int(':' in hostname)
    features['url_entropy'] = shannon_entropy(url)

    return pd.Series(features)

# =====================
# MODEL LOADING WITH PICKLE FALLBACK
# =====================
def download_and_extract_from_s3(bucket, key, dest_dir):
    print(f"Downloading model from s3://{bucket}/{key}")
    s3 = boto3.client("s3")
    local_tar = dest_dir / "model.tar.gz"
    s3.download_file(bucket, key, str(local_tar))
    print(f"Downloaded to {local_tar}")

    with tarfile.open(str(local_tar), "r:gz") as tar:
        tar.extractall(path=str(dest_dir))
    print("Extracted files:", list(dest_dir.rglob("*")))

def load_bst_model(model_dir: Path):
    print("Searching for .bst model file in", model_dir)
    candidates = list(model_dir.rglob("*.bst"))
    if not candidates:
        raise RuntimeError("No .bst model file found in " + str(model_dir))

    model_file = candidates[0]
    print(f"Loading XGBoost model from {model_file}")

    booster = xgb.Booster()
    booster.load_model(str(model_file))
    print("Model loaded successfully")
    return booster

# =====================
# FASTAPI APP
# =====================
app = FastAPI(title="Malicious URL Detection Service")
MODEL = None

@app.on_event("startup")
def load_model_startup():
    global MODEL
    try:
        download_and_extract_from_s3(MODEL_S3_BUCKET, MODEL_S3_KEY, LOCAL_MODEL_DIR)
        MODEL = load_bst_model(LOCAL_MODEL_DIR)
    except Exception as e:
        print("Error loading model:", e)
        raise

@app.get("/predict/{url:path}")
def predict_url(url: str):
    global MODEL
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        print(f"Raw URL: {url}")
        decoded_url = unquote(url)
        print(f"Decoded URL: {url}")
        features = extract_url_features(decoded_url)
        # Enforce feature column order
        features = features[TRAIN_COLUMNS]

        dmatrix = xgb.DMatrix(pd.DataFrame([features]))
        pred_class = int(MODEL.predict(dmatrix)[0])

        return {
            "url": decoded_url,
            "predicted_class": LABEL_MAPPING.get(pred_class, "unknown")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

