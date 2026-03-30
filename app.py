"""
app.py — Flask Backend for Dormancy Fraud Detection
COMP 385 Capstone | Full Stack Application

Endpoints:
    GET  /              → serves the frontend
    GET  /health        → model status check
    POST /predict       → fraud risk prediction
    GET  /history       → last 20 predictions

Scoring note:
    The trained XGBoost model outputs ~0.50 for all inputs due to the
    synthetic dataset having randomly assigned fraud labels (confirmed via
    feature separation analysis in EDA). A transparent rule-based scorer
    is used instead, grounded in the same dormancy signals identified
    during EDA. This is documented in the technical report.
"""

import sys
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("flask_app.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── APP SETUP ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="../Front-end", static_url_path="")
CORS(app)

# ── GLOBALS ───────────────────────────────────────────────────────────────────
MODEL         = None
META          = None
ENCODERS      = None
SCALER        = None
FREQ_MAPS     = None
IMPUTE_MED    = None
DORMANCY_BINS = None
HISTORY       = []

MODELS_DIR    = Path("models")
FEATURES_DIR  = Path("features")

RISK_BANDS = {
    "low":    (0.00, 0.40),
    "medium": (0.40, 0.65),
    "high":   (0.65, 1.00),
}

# ── MODEL LOADING ─────────────────────────────────────────────────────────────

def load_model():
    global MODEL, META, ENCODERS, SCALER, FREQ_MAPS, IMPUTE_MED, DORMANCY_BINS

    for model_file in ["xgboost_v2.joblib", "best_model.joblib", "xgboost.joblib"]:
        path = MODELS_DIR / model_file
        if path.exists():
            MODEL = joblib.load(path)
            logger.info(f"Model loaded: {path}")
            break

    for meta_file in ["training_meta_v2.joblib", "training_meta.joblib"]:
        path = MODELS_DIR / meta_file
        if path.exists():
            META = joblib.load(path)
            logger.info(f"Metadata loaded: {path}")
            break

    for fname, attr in [
        ("encoders.joblib",      "ENCODERS"),
        ("scaler.joblib",        "SCALER"),
        ("freq_maps.joblib",     "FREQ_MAPS"),
        ("impute_median.joblib", "IMPUTE_MED"),
        ("dormancy_bins.joblib", "DORMANCY_BINS"),
    ]:
        path = FEATURES_DIR / fname
        if path.exists():
            globals()[attr] = joblib.load(path)
            logger.info(f"Loaded: {path}")

    if MODEL is None:
        logger.warning("No model file found")
    if META is None:
        logger.warning("No metadata file found")


# ── RULE-BASED SCORER ─────────────────────────────────────────────────────────

def rule_based_score(data: dict) -> float:
    """
    Transparent rule-based fraud scorer grounded in EDA findings.

    Used because the synthetic dataset's fraud labels are randomly assigned
    (confirmed: dormancy_x_hour separation = 3.45 but model outputs 0.50
    for all inputs). Rules are derived from the same signals identified in EDA:
      - time_since_last_transaction (top RF feature, importance 0.1086)
      - is_first_transaction        (derived dormancy flag, diff = 0.185)
      - hour of day                 (fraud peak 00-05h confirmed in EDA)
      - transaction amount          (secondary signal)
      - geo_anomaly_score           (tertiary signal)
    """
    days  = data.get("days_since_last")
    hour  = int(data.get("hour", 12))
    amt   = float(data.get("amount", 0))
    vel   = float(data.get("velocity_score", 1))
    geo   = float(data.get("geo_anomaly_score", 0))

    score = 0.10  # base

    # Dormancy signal (primary)
    if days is None:
        score += 0.35          # first-ever transaction
    else:
        d = float(days)
        if d > 365:   score += 0.30
        elif d > 180: score += 0.25
        elif d > 90:  score += 0.15
        elif d > 30:  score += 0.06

    # Temporal signal (confirmed in EDA — fraud peaks 00-05h)
    if hour in [0, 1, 2, 3, 4]:  score += 0.20
    elif hour in [5, 22, 23]:     score += 0.12
    elif hour == 21:               score += 0.05

    # Amount signal
    if amt > 1000:   score += 0.12
    elif amt > 500:  score += 0.08
    elif amt > 200:  score += 0.04

    # Behavioural signals
    if geo > 0.7:   score += 0.06
    elif geo > 0.4: score += 0.03
    if vel <= 1:    score += 0.04

    return round(min(0.96, max(0.05, score)), 4)


# ── FEATURE ENGINEERING (inference-time) ─────────────────────────────────────

def engineer_features(data: dict):
    """Replicate pipeline feature engineering for a single transaction."""
    amount            = float(data.get("amount", 0))
    days_since_last   = data.get("days_since_last", None)
    hour              = int(data.get("hour", 12))
    transaction_type  = str(data.get("transaction_type", "transfer"))
    merchant_category = str(data.get("merchant_category", "retail"))
    location          = str(data.get("location", "Toronto"))
    device_used       = str(data.get("device_used", "mobile"))
    payment_channel   = str(data.get("payment_channel", "online"))
    sender_account    = str(data.get("sender_account", "ACC_DEMO"))
    ip_address        = str(data.get("ip_address", "0.0.0.0"))
    device_hash       = str(data.get("device_hash", "HASH_DEMO"))
    # Auto-computed server-side — not entered by user
    # spending_deviation: no account history available in demo → default 0
    # velocity_score: derived from dormancy (longer dormant = lower velocity)
    # geo_anomaly: no location history available in demo → default 0
    days_val = data.get("days_since_last")
    if days_val is None:
        velocity_score = 1.0        # first transaction = lowest velocity
    elif float(days_val) > 180:
        velocity_score = 2.0
    elif float(days_val) > 30:
        velocity_score = 5.0
    else:
        velocity_score = 10.0       # active account = high velocity

    spending_deviation = 0.0        # no account history in demo
    geo_anomaly        = 0.0        # no location history in demo

    is_first_transaction = 1 if days_since_last is None else 0
    tslt = float(IMPUTE_MED) if (days_since_last is None and IMPUTE_MED is not None) \
           else (float(days_since_last) if days_since_last is not None else 0.0)

    high_risk_hours   = [0, 1, 2, 3, 4, 5, 22, 23]
    is_high_risk_hour = 1 if hour in high_risk_hours else 0
    is_weekend        = 1 if datetime.now().weekday() >= 5 else 0
    amount_log        = float(np.log1p(amount))

    # Dormancy bucket
    dormancy_bucket     = "recent"
    dormancy_risk_score = 0
    dormancy_bucket_enc = 0

    if days_since_last is not None:
        d = float(days_since_last)
        if DORMANCY_BINS is not None:
            bins   = DORMANCY_BINS
            labels = ["recent", "moderate", "dormant", "long_dormant"]
            risk   = {"recent": 0, "moderate": 1, "dormant": 2, "long_dormant": 3}
            for i in range(len(bins) - 1):
                if bins[i] <= tslt < bins[i + 1]:
                    dormancy_bucket     = labels[i]
                    dormancy_risk_score = risk[labels[i]]
                    dormancy_bucket_enc = risk[labels[i]]
                    break
        else:
            if d > 365:   dormancy_bucket, dormancy_risk_score, dormancy_bucket_enc = "long_dormant", 3, 3
            elif d > 180: dormancy_bucket, dormancy_risk_score, dormancy_bucket_enc = "dormant",      2, 2
            elif d > 30:  dormancy_bucket, dormancy_risk_score, dormancy_bucket_enc = "moderate",     1, 1

    def freq(col_name, value):
        if FREQ_MAPS and col_name in FREQ_MAPS:
            return FREQ_MAPS[col_name].get(value, 1)
        return 1

    def encode(col_name, value):
        if ENCODERS and col_name in ENCODERS:
            le  = ENCODERS[col_name]
            val = value if value in le.classes_ else le.classes_[0]
            return int(le.transform([val])[0])
        return 0

    row = {
        "amount":                      amount,
        "time_since_last_transaction": tslt,
        "spending_deviation_score":    spending_deviation,
        "velocity_score":              velocity_score,
        "geo_anomaly_score":           geo_anomaly,
        "hour":                        hour,
        "day_of_week":                 datetime.now().weekday(),
        "month":                       datetime.now().month,
        "is_weekend":                  is_weekend,
        "is_high_risk_hour":           is_high_risk_hour,
        "is_first_transaction":        is_first_transaction,
        "dormancy_risk_score":         dormancy_risk_score,
        "sender_account_freq":         freq("sender_account", sender_account),
        "receiver_account_freq":       1,
        "device_hash_freq":            freq("device_hash", device_hash),
        "ip_freq":                     freq("ip_address", ip_address),
        "amount_log":                  amount_log,
        "transaction_type_enc":        encode("transaction_type",  transaction_type),
        "merchant_category_enc":       encode("merchant_category", merchant_category),
        "location_enc":                encode("location",           location),
        "device_used_enc":             encode("device_used",        device_used),
        "payment_channel_enc":         encode("payment_channel",    payment_channel),
        "dormancy_bucket_enc":         dormancy_bucket_enc,
    }

    df = pd.DataFrame([row])

    if SCALER is not None:
        scale_cols = [c for c in [
            "amount", "amount_log", "time_since_last_transaction",
            "spending_deviation_score", "velocity_score",
            "geo_anomaly_score", "ip_freq"
        ] if c in df.columns]
        df[scale_cols] = SCALER.transform(df[scale_cols])

    if META and META.get("model_name") == "xgboost_v2":
        df["dormancy_x_hour"]     = df["dormancy_risk_score"] * df["hour"]
        df["dormancy_x_highrisk"] = df["dormancy_risk_score"] * df["is_high_risk_hour"]
        df["dormancy_x_amount"]   = df["dormancy_risk_score"] * df["amount_log"]
        df["first_txn_x_hour"]    = df["is_first_transaction"] * df["hour"]
        df["first_txn_x_amount"]  = df["is_first_transaction"] * df["amount_log"]
        for col in ["spending_deviation_score", "velocity_score", "geo_anomaly_score", "amount"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    if META and "feature_names" in META:
        for col in META["feature_names"]:
            if col not in df.columns:
                df[col] = 0
        df = df[META["feature_names"]]

    return df, dormancy_bucket, dormancy_risk_score, is_first_transaction


def get_risk_level(prob: float) -> str:
    if prob >= RISK_BANDS["high"][0]:   return "high"
    if prob >= RISK_BANDS["medium"][0]: return "medium"
    return "low"


def get_decision(risk_level: str) -> str:
    return {"high": "Block", "medium": "Review", "low": "Approve"}[risk_level]


def get_risk_factors(data: dict, dormancy_bucket: str, is_first: int) -> list:
    factors = []
    days   = data.get("days_since_last")
    hour   = int(data.get("hour", 12))
    amount = float(data.get("amount", 0))
    geo    = float(data.get("geo_anomaly_score", 0))

    # Dormancy factor
    if is_first:
        factors.append({"label": "First-ever transaction", "level": "high",
                        "detail": "No transaction history — is_first_transaction flag triggered"})
    elif days is not None:
        d = float(days)
        if d > 365:
            factors.append({"label": f"Very long dormancy ({int(d)} days)", "level": "high",
                            "detail": f"Account inactive over 1 year — bucket: {dormancy_bucket}"})
        elif d > 180:
            factors.append({"label": f"Long dormancy ({int(d)} days)", "level": "high",
                            "detail": f"Account inactive 6–12 months — bucket: {dormancy_bucket}"})
        elif d > 90:
            factors.append({"label": f"Moderate dormancy ({int(d)} days)", "level": "medium",
                            "detail": f"Account inactive 3–6 months — bucket: {dormancy_bucket}"})
        elif d > 30:
            factors.append({"label": f"Low dormancy ({int(d)} days)", "level": "low",
                            "detail": "Account inactive 1–3 months"})
        else:
            factors.append({"label": f"Active account ({int(d)} days ago)", "level": "low",
                            "detail": "Recent transaction history present"})

    # Hour factor
    if hour in [0, 1, 2, 3, 4]:
        factors.append({"label": f"High-risk hour ({hour}:00)", "level": "high",
                        "detail": "00–04h window — peak fraud period confirmed in EDA"})
    elif hour in [5, 22, 23]:
        factors.append({"label": f"Off-hours transaction ({hour}:00)", "level": "medium",
                        "detail": "Transaction outside normal business hours"})
    else:
        factors.append({"label": f"Normal hour ({hour}:00)", "level": "low",
                        "detail": "Transaction within standard business hours"})

    # Amount factor
    if amount > 1000:
        factors.append({"label": f"Large amount (${amount:,.2f})", "level": "high",
                        "detail": "High-value transaction on potentially dormant account"})
    elif amount > 500:
        factors.append({"label": f"Elevated amount (${amount:,.2f})", "level": "high",
                        "detail": "Transaction amount exceeds $500"})
    elif amount > 200:
        factors.append({"label": f"Moderate amount (${amount:,.2f})", "level": "medium",
                        "detail": "Transaction amount between $200–$500"})
    else:
        factors.append({"label": f"Normal amount (${amount:,.2f})", "level": "low",
                        "detail": "Transaction amount within normal range"})

    # Geo factor
    if geo > 0.7:
        factors.append({"label": f"High geo anomaly ({geo:.2f})", "level": "high",
                        "detail": "Location significantly different from account history"})
    elif geo > 0.4:
        factors.append({"label": f"Moderate geo anomaly ({geo:.2f})", "level": "medium",
                        "detail": "Some geographic deviation detected"})

    return factors


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/health")
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": MODEL is not None,
        "model_name":   META.get("model_name", "unknown") if META else "rule_based",
        "features":     len(META.get("feature_names", [])) if META else 0,
        "threshold":    META.get("threshold", 0.5) if META else 0.5,
        "scorer":       "rule_based",
    })


@app.route("/predict", methods=["POST"])
def predict():
    import time
    start = time.time()

    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        df, dormancy_bucket, dormancy_risk_score, is_first = engineer_features(data)

        # Rule-based score (consistent with EDA findings)
        prob = rule_based_score(data)

        # Also run model inference for logging/comparison
        if MODEL is not None:
            try:
                raw        = MODEL.predict_proba(df)
                model_prob = float(raw[:, 1][0]) if raw.ndim == 2 else float(raw[0])
                logger.info(f"Model raw: {model_prob:.4f} | Rule score: {prob:.4f}")
            except Exception as e:
                logger.warning(f"Model inference failed: {e}")

        elapsed_ms   = round((time.time() - start) * 1000, 1)
        risk_level   = get_risk_level(prob)
        decision     = get_decision(risk_level)
        risk_factors = get_risk_factors(data, dormancy_bucket, is_first)

        result = {
            "fraud_probability":     round(prob, 4),
            "fraud_probability_pct": round(prob * 100, 1),
            "risk_level":            risk_level,
            "decision":              decision,
            "dormancy_bucket":       dormancy_bucket,
            "dormancy_risk_score":   dormancy_risk_score,
            "is_first_transaction":  bool(is_first),
            "risk_factors":          risk_factors,
            "inference_ms":          elapsed_ms,
            "model":                 "rule_based_scorer",
            "threshold":             META.get("threshold", 0.5) if META else 0.5,
            "timestamp":             datetime.now().strftime("%H:%M:%S"),
        }

        HISTORY.append({
            "timestamp":             result["timestamp"],
            "amount":                data.get("amount"),
            "transaction_type":      data.get("transaction_type"),
            "days_since_last":       data.get("days_since_last"),
            "hour":                  data.get("hour"),
            "fraud_probability_pct": result["fraud_probability_pct"],
            "risk_level":            risk_level,
            "decision":              decision,
        })
        if len(HISTORY) > 50:
            HISTORY.pop(0)

        logger.info(
            f"Predict: amount=${data.get('amount')} days={data.get('days_since_last')} "
            f"hour={data.get('hour')} → {prob:.4f} [{risk_level}] {elapsed_ms}ms"
        )
        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/history")
def history():
    return jsonify({"history": list(reversed(HISTORY[-20:]))})


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_model()
    print("\n" + "=" * 60)
    print("  DORMANCY FRAUD DETECTION — Flask Backend")
    print(f"  Model:  {'Loaded ✓' if MODEL else 'Not found'}")
    print("  Scorer: Rule-based (EDA-grounded dormancy signals)")
    print("  URL:    http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000)