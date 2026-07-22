# ============================================================
# MediNLP Web Backend
# FastAPI + MultinomialNB + Red-Flag JSON + SHAP + Recommendation JSON + Symptom Normalization
# ============================================================

import os
import re
import json
import time
import hmac
import base64
import hashlib
import secrets
import sqlite3

try:
    import mysql.connector
    from mysql.connector import Error as MySQLError
except Exception:  # MySQL support is optional until configured
    mysql = None
    MySQLError = Exception
from datetime import datetime
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from joblib import load
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from symptom_engine import SymptomNormalizer

# Load environment variables from .env file
load_dotenv()

try:
    import shap
except Exception:
    shap = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(BASE_DIR, "model_files"))

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(MODEL_DIR, "final_multinomial_nb_model.joblib"))
FEATURE_PATH = os.getenv("FEATURE_PATH", os.path.join(MODEL_DIR, "feature_names.joblib"))
SYMPTOM_LIST_PATH = os.getenv("SYMPTOM_LIST_PATH", os.path.join(MODEL_DIR, "symptom_feature_list.csv"))
RECOMMENDATION_JSON_PATH = os.getenv("RECOMMENDATION_JSON_PATH", os.path.join(MODEL_DIR, "disease_recommendation_lookup_minimal.json"))
RED_FLAG_JSON_PATH = os.getenv("RED_FLAG_JSON_PATH", os.path.join(MODEL_DIR, "red_flag_rules.json"))
DATABASE_PATH = os.getenv("DATABASE_PATH", os.path.join(BASE_DIR, "medinlp_app.sqlite3"))
DB_PROVIDER = os.getenv("DB_PROVIDER", os.getenv("DATABASE_PROVIDER", "sqlite")).strip().lower()
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "medinlp_chatbot")
MYSQL_SSL = os.getenv("MYSQL_SSL", "false").strip().lower() == "true"
MYSQL_SSL_CA = os.getenv("MYSQL_SSL_CA", "/etc/ssl/certs/ca-certificates.crt")
AUTH_SECRET = os.getenv("AUTH_SECRET", "medinlp-dev-secret-change-this")
TOKEN_TTL_SECONDS = int(os.getenv("TOKEN_TTL_SECONDS", str(60 * 60 * 24 * 14)))

TOP_K_DEFAULT = int(os.getenv("TOP_K", "3"))
ENABLE_SHAP = os.getenv("ENABLE_SHAP", "true").lower() == "true"
SHAP_NSAMPLES_DEFAULT = int(os.getenv("SHAP_NSAMPLES", "100"))


def clean_symptom_text(text: Any) -> str:
    text = str(text).replace("\xa0", " ").strip().lower()
    return re.sub(r"\s+", " ", text)


def normalize_symptom_set(symptom_set) -> set:
    return {clean_symptom_text(symptom) for symptom in symptom_set}


def clean_disease_key(text: Any) -> str:
    text = str(text).replace("\xa0", " ").strip().lower()
    return re.sub(r"\s+", " ", text)


app = FastAPI(title="MediNLP Medical Chatbot API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
feature_names: List[str] = []
feature_set = set()
symptom_df = None
recommendation_lookup: Dict[str, Any] = {}
CRITICAL_RED_FLAG_SYMPTOMS = set()
MAJOR_RED_FLAG_SYMPTOMS = set()
RED_FLAG_MESSAGES = {
    "critical": "Serious symptoms are present. Please contact a doctor or emergency medical service as soon as possible.",
    "major": "Multiple serious symptoms are present. Please contact a doctor as soon as possible.",
    "none": "No red flag detected. Prediction can continue."
}
MAJOR_RED_FLAG_THRESHOLD = 2
shap_explainer = None
shap_background = None
symptom_normalizer: Optional[SymptomNormalizer] = None


# ============================================================
# MySQL/SQLite Auth + Chat History Storage
# ER idea used:
# users(id, first_name, last_name, name, email, password)
# chat(id, user_id, title, messages_json)
# report(id, user_id, chat_id, result)
# ============================================================

def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


class MySQLConnectionWrapper:
    """Small wrapper so the existing conn.execute(... ?) style also works with MySQL."""

    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.connection.close()

    def execute(self, query: str, params: tuple = ()):  # sqlite-compatible API
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute(query.replace("?", "%s"), params or ())
        return cursor

    def commit(self):
        self.connection.commit()


def get_mysql_config(database: Optional[str] = None) -> Dict[str, Any]:
    """Build a MySQL Connector/Python config for local MySQL or TiDB Cloud."""
    config: Dict[str, Any] = {
        "host": MYSQL_HOST,
        "port": MYSQL_PORT,
        "user": MYSQL_USER,
        "password": MYSQL_PASSWORD,
        "charset": "utf8mb4",
        "use_unicode": True,
        "connection_timeout": 20,
    }
    if database:
        config["database"] = database
    if MYSQL_SSL:
        config.update(
            {
                "ssl_disabled": False,
                "ssl_ca": MYSQL_SSL_CA,
                "ssl_verify_cert": True,
                "ssl_verify_identity": True,
            }
        )
    return config


def ensure_mysql_database() -> None:
    if mysql is None:
        raise RuntimeError("mysql-connector-python is not installed. Run: python -m pip install mysql-connector-python")
    server_conn = mysql.connector.connect(**get_mysql_config())
    cursor = server_conn.cursor()
    cursor.execute(
        f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}` "
        "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    )
    server_conn.commit()
    cursor.close()
    server_conn.close()


def get_db_connection():
    if DB_PROVIDER == "mysql":
        if mysql is None:
            raise HTTPException(status_code=500, detail="MySQL support is not installed. Install mysql-connector-python.")
        try:
            connection = mysql.connector.connect(
                **get_mysql_config(MYSQL_DATABASE)
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not connect to MySQL database: {exc}")
        return MySQLConnectionWrapper(connection)

    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    if DB_PROVIDER == "mysql":
        ensure_mysql_database()
        with get_db_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    first_name VARCHAR(100) NOT NULL,
                    last_name VARCHAR(100) NOT NULL,
                    name VARCHAR(220) NOT NULL,
                    email VARCHAR(255) NOT NULL UNIQUE,
                    password VARCHAR(255) NOT NULL,
                    created_at VARCHAR(40) NOT NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    user_id INT NOT NULL,
                    title VARCHAR(120) NOT NULL,
                    messages_json LONGTEXT NOT NULL,
                    created_at VARCHAR(40) NOT NULL,
                    updated_at VARCHAR(40) NOT NULL,
                    INDEX idx_chat_user_updated (user_id, updated_at),
                    CONSTRAINT fk_chat_user FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS report (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    user_id INT NOT NULL,
                    chat_id INT NULL,
                    result LONGTEXT NOT NULL,
                    created_at VARCHAR(40) NOT NULL,
                    INDEX idx_report_user (user_id),
                    CONSTRAINT fk_report_user FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                    CONSTRAINT fk_report_chat FOREIGN KEY(chat_id) REFERENCES chat(id) ON DELETE SET NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
            )
            conn.commit()
        return

    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                messages_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS report (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                chat_id INTEGER,
                result TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY(chat_id) REFERENCES chat(id) ON DELETE SET NULL
            )
            """
        )
        conn.commit()

def user_to_public_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "first_name": row["first_name"],
        "last_name": row["last_name"],
        "name": row["name"],
        "email": row["email"],
    }


def normalize_email(email: str) -> str:
    return str(email or "").strip().lower()


def hash_password(password: str, salt: Optional[str] = None) -> str:
    salt = salt or secrets.token_hex(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        str(password).encode("utf-8"),
        salt.encode("utf-8"),
        120_000,
    ).hex()
    return f"{salt}${password_hash}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, expected_hash = stored_hash.split("$", 1)
    except ValueError:
        return False
    actual_hash = hash_password(password, salt).split("$", 1)[1]
    return hmac.compare_digest(actual_hash, expected_hash)


def b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def create_auth_token(user: Dict[str, Any]) -> str:
    payload = {
        "user_id": int(user["id"]),
        "email": user["email"],
        "exp": int(time.time()) + TOKEN_TTL_SECONDS,
    }
    encoded_payload = b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(AUTH_SECRET.encode("utf-8"), encoded_payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{encoded_payload}.{signature}"


def decode_auth_token(token: str) -> Dict[str, Any]:
    try:
        encoded_payload, signature = token.split(".", 1)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid token")

    expected_signature = hmac.new(AUTH_SECRET.encode("utf-8"), encoded_payload.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        raise HTTPException(status_code=401, detail="Invalid token")

    try:
        payload = json.loads(b64url_decode(encoded_payload).decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

    if int(payload.get("exp", 0)) < int(time.time()):
        raise HTTPException(status_code=401, detail="Token expired")
    return payload


def get_current_user(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Login required")
    token = authorization.split(" ", 1)[1].strip()
    payload = decode_auth_token(token)
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (payload["user_id"],)).fetchone()
    if row is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user_to_public_dict(row)


def parse_messages_json(messages_json: str) -> List[Dict[str, Any]]:
    try:
        value = json.loads(messages_json or "[]")
        return value if isinstance(value, list) else []
    except Exception:
        return []


def load_recommendation_lookup(json_path: str) -> Dict[str, Any]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Recommendation JSON not found at: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw_lookup = data.get("disease_recommendations", data)
    return {clean_disease_key(disease_name): info for disease_name, info in raw_lookup.items()}


def get_disease_recommendation(disease_name: str, lookup: Dict[str, Any]) -> Dict[str, Any]:
    info = lookup.get(clean_disease_key(disease_name))
    if info is None:
        return {
            "found": False,
            "doctor_type_patient_should_see": "Recommendation not found.",
            "common_tests_to_discuss_with_clinician": [],
            "short_care_note": "No care note available for this disease."
        }
    doctor_type = info.get("doctor_type_patient_should_see") or info.get("doctor_type") or info.get("specialist") or "Not specified."
    tests = info.get("common_tests_to_discuss_with_clinician") or info.get("tests") or info.get("recommended_tests") or []
    if isinstance(tests, str):
        tests = [t.strip() for t in tests.split(";") if t.strip()]
    care_note = info.get("short_care_note") or info.get("care_note") or info.get("care_notes") or "No care note available."
    response = {
        "found": True,
        "doctor_type_patient_should_see": doctor_type,
        "common_tests_to_discuss_with_clinician": tests,
        "short_care_note": care_note
    }
    if "urgency_level" in info:
        response["urgency_level"] = info["urgency_level"]
    return response


def load_red_flag_rules(json_path: str, feature_set_local: set) -> Dict[str, Any]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Red-flag JSON not found at: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    critical_set = normalize_symptom_set(data.get("critical_red_flag_symptoms", [])).intersection(feature_set_local)
    major_set = normalize_symptom_set(data.get("major_red_flag_symptoms", [])).intersection(feature_set_local)
    messages = data.get("messages", {})
    metadata = data.get("metadata", {})
    return {
        "critical_symptoms": critical_set,
        "major_symptoms": major_set,
        "messages": {
            "critical": messages.get("critical", RED_FLAG_MESSAGES["critical"]),
            "major": messages.get("major", RED_FLAG_MESSAGES["major"]),
            "none": messages.get("none", RED_FLAG_MESSAGES["none"]),
        },
        "major_threshold": metadata.get("major_red_flag_threshold", 2),
        "metadata": metadata
    }


@app.on_event("startup")
def load_all_assets():
    global model, feature_names, feature_set, symptom_df
    global recommendation_lookup
    global CRITICAL_RED_FLAG_SYMPTOMS, MAJOR_RED_FLAG_SYMPTOMS
    global RED_FLAG_MESSAGES, MAJOR_RED_FLAG_THRESHOLD
    global shap_explainer, shap_background
    global symptom_normalizer

    init_database()

    model = load(MODEL_PATH)
    feature_names = [clean_symptom_text(f) for f in load(FEATURE_PATH)]
    feature_set = set(feature_names)
    if os.path.exists(SYMPTOM_LIST_PATH):
        symptom_df = pd.read_csv(SYMPTOM_LIST_PATH)
    recommendation_lookup = load_recommendation_lookup(RECOMMENDATION_JSON_PATH)
    rules = load_red_flag_rules(RED_FLAG_JSON_PATH, feature_set)
    CRITICAL_RED_FLAG_SYMPTOMS = rules["critical_symptoms"]
    MAJOR_RED_FLAG_SYMPTOMS = rules["major_symptoms"]
    RED_FLAG_MESSAGES = rules["messages"]
    MAJOR_RED_FLAG_THRESHOLD = rules["major_threshold"]
    symptom_normalizer = SymptomNormalizer(feature_names)
    if ENABLE_SHAP and shap is not None:
        shap_background = pd.DataFrame([[0] * len(feature_names)], columns=feature_names)
        shap_explainer = shap.KernelExplainer(predict_proba_for_shap, shap_background)
    print("MediNLP assets loaded.")
    print("Features:", len(feature_names), "Diseases:", len(model.classes_))
    print("Symptom normalizer aliases:", len(symptom_normalizer.alias_map))


def extract_symptoms_from_message(message: str) -> Dict[str, Any]:
    """Extract model-ready symptoms from natural user text."""
    if symptom_normalizer is not None:
        return symptom_normalizer.extract(message)

    # Safe fallback if startup assets are not loaded yet.
    cleaned = clean_symptom_text(message)
    extracted = []
    for part in re.split(r"[,;\n]+", cleaned):
        part = clean_symptom_text(part)
        if part in feature_set and part not in extracted:
            extracted.append(part)
    for symptom in sorted(feature_names, key=len, reverse=True):
        pattern = r"\b" + re.escape(symptom) + r"\b"
        if re.search(pattern, cleaned) and symptom not in extracted:
            extracted.append(symptom)
    if "pain" in extracted:
        specific_pain = [s for s in extracted if s != "pain" and "pain" in s.split()]
        if specific_pain:
            extracted.remove("pain")
    return {
        "raw_input": message,
        "cleaned_input": cleaned,
        "accepted_symptoms": [
            {"symptom": symptom, "matched_text": symptom, "method": "fallback_exact", "score": 1.0, "status": "accepted"}
            for symptom in extracted
        ],
        "possible_symptoms": [],
        "negated_symptoms": [],
        "model_input": extracted,
    }


def create_input_vector(user_symptoms: List[str]):
    input_df = pd.DataFrame([[0] * len(feature_names)], columns=feature_names)
    matched_symptoms, unmatched_symptoms, matched_set = [], [], set()
    for symptom in user_symptoms:
        cleaned_symptom = clean_symptom_text(symptom)
        if cleaned_symptom in feature_set:
            input_df.loc[0, cleaned_symptom] = 1
            if cleaned_symptom not in matched_set:
                matched_symptoms.append(cleaned_symptom)
                matched_set.add(cleaned_symptom)
        else:
            unmatched_symptoms.append(cleaned_symptom)
    return input_df, matched_symptoms, unmatched_symptoms


def check_red_flag_rule(extracted_symptoms: List[str], major_threshold: Optional[int] = None):
    if major_threshold is None:
        major_threshold = MAJOR_RED_FLAG_THRESHOLD
    symptom_set_local = normalize_symptom_set(extracted_symptoms)
    critical_hits = sorted(symptom_set_local.intersection(CRITICAL_RED_FLAG_SYMPTOMS))
    major_hits = sorted(symptom_set_local.intersection(MAJOR_RED_FLAG_SYMPTOMS))
    if critical_hits:
        return {"red_flag": True, "severity": "critical", "reason": "Critical red-flag symptom detected.", "triggered_symptoms": critical_hits, "critical_symptoms": critical_hits, "major_symptoms": major_hits, "message": RED_FLAG_MESSAGES["critical"]}
    if len(major_hits) >= major_threshold:
        return {"red_flag": True, "severity": "major", "reason": f"{len(major_hits)} major red-flag symptoms detected.", "triggered_symptoms": major_hits, "critical_symptoms": critical_hits, "major_symptoms": major_hits, "message": RED_FLAG_MESSAGES["major"]}
    return {"red_flag": False, "severity": "none", "reason": "No red-flag rule triggered.", "triggered_symptoms": [], "critical_symptoms": [], "major_symptoms": major_hits, "message": RED_FLAG_MESSAGES["none"]}


def predict_proba_for_shap(data):
    if isinstance(data, pd.DataFrame):
        input_data = data.copy()
    else:
        input_data = pd.DataFrame(data, columns=feature_names)
    return model.predict_proba(input_data)


def extract_shap_vector(raw_shap_values, class_index: int, n_features: int, n_classes: int):
    if isinstance(raw_shap_values, list):
        class_values = raw_shap_values[class_index]
        return class_values[0] if len(class_values.shape) == 2 else class_values
    arr = np.array(raw_shap_values)
    if arr.ndim == 3:
        if arr.shape[0] == 1 and arr.shape[1] == n_features and arr.shape[2] == n_classes:
            return arr[0, :, class_index]
        if arr.shape[0] == n_classes and arr.shape[1] == 1 and arr.shape[2] == n_features:
            return arr[class_index, 0, :]
    if arr.ndim == 2:
        return arr[0]
    raise ValueError(f"Unsupported SHAP output shape: {arr.shape}")


def explain_prediction_with_shap(input_df, top_predictions, nsamples: int = 100):
    if shap_explainer is None:
        return top_predictions
    raw_shap_values = shap_explainer.shap_values(input_df, nsamples=nsamples)
    class_list = [str(c) for c in model.classes_]
    input_values = input_df.iloc[0].values
    n_features, n_classes = len(feature_names), len(class_list)
    explained = []
    for pred in top_predictions:
        disease = str(pred["disease"])
        if disease not in class_list:
            pred["shap_explanation"] = {"error": "Disease class not found."}
            explained.append(pred)
            continue
        class_index = class_list.index(disease)
        shap_vector = extract_shap_vector(raw_shap_values, class_index, n_features, n_classes)
        explanation_df = pd.DataFrame({"symptom": feature_names, "input_value": input_values, "shap_value": shap_vector})
        present_df = explanation_df[explanation_df["input_value"] == 1].copy().sort_values(by="shap_value", ascending=False)
        pred["shap_explanation"] = {
            "present_symptom_contributions": [
                {"symptom": row["symptom"], "contribution": round(float(row["shap_value"]), 6)}
                for _, row in present_df.head(10).iterrows()
            ]
        }
        explained.append(pred)
    return explained


def attach_recommendations_to_predictions(top_predictions):
    for pred in top_predictions:
        pred["recommendation"] = get_disease_recommendation(pred["disease"], recommendation_lookup)
    return top_predictions


def predict_pipeline(user_symptoms: List[str], top_k: int = TOP_K_DEFAULT, enable_shap: bool = ENABLE_SHAP, shap_nsamples: int = SHAP_NSAMPLES_DEFAULT):
    input_df, matched_symptoms, unmatched_symptoms = create_input_vector(user_symptoms)
    if not matched_symptoms:
        return {"status": "failed", "red_flag": False, "message": "No symptoms matched the model feature list.", "matched_symptoms": [], "unmatched_symptoms": unmatched_symptoms, "red_flag_result": None, "top_predictions": []}
    red_flag_result = check_red_flag_rule(matched_symptoms)
    if red_flag_result["red_flag"]:
        return {"status": "red_flag", "red_flag": True, "message": red_flag_result["message"], "matched_symptoms": matched_symptoms, "unmatched_symptoms": unmatched_symptoms, "red_flag_result": red_flag_result, "top_predictions": []}
    probabilities = model.predict_proba(input_df)[0]
    top_indexes = np.argsort(probabilities)[-top_k:][::-1]
    top_predictions = [{"rank": rank, "disease": str(model.classes_[index]), "confidence": round(float(probabilities[index]), 4), "confidence_percent": round(float(probabilities[index]) * 100, 2)} for rank, index in enumerate(top_indexes, start=1)]
    if enable_shap and shap_explainer is not None:
        top_predictions = explain_prediction_with_shap(input_df, top_predictions, nsamples=shap_nsamples)
    top_predictions = attach_recommendations_to_predictions(top_predictions)
    return {"status": "success", "red_flag": False, "message": "No red flag detected. Prediction completed.", "matched_symptoms": matched_symptoms, "unmatched_symptoms": unmatched_symptoms, "red_flag_result": red_flag_result, "top_predictions": top_predictions}


class ChatRequest(BaseModel):
    message: Optional[str] = None
    symptoms: Optional[List[str]] = None
    top_k: Optional[int] = TOP_K_DEFAULT
    enable_shap: Optional[bool] = ENABLE_SHAP
    shap_nsamples: Optional[int] = SHAP_NSAMPLES_DEFAULT


class ChatResponse(BaseModel):
    status: str
    red_flag: bool
    message: str
    extracted_symptoms: List[str]
    matched_symptoms: List[str]
    unmatched_symptoms: List[str]
    red_flag_result: Optional[Dict[str, Any]]
    top_predictions: List[Dict[str, Any]]
    possible_symptoms: List[Dict[str, Any]] = []
    negated_symptoms: List[str] = []
    symptom_extraction: Optional[Dict[str, Any]] = None






class RegisterRequest(BaseModel):
    first_name: str
    last_name: str = ""
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class ChatSaveRequest(BaseModel):
    title: str = "New chat"
    messages: List[Dict[str, Any]] = []


class ChatUpdateRequest(BaseModel):
    title: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None


class ReportSaveRequest(BaseModel):
    chat_id: Optional[int] = None
    result: Dict[str, Any]


@app.post("/api/auth/register")
def register_user(request: RegisterRequest):
    first_name = str(request.first_name or "").strip()
    last_name = str(request.last_name or "").strip()
    email = normalize_email(request.email)
    password = str(request.password or "")

    if not first_name:
        raise HTTPException(status_code=400, detail="First name is required")
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        raise HTTPException(status_code=400, detail="Valid email is required")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    name = f"{first_name} {last_name}".strip()
    password_hash = hash_password(password)
    created_at = now_iso()

    try:
        with get_db_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO users (first_name, last_name, name, email, password, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (first_name, last_name, name, email, password_hash, created_at),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM users WHERE id = ?", (cursor.lastrowid,)).fetchone()
    except Exception as exc:
        message = str(exc).lower()
        if isinstance(exc, sqlite3.IntegrityError) or "duplicate" in message or "unique" in message:
            raise HTTPException(status_code=409, detail="This email is already registered")
        raise

    user = user_to_public_dict(row)
    return {"token": create_auth_token(user), "user": user}


@app.post("/api/auth/login")
def login_user(request: LoginRequest):
    email = normalize_email(request.email)
    password = str(request.password or "")
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

    if row is None or not verify_password(password, row["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    user = user_to_public_dict(row)
    return {"token": create_auth_token(user), "user": user}


@app.get("/api/auth/me")
def auth_me(current_user: Dict[str, Any] = Depends(get_current_user)):
    return {"user": current_user}


@app.get("/api/chats")
def list_chats(current_user: Dict[str, Any] = Depends(get_current_user)):
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT id, user_id, title, created_at, updated_at FROM chat WHERE user_id = ? ORDER BY updated_at DESC",
            (current_user["id"],),
        ).fetchall()
    return {"chats": [dict(row) for row in rows]}


@app.post("/api/chats")
def create_chat(request: ChatSaveRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    title = str(request.title or "New chat").strip()[:80] or "New chat"
    messages_json = json.dumps(request.messages or [], ensure_ascii=False)
    created_at = now_iso()
    with get_db_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO chat (user_id, title, messages_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (current_user["id"], title, messages_json, created_at, created_at),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM chat WHERE id = ? AND user_id = ?", (cursor.lastrowid, current_user["id"])).fetchone()
    return {"chat": {"id": row["id"], "user_id": row["user_id"], "title": row["title"], "messages": parse_messages_json(row["messages_json"]), "created_at": row["created_at"], "updated_at": row["updated_at"]}}


@app.get("/api/chats/{chat_id}")
def read_chat(chat_id: int, current_user: Dict[str, Any] = Depends(get_current_user)):
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM chat WHERE id = ? AND user_id = ?", (chat_id, current_user["id"])).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"chat": {"id": row["id"], "user_id": row["user_id"], "title": row["title"], "messages": parse_messages_json(row["messages_json"]), "created_at": row["created_at"], "updated_at": row["updated_at"]}}


@app.put("/api/chats/{chat_id}")
def update_chat(chat_id: int, request: ChatUpdateRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM chat WHERE id = ? AND user_id = ?", (chat_id, current_user["id"])).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Chat not found")

        title = str(request.title if request.title is not None else row["title"]).strip()[:80] or "New chat"
        messages_json = row["messages_json"] if request.messages is None else json.dumps(request.messages, ensure_ascii=False)
        updated_at = now_iso()
        conn.execute(
            "UPDATE chat SET title = ?, messages_json = ?, updated_at = ? WHERE id = ? AND user_id = ?",
            (title, messages_json, updated_at, chat_id, current_user["id"]),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM chat WHERE id = ? AND user_id = ?", (chat_id, current_user["id"])).fetchone()
    return {"chat": {"id": row["id"], "user_id": row["user_id"], "title": row["title"], "messages": parse_messages_json(row["messages_json"]), "created_at": row["created_at"], "updated_at": row["updated_at"]}}


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: int, current_user: Dict[str, Any] = Depends(get_current_user)):
    with get_db_connection() as conn:
        cursor = conn.execute("DELETE FROM chat WHERE id = ? AND user_id = ?", (chat_id, current_user["id"]))
        conn.commit()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"status": "deleted", "chat_id": chat_id}


@app.post("/api/reports")
def save_report(request: ReportSaveRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    created_at = now_iso()
    result_json = json.dumps(request.result or {}, ensure_ascii=False)
    with get_db_connection() as conn:
        if request.chat_id is not None:
            chat_row = conn.execute("SELECT id FROM chat WHERE id = ? AND user_id = ?", (request.chat_id, current_user["id"])).fetchone()
            if chat_row is None:
                raise HTTPException(status_code=404, detail="Chat not found")
        cursor = conn.execute(
            "INSERT INTO report (user_id, chat_id, result, created_at) VALUES (?, ?, ?, ?)",
            (current_user["id"], request.chat_id, result_json, created_at),
        )
        conn.commit()
    return {"report": {"id": cursor.lastrowid, "user_id": current_user["id"], "chat_id": request.chat_id, "created_at": created_at}}


@app.get("/")
def root():
    return {"app": "MediNLP Medical Chatbot API", "status": "running", "features": len(feature_names), "diseases": len(model.classes_) if model is not None else 0, "input_note": "Natural English/Banglish/Bangla symptom text is normalized to model symptoms before prediction."}


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "feature_count": len(feature_names), "disease_count": len(model.classes_) if model is not None else 0, "recommendation_count": len(recommendation_lookup), "critical_red_flags": len(CRITICAL_RED_FLAG_SYMPTOMS), "major_red_flags": len(MAJOR_RED_FLAG_SYMPTOMS), "shap_enabled": ENABLE_SHAP and shap_explainer is not None, "symptom_normalizer_loaded": symptom_normalizer is not None, "symptom_alias_count": len(symptom_normalizer.alias_map) if symptom_normalizer is not None else 0, "nearby_doctor_search": "Google Maps text link only; no Hospital API/Foursquare/Overpass backend", "database_provider": DB_PROVIDER}



@app.get("/api/symptoms")
def get_symptoms(q: Optional[str] = None, limit: int = 50):
    symptoms = feature_names
    if q:
        query = clean_symptom_text(q)
        symptoms = [s for s in symptoms if query in s]
    return {"count": min(len(symptoms), limit), "symptoms": symptoms[:limit]}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    extracted_symptoms = []
    extraction_details = None
    possible_symptoms: List[Dict[str, Any]] = []
    negated_symptoms: List[str] = []

    # Direct symptom list still works for testing/autocomplete/manual calls.
    if request.symptoms:
        extracted_symptoms.extend([clean_symptom_text(s) for s in request.symptoms])

    # Natural text goes through dataset-aware normalization first.
    if request.message:
        extraction_details = extract_symptoms_from_message(request.message)
        possible_symptoms = extraction_details.get("possible_symptoms", [])
        negated_symptoms = extraction_details.get("negated_symptoms", [])

        for symptom in extraction_details.get("model_input", []):
            if symptom not in extracted_symptoms:
                extracted_symptoms.append(symptom)

    result = predict_pipeline(
        user_symptoms=extracted_symptoms,
        top_k=request.top_k or TOP_K_DEFAULT,
        enable_shap=request.enable_shap if request.enable_shap is not None else ENABLE_SHAP,
        shap_nsamples=request.shap_nsamples or SHAP_NSAMPLES_DEFAULT,
    )

    message = result["message"]
    if result["status"] == "failed" and possible_symptoms:
        message = "No high-confidence symptom matched the model feature list. Possible symptoms were detected; please rephrase or use one of the suggestions."

    return {
        "status": result["status"],
        "red_flag": result["red_flag"],
        "message": message,
        "extracted_symptoms": extracted_symptoms,
        "matched_symptoms": result["matched_symptoms"],
        "unmatched_symptoms": result["unmatched_symptoms"],
        "red_flag_result": result["red_flag_result"],
        "top_predictions": result["top_predictions"],
        "possible_symptoms": possible_symptoms,
        "negated_symptoms": negated_symptoms,
        "symptom_extraction": extraction_details,
    }

