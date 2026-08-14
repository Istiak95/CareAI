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
import smtplib
import ssl
from email.message import EmailMessage

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
from fastapi import (
    FastAPI,
    HTTPException,
    Header,
    Depends,
    UploadFile,
    File,
)
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from symptom_engine import SymptomNormalizer
from gemini_symptom_engine import GeminiSymptomEngine

try:
    import certifi
except Exception:
    certifi = None

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
_DEFAULT_SSL_CA = certifi.where() if certifi is not None else "/etc/ssl/certs/ca-certificates.crt"
MYSQL_SSL_CA = os.getenv("MYSQL_SSL_CA", _DEFAULT_SSL_CA).strip()
AUTH_SECRET = os.getenv("AUTH_SECRET", "medinlp-dev-secret-change-this")
TOKEN_TTL_SECONDS = int(os.getenv("TOKEN_TTL_SECONDS", str(60 * 60 * 24 * 14)))
# ============================================================
# Password Reset / Email Configuration
# ============================================================

PASSWORD_RESET_TTL_SECONDS = int(
    os.getenv("PASSWORD_RESET_TTL_SECONDS", "1800")
)

FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "http://localhost:5173"
).rstrip("/")

SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()

SMTP_FROM_EMAIL = os.getenv(
    "SMTP_FROM_EMAIL",
    SMTP_USER
).strip()

SMTP_USE_TLS = (
    os.getenv("SMTP_USE_TLS", "true")
    .strip()
    .lower()
    == "true"
)

TOP_K_DEFAULT = int(os.getenv("TOP_K", "3"))
ENABLE_SHAP = os.getenv("ENABLE_SHAP", "true").lower() == "true"
SHAP_NSAMPLES_DEFAULT = int(os.getenv("SHAP_NSAMPLES", "30"))
# ============================================================
# Gemini Symptom NLP Configuration
# ============================================================

ENABLE_GEMINI_NLP = (
    os.getenv("ENABLE_GEMINI_NLP", "true")
    .strip()
    .lower()
    == "true"
)

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    "",
).strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash",
).strip()

# ============================================================
# Voice Speech-to-Text Configuration
# ============================================================

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
    "",
).strip()

GROQ_TRANSCRIPTION_MODEL = os.getenv(
    "GROQ_TRANSCRIPTION_MODEL",
    "whisper-large-v3",
).strip()

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is missing from the server environment.",
        )

    return Groq(api_key=api_key)
# Vercel request body maximum 4.5 MB.
# Keep our audio below that limit.
MAX_AUDIO_SIZE_BYTES = 3_500_000

ALLOWED_AUDIO_TYPES = {
    "audio/webm": ".webm",
    "audio/mp4": ".mp4",
    "audio/x-m4a": ".m4a",
    "audio/m4a": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "application/octet-stream": ".webm",
}
ENABLE_SEMANTIC = os.getenv("ENABLE_SEMANTIC", "true").lower() == "true"
_BUNDLED_SEMANTIC_MODEL = os.path.join(MODEL_DIR, "semantic_model")
SEMANTIC_MODEL_PATH = os.getenv(
    "SEMANTIC_MODEL_PATH",
    _BUNDLED_SEMANTIC_MODEL
    if os.path.isdir(_BUNDLED_SEMANTIC_MODEL)
    else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)


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
gemini_symptom_engine: Optional[
    GeminiSymptomEngine
] = None


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
# ============================================================
# Password Reset Token Helpers
# ============================================================

def password_hash_version(stored_hash: str) -> str:
    """
    Creates a fingerprint of the current stored password hash.
    After a successful password reset, old reset links automatically
    become invalid because the stored password hash changes.
    """
    return hashlib.sha256(
        str(stored_hash).encode("utf-8")
    ).hexdigest()


def create_password_reset_token(user_row) -> str:
    payload = {
        "user_id": int(user_row["id"]),
        "email": str(user_row["email"]),
        "purpose": "password_reset",
        "password_version": password_hash_version(
            user_row["password"]
        ),
        "exp": int(time.time()) + PASSWORD_RESET_TTL_SECONDS,
    }

    encoded_payload = b64url_encode(
        json.dumps(
            payload,
            separators=(",", ":")
        ).encode("utf-8")
    )

    signing_input = (
        f"password-reset:{encoded_payload}"
    ).encode("utf-8")

    signature = hmac.new(
        AUTH_SECRET.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).hexdigest()

    return f"{encoded_payload}.{signature}"


def decode_password_reset_token(token: str) -> Dict[str, Any]:
    try:
        encoded_payload, signature = str(
            token or ""
        ).split(".", 1)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired password reset link",
        )

    signing_input = (
        f"password-reset:{encoded_payload}"
    ).encode("utf-8")

    expected_signature = hmac.new(
        AUTH_SECRET.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(
        signature,
        expected_signature
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired password reset link",
        )

    try:
        payload = json.loads(
            b64url_decode(
                encoded_payload
            ).decode("utf-8")
        )
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired password reset link",
        )

    if payload.get("purpose") != "password_reset":
        raise HTTPException(
            status_code=400,
            detail="Invalid password reset link",
        )

    if int(payload.get("exp", 0)) < int(time.time()):
        raise HTTPException(
            status_code=400,
            detail="Password reset link has expired",
        )

    return payload


def send_password_reset_email(
    recipient_email: str,
    reset_token: str,
) -> None:

    if not SMTP_HOST:
        raise RuntimeError(
            "SMTP_HOST is not configured"
        )

    if not SMTP_FROM_EMAIL:
        raise RuntimeError(
            "SMTP_FROM_EMAIL is not configured"
        )

    reset_url = (
        f"{FRONTEND_URL}/?reset_token={reset_token}"
    )

    message = EmailMessage()

    message["Subject"] = "Reset your CareAI password"
    message["From"] = SMTP_FROM_EMAIL
    message["To"] = recipient_email

    expiry_minutes = max(
        1,
        PASSWORD_RESET_TTL_SECONDS // 60
    )

    message.set_content(
        f"""
Hello,

We received a request to reset your CareAI password.

Open the following link to create a new password:

{reset_url}

This link will expire in approximately {expiry_minutes} minutes.

If you did not request a password reset, you can ignore this email.

CareAI
"""
    )

    context = ssl.create_default_context()

    with smtplib.SMTP(
        SMTP_HOST,
        SMTP_PORT,
        timeout=20,
    ) as server:

        server.ehlo()

        if SMTP_USE_TLS:
            server.starttls(context=context)
            server.ehlo()

        if SMTP_USER and SMTP_PASSWORD:
            server.login(
                SMTP_USER,
                SMTP_PASSWORD,
            )

        server.send_message(message)

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
    global gemini_symptom_engine

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
    symptom_normalizer = SymptomNormalizer(
        feature_names,
        enable_semantic=ENABLE_SEMANTIC,
        semantic_model_name=SEMANTIC_MODEL_PATH,
    )

    gemini_symptom_engine = None

    if ENABLE_GEMINI_NLP and GEMINI_API_KEY:
        try:
            gemini_symptom_engine = GeminiSymptomEngine(
                feature_names=feature_names,
                api_key=GEMINI_API_KEY,
                model=GEMINI_MODEL,
            )

            print(
                "Gemini symptom NLP loaded:",
                GEMINI_MODEL,
            )

        except Exception as exc:
            print(
                "Gemini symptom NLP disabled:",
                exc,
            )
    if ENABLE_SHAP and shap is not None:
        shap_background = pd.DataFrame([[0] * len(feature_names)], columns=feature_names)
        shap_explainer = shap.KernelExplainer(predict_proba_for_shap, shap_background)
    print("MediNLP assets loaded.")
    print("Features:", len(feature_names), "Diseases:", len(model.classes_))
    print("Symptom normalizer aliases:", len(symptom_normalizer.alias_map))


def apply_red_flag_alias_resolver(
    message: str,
    symptoms: List[str],
    negated_symptoms: List[str],
) -> Dict[str, List[str]]:
    """
    Generic deterministic resolver for ALL red-flag symptoms.

    Uses the existing SymptomNormalizer alias map rather than
    hardcoding one symptom such as shortness of breath.

    Red-flag symptoms are only added from curated aliases /
    exact phrases, not semantic similarity.
    """

    if symptom_normalizer is None:
        return {
            "symptoms": symptoms,
            "negated_symptoms": negated_symptoms,
            "matched_by_safety_alias": [],
        }

    cleaned = clean_symptom_text(message)

    final_symptoms = list(symptoms)
    final_negated = list(negated_symptoms)
    safety_matches = []

    red_flag_set = (
        set(CRITICAL_RED_FLAG_SYMPTOMS)
        | set(MAJOR_RED_FLAG_SYMPTOMS)
    )

    # Longest aliases first so specific phrases win.
    alias_items = sorted(
        symptom_normalizer.alias_map.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    negation_words = {
        "no",
        "not",
        "without",
        "never",
        "nai",
        "nei",
        "na",
        "নাই",
        "নেই",
        "না",
    }

    for alias, symptom in alias_items:

        if symptom not in red_flag_set:
            continue

        alias = clean_symptom_text(alias)

        if not alias:
            continue

        # --------------------------------------------------
        # Generic modifier-tolerant alias matching
        #
        # Examples:
        # shash nite kosto
        # shash nite onek kosto
        # shash nite khub beshi kosto
        #
        # Works for ALL curated red-flag aliases, not only
        # shortness of breath.
        # --------------------------------------------------

        modifier_words = {
            "onek",
            "khub",
            "beshi",
            "onekta",
            "very",
            "really",
            "extremely",
            "severely",
            "অনেক",
            "খুব",
            "বেশি",
        }

        alias_tokens = alias.split()

        modifier_pattern = "|".join(
            re.escape(word)
            for word in sorted(
                modifier_words,
                key=len,
                reverse=True,
            )
        )

        # Permit up to two controlled intensity words
        # between words of a curated alias.
        gap_pattern = (
            r"\s+"
            r"(?:(?:"
            + modifier_pattern
            + r")\s+){0,2}"
        )

        flexible_alias = gap_pattern.join(
            re.escape(token)
            for token in alias_tokens
        )

        pattern = (
            r"(?<![A-Za-z0-9_\u0980-\u09FF])"
            + flexible_alias
            + r"(?![A-Za-z0-9_\u0980-\u09FF])"
        )

        match = re.search(
            pattern,
            cleaned,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        before = cleaned[:match.start()].split()
        after = cleaned[match.end():].split()

        is_negated_match = False

        # Negation immediately before the phrase.
        if before and before[-1] in negation_words:
            is_negated_match = True

        # Negation immediately after the phrase.
        if after and after[0] in negation_words:
            is_negated_match = True

        # One helper word between symptom and negation:
        # "shash nite kosto hocche na"
        helper_words = {
            "hocche",
            "hoche",
            "ase",
            "ache",
            "is",
            "are",
            "ekdom",
        }

        if (
            len(after) >= 2
            and after[0] in helper_words
            and after[1] in negation_words
        ):
            is_negated_match = True

        if is_negated_match:

            if symptom not in final_negated:
                final_negated.append(symptom)

            if symptom in final_symptoms:
                final_symptoms.remove(symptom)

            continue

        if (
            symptom not in final_symptoms
            and symptom not in final_negated
        ):
            final_symptoms.append(symptom)

            safety_matches.append(
                {
                    "symptom": symptom,
                    "alias": alias,
                }
            )

    return {
        "symptoms": final_symptoms,
        "negated_symptoms": final_negated,
        "matched_by_safety_alias": safety_matches,
    }


def extract_symptoms_from_message(
    message: str,
) -> Dict[str, Any]:
    """
    Final CareAI hybrid NLP:

    OLD ENGINE
    ----------
    exact / alias / compact / fuzzy /
    SentenceTransformer semantic

    PLUS

    GEMINI
    ------
    Bangla/Banglish/context/intent/
    negation/clarification

    Gemini failure never disables the old engine.
    """

    # ========================================================
    # A. OLD CAREAI ENGINE ALWAYS RUNS
    # ========================================================

    if symptom_normalizer is not None:
        local_result = symptom_normalizer.extract(
            message
        )
    else:
        local_result = {
            "raw_input": message,
            "cleaned_input":
                clean_symptom_text(message),
            "accepted_symptoms": [],
            "possible_symptoms": [],
            "negated_symptoms": [],
            "model_input": [],
        }

    # ========================================================
    # B. GEMINI RUNS AS SECOND ENGINE
    # ========================================================

    gemini_result = None

    if gemini_symptom_engine is not None:
        gemini_result = (
            gemini_symptom_engine.analyze(
                message
            )
        )

    # ========================================================
    # C. GEMINI DOWN? OLD CAREAI CONTINUES
    # ========================================================

    if not gemini_result:

        fallback = dict(local_result)

        fallback.update(
            {
                "old_engine_used":
                    True,

                "gemini_used":
                    False,

                "hybrid_mode":
                    "old_engine_fallback",

                "intent":
                    "symptom_input",

                "intent_confidence":
                    0.0,

                "old_model_input":
                    list(
                        local_result.get(
                            "model_input",
                            [],
                        )
                    ),

                "gemini_model_input":
                    [],
            }
        )

        return fallback

    # ========================================================
    # D. READ OLD ENGINE EVIDENCE
    # ========================================================

    trusted_local = []
    semantic_local = []

    local_details = list(
        local_result.get(
            "accepted_symptoms",
            [],
        )
    )

    for item in local_details:

        if not isinstance(item, dict):
            continue

        symptom = clean_symptom_text(
            item.get(
                "symptom",
                "",
            )
        )

        if symptom not in feature_set:
            continue

        method = str(
            item.get(
                "method",
                "",
            )
        ).strip()

        try:
            score = float(
                item.get(
                    "score",
                    0.0,
                )
            )
        except Exception:
            score = 0.0

        # Strong old-engine evidence.
        if method in {
            "dataset_phrase",
            "alias_mapping",
            "compact_alias",
        }:

            if symptom not in trusted_local:
                trusted_local.append(
                    symptom
                )

        elif (
            method
            == "fuzzy_matching"
            and score >= 0.92
        ):

            if symptom not in trusted_local:
                trusted_local.append(
                    symptom
                )

        elif method == "semantic_matching":

            semantic_local.append(
                {
                    "symptom":
                        symptom,

                    "score":
                        score,
                }
            )

    # ========================================================
    # E. GEMINI INTENT
    # ========================================================

    intent = str(
        gemini_result.get(
            "intent",
            "symptom_input",
        )
    )

    try:
        intent_confidence = float(
            gemini_result.get(
                "intent_confidence",
                0.0,
            )
        )
    except Exception:
        intent_confidence = 0.0

    non_symptom_intents = {
        "explanation_request",
        "greeting",
        "other",
    }

    # Only suppress prediction when Gemini is highly
    # confident AND old engine has no strong direct evidence.
    if (
        intent in non_symptom_intents
        and intent_confidence >= 0.85
        and not trusted_local
    ):

        language = gemini_result.get(
            "language",
            "unknown",
        )

        if intent == "explanation_request":

            if language in {
                "bangla",
                "banglish",
                "mixed",
            }:
                assistant_message = (
                    "এটা নতুন symptom description মনে হচ্ছে না; "
                    "তুমি আগের result বুঝতে চাচ্ছ। তাই আমি নতুন "
                    "disease prediction চালাইনি।"
                )
            else:
                assistant_message = (
                    "This looks like a request to explain the "
                    "previous result, not a new symptom description, "
                    "so I did not run a new disease prediction."
                )

        elif intent == "greeting":

            if language in {
                "bangla",
                "banglish",
                "mixed",
            }:
                assistant_message = (
                    "হ্যালো 👋 তোমার symptoms লিখো—Bangla, "
                    "Banglish বা English যেকোনোভাবে।"
                )
            else:
                assistant_message = (
                    "Hello 👋 Describe your symptoms in English, "
                    "Bangla, or Banglish."
                )

        else:

            if language in {
                "bangla",
                "banglish",
                "mixed",
            }:
                assistant_message = (
                    "এই message-এ নতুন symptom description পাইনি। "
                    "Symptom থাকলে একটু বিস্তারিত লিখো।"
                )
            else:
                assistant_message = (
                    "I did not find a new symptom description in "
                    "that message. Please describe your symptoms."
                )

        return {
            "raw_input":
                message,

            "cleaned_input":
                clean_symptom_text(message),

            "accepted_symptoms":
                [],

            "possible_symptoms":
                [],

            "negated_symptoms":
                [],

            "model_input":
                [],

            "language":
                gemini_result.get(
                    "language",
                    "unknown",
                ),

            "intent":
                intent,

            "intent_confidence":
                intent_confidence,

            "clarification_needed":
                False,

            "follow_up_question":
                None,

            "skip_prediction":
                True,

            "assistant_message":
                assistant_message,

            "old_engine_used":
                True,

            "gemini_used":
                True,

            "hybrid_mode":
                "intent_gate",

            # Debug fields let you prove both engines ran.
            "old_model_input":
                list(
                    local_result.get(
                        "model_input",
                        [],
                    )
                ),

            "gemini_model_input":
                list(
                    gemini_result.get(
                        "model_input",
                        [],
                    )
                ),
        }

    # ========================================================
    # F. GEMINI POSITIVE SYMPTOMS
    # ========================================================

    gemini_positive = []

    for symptom in gemini_result.get(
        "model_input",
        [],
    ):

        symptom = clean_symptom_text(
            symptom
        )

        if (
            symptom in feature_set
            and symptom not in gemini_positive
        ):
            gemini_positive.append(
                symptom
            )

    # ========================================================
    # ========================================================
    # G. FINAL HYBRID POSITIVE MERGE
    # ========================================================

    # A follow-up message contains the original vague text
    # plus the user's clarification.
    #
    # Example:
    # amar betha hoche
    # User clarification: buker ba pashe
    #
    # In this situation Gemini's resolved canonical symptoms
    # are authoritative. Otherwise an old fuzzy guess such as
    # "pain foot" can survive beside the correct "pain chest".

    is_followup_context = (
        intent == "symptom_followup"
        or "user clarification:" in message.lower()
    )

    combined = []

    if is_followup_context:

        # FOLLOW-UP:
        # Use Gemini's resolved canonical symptoms.
        # Do not carry incompatible guesses from the
        # original vague message into disease prediction.
        for symptom in gemini_positive:

            if symptom not in combined:
                combined.append(
                    symptom
                )

    else:

        # NORMAL MESSAGE:
        # Preserve the original CareAI + Gemini hybrid logic.

        # 1. Strong old exact/alias/fuzzy evidence.
        for symptom in trusted_local:

            if symptom not in combined:
                combined.append(
                    symptom
                )

        # 2. Gemini evidence.
        for symptom in gemini_positive:

            if symptom not in combined:
                combined.append(
                    symptom
                )

    # 3. SentenceTransformer semantic evidence remains.
    #
    # Normal input:
    #   keep when Gemini agrees OR score >= 0.90
    #
    # Follow-up:
    #   keep ONLY when Gemini agrees, because the original
    #   message was explicitly ambiguous.

    semantic_consensus = []

    for item in semantic_local:

        symptom = item["symptom"]
        score = item["score"]

        if is_followup_context:

            keep_semantic = (
                symptom in gemini_positive
            )

        else:

            keep_semantic = (
                symptom in gemini_positive
                or score >= 0.90
            )

        if keep_semantic:

            if symptom not in combined:
                combined.append(
                    symptom
                )

            semantic_consensus.append(
                symptom
            )

    # ========================================================
    # H. BANGLISH ASE / ACHE SAFETY
    # ========================================================

    message_words = set(
        message.lower().split()
    )

    banglish_copula_words = {
        "ase",
        "asey",
        "achey",
    }

    pain_markers = {
        "pain",
        "betha",
        "byatha",
        "jontrona",
        "aching",
        "headache",
        "backache",
        "toothache",
    }

    has_banglish_copula = bool(
        message_words
        & banglish_copula_words
    )

    has_pain_context = any(
        marker in message.lower()
        for marker in pain_markers
    )

    if (
        "ache" in combined
        and has_banglish_copula
        and not has_pain_context
    ):
        combined.remove(
            "ache"
        )

    # ========================================================
    # I. NEGATION
    # ========================================================
    #
    # Gemini is authoritative for scoped negation when it
    # successfully analyzed the message.
    #
    # If Gemini fails entirely, old negation already works
    # through the fallback return above.

    negated = []

    for symptom in gemini_result.get(
        "negated_symptoms",
        [],
    ):

        symptom = clean_symptom_text(
            symptom
        )

        if (
            symptom in feature_set
            and symptom not in negated
        ):
            negated.append(
                symptom
            )

    # Never feed negated symptoms to the model.
    combined = [
        symptom
        for symptom in combined
        if symptom not in negated
    ]

    # ========================================================
    # J. ACCEPTED DETAILS / PROVENANCE
    # ========================================================

    accepted = []

    existing = set()

    for item in local_details:

        if not isinstance(
            item,
            dict,
        ):
            continue

        symptom = clean_symptom_text(
            item.get(
                "symptom",
                "",
            )
        )

        if (
            symptom in combined
            and symptom not in negated
        ):

            accepted.append(
                item
            )

            existing.add(
                symptom
            )

    for symptom in gemini_positive:

        if (
            symptom in combined
            and symptom not in existing
        ):

            accepted.append(
                {
                    "symptom":
                        symptom,

                    "matched_text":
                        message,

                    "method":
                        "gemini_semantic_nlp",

                    "score":
                        1.0,

                    "status":
                        "accepted",
                }
            )

            existing.add(
                symptom
            )

    # ========================================================
    # ========================================================
    # GENERIC_RED_FLAG_POSSIBLE_PROMOTION
    # ========================================================
    # If the OLD CareAI engine already found a red-flag
    # symptom as a strong fuzzy possible match, promote it
    # conservatively instead of losing it before safety check.
    #
    # This applies to ALL configured red-flag symptoms,
    # not only shortness of breath.

    red_flag_possible_promotions = []

    red_flag_feature_set = (
        set(CRITICAL_RED_FLAG_SYMPTOMS)
        | set(MAJOR_RED_FLAG_SYMPTOMS)
    )

    for item in local_result.get(
        "possible_symptoms",
        [],
    ):

        if not isinstance(item, dict):
            continue

        symptom = clean_symptom_text(
            item.get("symptom", "")
        )

        method = str(
            item.get("method", "")
        ).strip()

        try:
            score = float(
                item.get("score", 0.0)
            )
        except Exception:
            score = 0.0

        matched_text = clean_symptom_text(
            item.get("matched_text", "")
        )

        if symptom not in red_flag_feature_set:
            continue

        # Never restore a symptom already marked negated.
        if symptom in negated:
            continue

        # Only promote fuzzy evidence here.
        # Semantic similarity alone is not enough for
        # a safety-critical symptom.
        if method != "fuzzy_possible":
            continue

        token_count = len(
            matched_text.split()
        )

        # Multi-word phrases can use a slightly lower
        # threshold because they carry more context.
        threshold = (
            0.85
            if token_count >= 2
            else 0.93
        )

        if score < threshold:
            continue

        if symptom not in combined:
            combined.append(symptom)

        red_flag_possible_promotions.append(
            {
                "symptom": symptom,
                "matched_text": matched_text,
                "score": score,
                "method": method,
            }
        )

    # GENERIC_RED_FLAG_ALIAS_RESOLUTION
    safety_result = apply_red_flag_alias_resolver(
        message,
        combined,
        negated,
    )

    combined = safety_result[
        "symptoms"
    ]

    negated = safety_result[
        "negated_symptoms"
    ]

    safety_alias_matches = safety_result[
        "matched_by_safety_alias"
    ]

    # K. FINAL RESULT
    # ========================================================

    return {
        "raw_input":
            message,

        "cleaned_input":
            clean_symptom_text(
                message
            ),

        "accepted_symptoms":
            accepted,

        "possible_symptoms":
            local_result.get(
                "possible_symptoms",
                [],
            ),

        "negated_symptoms":
            sorted(
                negated
            ),

        "model_input":
            combined,

        "language":
            gemini_result.get(
                "language",
                "unknown",
            ),

        "intent":
            intent,

        "intent_confidence":
            intent_confidence,

        "clarification_needed":
            gemini_result.get(
                "clarification_needed",
                False,
            ),

        "follow_up_question":
            gemini_result.get(
                "follow_up_question"
            ),

        # Debug/provenance
        "old_engine_used":
            True,

        "gemini_used":
            True,

        "hybrid_mode":
            "old_plus_gemini",

        "old_model_input":
            list(
                local_result.get(
                    "model_input",
                    [],
                )
            ),

        "gemini_model_input":
            gemini_positive,

        "semantic_consensus":
            semantic_consensus,

        "followup_gemini_authoritative":
            is_followup_context,

        "safety_alias_matches":
            safety_alias_matches,

        "red_flag_possible_promotions":
            red_flag_possible_promotions,
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

    # Previous CareAI result used ONLY when
    # the user asks to explain an earlier result.
    previous_result: Optional[
        Dict[str, Any]
    ] = None


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

class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
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
@app.post("/api/auth/forgot-password")
def forgot_password(request: ForgotPasswordRequest):

    email = normalize_email(request.email)

    if not re.match(
        r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
        email,
    ):
        raise HTTPException(
            status_code=400,
            detail="Valid email is required",
        )

    # Same response whether account exists or not.
    # This prevents email/account enumeration.
    generic_response = {
        "message": (
            "If an account exists for this email, "
            "a password reset link has been sent."
        )
    }

    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,),
        ).fetchone()

    if row is None:
        return generic_response

    reset_token = create_password_reset_token(row)

    try:
        send_password_reset_email(
            email,
            reset_token,
        )

    except Exception as exc:
        print(
            "Password reset email error:",
            exc,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Password reset email could not "
                "be sent. Please try again later."
            ),
        )

    return generic_response


@app.post("/api/auth/reset-password")
def reset_password(request: ResetPasswordRequest):

    new_password = str(
        request.password or ""
    )

    if len(new_password) < 6:
        raise HTTPException(
            status_code=400,
            detail=(
                "Password must be at least "
                "6 characters"
            ),
        )

    payload = decode_password_reset_token(
        request.token
    )

    user_id = payload.get("user_id")

    with get_db_connection() as conn:

        row = conn.execute(
            "SELECT * FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()

        if row is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid or expired "
                    "password reset link"
                ),
            )

        if normalize_email(
            row["email"]
        ) != normalize_email(
            payload.get("email", "")
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid or expired "
                    "password reset link"
                ),
            )

        current_password_version = (
            password_hash_version(
                row["password"]
            )
        )

        token_password_version = str(
            payload.get(
                "password_version",
                "",
            )
        )

        if not hmac.compare_digest(
            current_password_version,
            token_password_version,
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "This password reset link "
                    "has already been used "
                    "or is no longer valid."
                ),
            )

        new_password_hash = hash_password(
            new_password
        )

        conn.execute(
            """
            UPDATE users
            SET password = ?
            WHERE id = ?
            """,
            (
                new_password_hash,
                user_id,
            ),
        )

        conn.commit()

    return {
        "message": (
            "Password reset successful. "
            "You can now log in."
        )
    }

@app.get("/api/auth/me")
def auth_me(current_user: Dict[str, Any] = Depends(get_current_user)):
    return {"user": current_user}


@app.get("/api/chats")
def list_chats(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                id,
                user_id,
                title,
                messages_json,
                created_at,
                updated_at
            FROM chat
            WHERE user_id = ?
            ORDER BY updated_at DESC
            """,
            (current_user["id"],),
        ).fetchall()

    chats = []

    for row in rows:
        chats.append(
            {
                "id": row["id"],
                "user_id": row["user_id"],
                "title": row["title"],
                "messages": parse_messages_json(
                    row["messages_json"]
                ),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )

    return {"chats": chats}


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
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "feature_count": len(feature_names),
        "disease_count": (
            len(model.classes_)
            if model is not None
            else 0
        ),
        "recommendation_count": len(
            recommendation_lookup
        ),
        "critical_red_flags": len(
            CRITICAL_RED_FLAG_SYMPTOMS
        ),
        "major_red_flags": len(
            MAJOR_RED_FLAG_SYMPTOMS
        ),
        "shap_enabled": (
            ENABLE_SHAP
            and shap_explainer is not None
        ),
        "symptom_normalizer_loaded": (
            symptom_normalizer is not None
        ),
        "symptom_alias_count": (
            len(symptom_normalizer.alias_map)
            if symptom_normalizer is not None
            else 0
        ),
        "nearby_doctor_search": (
            "Google Maps text link only"
        ),
        "database_provider": DB_PROVIDER,
    }


@app.get("/api/symptoms")
def get_symptoms(
    q: Optional[str] = None,
    limit: int = 50,
):
    symptoms = feature_names

    if q:
        query = clean_symptom_text(q)

        symptoms = [
            symptom
            for symptom in symptoms
            if query in symptom
        ]

    return {
        "count": min(len(symptoms), limit),
        "symptoms": symptoms[:limit],
    }



def detect_explanation_request_language(
    message: str,
) -> Optional[str]:
    """
    Return:
        "bangla"  -> explain in Bangla script
        "english" -> explain in English
        None      -> not an explicit explanation request

    This does NOT depend on Gemini, so it still works
    when Gemini quota is temporarily unavailable.
    """

    raw = str(
        message or ""
    ).strip()

    text = (
        raw.lower()
        .replace("’", "'")
    )

    if not text:
        return None

    # --------------------------------------------------------
    # Bangla-script explanation requests
    # --------------------------------------------------------

    bangla_markers = [
        "বুঝিয়ে",
        "বুঝাই",
        "বুঝতে পারছি না",
        "বুঝতে পারলাম না",
        "বুঝলাম না",
        "বুঝি নাই",
        "বুঝি না",
        "ব্যাখ্যা",
        "সহজ করে",
        "মানে কি",
        "মানে কী",
    ]

    if any(
        marker in text
        for marker in bangla_markers
    ):
        return "bangla"

    # --------------------------------------------------------
    # Explicit English requests
    # --------------------------------------------------------

    english_markers = [
        "i can't understand",
        "i cant understand",
        "i cannot understand",
        "i don't understand",
        "i dont understand",
        "i do not understand",
        "explain this",
        "explain it",
        "explain the result",
        "explain this result",
        "can you explain",
        "could you explain",
        "what does this mean",
        "make it easier to understand",
    ]

    if any(
        marker in text
        for marker in english_markers
    ):
        return "english"

    # --------------------------------------------------------
    # Banglish requests -> response MUST be Bangla script
    # --------------------------------------------------------

    banglish_markers = [
        "amk bujai",
        "amake bujai",
        "bujai dao",
        "bujhai dao",
        "bujhaia dao",
        "bujay dao",
        "bujhi nai",
        "bujhi na",
        "bujhlam na",
        "bujlam na",
        "bujte partesi na",
        "bujhte partesi na",
        "bujhte parchi na",
        "easy kore bolo",
        "sohoj kore bolo",
        "banglay bolo",
        "bangla te bolo",
    ]

    if any(
        marker in text
        for marker in banglish_markers
    ):
        return "bangla"

    return None


def build_local_previous_result_explanation(
    previous_result: Dict[str, Any],
    language: str,
) -> str:
    """
    Natural, Gemini-style deterministic fallback.

    Important:
    - No new prediction
    - No invented symptoms
    - No invented SHAP values
    - Uses only the previous CareAI result
    """

    if not isinstance(previous_result, dict):

        if language == "bangla":
            return (
                "অবশ্যই বুঝিয়ে দিতে পারি। তবে আগে তোমার symptoms "
                "দিয়ে একটি prediction তৈরি করতে হবে। Result আসার "
                "পর বলো, ‘আমাকে এটা বুঝিয়ে দাও’।"
            )

        return (
            "Sure, I can explain it. First, generate a prediction "
            "from your symptoms, then ask me to explain the result."
        )

    matched = (
        previous_result.get("matched_symptoms")
        or previous_result.get("extracted_symptoms")
        or []
    )

    negated = (
        previous_result.get("negated_symptoms")
        or []
    )

    predictions = (
        previous_result.get("top_predictions")
        or []
    )

    red_flag = bool(
        previous_result.get("red_flag")
    )

    red_flag_result = (
        previous_result.get("red_flag_result")
        or {}
    )

    # ========================================================
    # RED FLAG
    # ========================================================

    if red_flag:

        triggered = (
            red_flag_result.get("triggered_symptoms")
            or []
        )

        severity = (
            red_flag_result.get("severity")
            or "serious"
        )

        symptoms_text = (
            ", ".join(triggered)
            if triggered
            else "serious symptoms"
        )

        if language == "bangla":

            return (
                "সহজভাবে বললে, CareAI তোমার আগের result-এ একটি "
                f"{severity} safety warning দেখিয়েছে। এর কারণ হলো "
                f"{symptoms_text} এর মতো symptom detect হয়েছে। "
                "এখানে system কোনো disease নিশ্চিতভাবে diagnose "
                "করেনি—বরং এই symptom pattern serious হতে পারে বলে "
                "normal prediction-এর আগে safety alert দিয়েছে। "
                "এই ধরনের red-flag result হলে medical evaluation "
                "delay না করাই ভালো।"
            )

        return (
            "In simple terms, CareAI triggered a "
            f"{severity} safety warning because it detected "
            f"{symptoms_text}. This does not mean a disease has "
            "been confirmed. The system raised the alert because "
            "this symptom pattern may require prompt medical "
            "assessment, so a red-flag warning should not be ignored."
        )

    # ========================================================
    # NO PREDICTION
    # ========================================================

    if not predictions:

        if language == "bangla":
            return (
                "আগের result-এ explain করার মতো কোনো disease "
                "prediction নেই। নতুন করে symptoms দিলে আমি সেই "
                "resultটা সহজভাবে বুঝিয়ে বলতে পারব।"
            )

        return (
            "There is no disease prediction in the previous result "
            "for me to explain. Enter your symptoms first, and I can "
            "then explain the result in simple terms."
        )

    # ========================================================
    # TOP PREDICTION
    # ========================================================

    top = predictions[0]

    disease = str(
        top.get("disease")
        or "Unknown condition"
    )

    confidence = top.get("confidence_percent")

    if confidence is None:
        try:
            confidence = round(
                float(top.get("confidence", 0)) * 100,
                2,
            )
        except Exception:
            confidence = 0

    try:
        confidence_num = float(confidence)
    except Exception:
        confidence_num = 0.0

    # ========================================================
    # SHAP
    # ========================================================

    shap_data = (
        top.get("shap_explanation")
        or {}
    )

    shap_items = (
        shap_data.get(
            "present_symptom_contributions"
        )
        or []
    )

    def contribution_value(item):
        try:
            return abs(
                float(
                    item.get(
                        "contribution",
                        item.get("shap_value", 0),
                    )
                )
            )
        except Exception:
            return 0.0

    shap_items = sorted(
        shap_items,
        key=contribution_value,
        reverse=True,
    )[:3]

    shap_names = [
        str(item.get("symptom")).strip()
        for item in shap_items
        if item.get("symptom")
    ]

    # ========================================================
    # OTHER PREDICTIONS
    # ========================================================

    other_predictions = []

    for pred in predictions[1:3]:

        name = pred.get("disease")

        if not name:
            continue

        percent = pred.get(
            "confidence_percent"
        )

        if percent is None:
            try:
                percent = round(
                    float(
                        pred.get(
                            "confidence",
                            0,
                        )
                    ) * 100,
                    2,
                )
            except Exception:
                percent = None

        if percent is not None:
            other_predictions.append(
                f"{name} ({percent}%)"
            )
        else:
            other_predictions.append(
                str(name)
            )

    # ========================================================
    # OPTIONAL RECOMMENDATION
    # ========================================================

    recommendation = (
        top.get("recommendation")
        or {}
    )

    doctor_type = (
        recommendation.get(
            "doctor_type_patient_should_see"
        )
        or recommendation.get("doctor_type")
        or recommendation.get("specialist")
    )

    tests = (
        recommendation.get(
            "common_tests_to_discuss_with_clinician"
        )
        or recommendation.get("tests")
        or []
    )

    if isinstance(tests, str):
        tests = [
            value.strip()
            for value in tests.split(";")
            if value.strip()
        ]

    tests = list(tests)[:3]

    # ========================================================
    # BANGLA NATURAL EXPLANATION
    # ========================================================

    if language == "bangla":

        parts = []

        parts.append(
            f"সহজভাবে বললে, তোমার দেওয়া symptomগুলো analyse করে "
            f"CareAI-এর model {disease}-কে সবচেয়ে উপরের সম্ভাব্য "
            f"condition হিসেবে দেখিয়েছে। এর model confidence "
            f"{confidence}%।"
        )

        if matched:

            parts.append(
                "এই result তৈরি করার সময় model মূলত "
                + ", ".join(matched)
                + " symptomগুলো consider করেছে।"
            )

        if negated:

            parts.append(
                "আর তুমি যেগুলো নেই বলে জানিয়েছিলে—"
                + ", ".join(negated)
                + "—সেগুলো present symptom হিসেবে prediction-এ "
                "ব্যবহার করা হয়নি।"
            )

        if shap_names:

            parts.append(
                "SHAP explanation অনুযায়ী "
                + ", ".join(shap_names)
                + " এই prediction-এর score-এ সবচেয়ে বেশি influence "
                "করেছে। সহজভাবে বলতে গেলে, model কেন এই result-এর "
                "দিকে গেছে সেটা বোঝার জন্য এগুলো গুরুত্বপূর্ণ "
                "signal ছিল।"
            )

        if confidence_num < 20:

            parts.append(
                "Confidence তুলনামূলকভাবে কম, তাই এই resultকে "
                "strong বা নিশ্চিত prediction হিসেবে ধরা উচিত নয়।"
            )

        elif confidence_num < 50:

            parts.append(
                "Confidence খুব বেশি নয়, তাই অন্য সম্ভাবনাগুলোও "
                "consider করা গুরুত্বপূর্ণ।"
            )

        if other_predictions:

            parts.append(
                "Model একই সাথে অন্য সম্ভাবনা হিসেবেও "
                + ", ".join(other_predictions)
                + " দেখিয়েছে।"
            )

        if doctor_type:

            parts.append(
                f"আরও নিশ্চিতভাবে evaluate করার প্রয়োজন হলে "
                f"recommendation অনুযায়ী {doctor_type}-এর সঙ্গে "
                "আলোচনা করা যেতে পারে।"
            )

        if tests:

            parts.append(
                "Clinician প্রয়োজন মনে করলে "
                + ", ".join(map(str, tests))
                + " এর মতো test নিয়ে আলোচনা করতে পারেন।"
            )

        parts.append(
            "সবচেয়ে গুরুত্বপূর্ণ বিষয় হলো—এটা কোনো confirmed "
            "diagnosis নয়। Confidence হলো model-এর prediction "
            "score, আর SHAP দেখায় কোন symptom model-এর decision-এ "
            "বেশি influence করেছে; এটি disease-এর কারণ প্রমাণ করে না।"
        )

        return " ".join(parts)

    # ========================================================
    # ENGLISH NATURAL EXPLANATION
    # ========================================================

    parts = []

    parts.append(
        f"In simple terms, after analysing the symptoms you provided, "
        f"CareAI ranked {disease} as the most likely condition in its "
        f"model output, with a model confidence of {confidence}%."
    )

    if matched:

        parts.append(
            "The model mainly used these present symptoms when making "
            "that prediction: "
            + ", ".join(matched)
            + "."
        )

    if negated:

        parts.append(
            "The symptoms you specifically said were absent—"
            + ", ".join(negated)
            + "—were not treated as present symptoms in the prediction."
        )

    if shap_names:

        parts.append(
            "According to the SHAP explanation, "
            + ", ".join(shap_names)
            + " had some of the strongest influence on this model "
            "score. In other words, they were important signals behind "
            "why the model leaned toward this result."
        )

    if confidence_num < 20:

        parts.append(
            "The confidence is relatively low, so this should not be "
            "treated as a strong or certain prediction."
        )

    elif confidence_num < 50:

        parts.append(
            "The confidence is not especially high, so the other "
            "possibilities should also be kept in mind."
        )

    if other_predictions:

        parts.append(
            "The model also listed "
            + ", ".join(other_predictions)
            + " as alternative possibilities."
        )

    if doctor_type:

        parts.append(
            f"If further clinical assessment is needed, the stored "
            f"CareAI recommendation suggests discussing the result "
            f"with a {doctor_type}."
        )

    if tests:

        parts.append(
            "A clinician may also consider discussing tests such as "
            + ", ".join(map(str, tests))
            + ", depending on the clinical situation."
        )

    parts.append(
        "Most importantly, this is not a confirmed diagnosis. "
        "The confidence is a model score, and SHAP only explains "
        "which symptoms influenced the model's decision; it does not "
        "prove that those symptoms caused a disease."
    )

    return " ".join(parts)


def build_previous_result_explanation_response(
    previous_result: Optional[Dict[str, Any]],
    user_message: str,
    language: str,
) -> Dict[str, Any]:

    explanation_source = "local_fallback"
    explanation = None

    if not previous_result:

        if language == "bangla":
            explanation = (
                "অবশ্যই বুঝিয়ে দিতে পারি। তবে আগে তোমার symptoms "
                "দিয়ে একটি prediction তৈরি করো। Result আসার পর "
                "বললেই আমি সহজভাবে বুঝিয়ে দেব।"
            )
        else:
            explanation = (
                "Sure, I can explain it. First generate a prediction "
                "from your symptoms, then ask me to explain the result."
            )

    else:

        # ----------------------------------------------------
        # Try Gemini first
        # ----------------------------------------------------

        if gemini_symptom_engine is not None:

            explanation = (
                gemini_symptom_engine
                .explain_previous_result(
                    previous_result=
                        previous_result,

                    language=
                        language,

                    user_message=
                        user_message,
                )
            )

            if explanation:
                explanation_source = "gemini"

        # ----------------------------------------------------
        # Gemini unavailable/quota/error -> natural local
        # ----------------------------------------------------

        if not explanation:

            explanation = (
                build_local_previous_result_explanation(
                    previous_result,
                    language,
                )
            )

            explanation_source = "local_fallback"

    return {
        "status":
            "explanation",

        "red_flag":
            False,

        "message":
            explanation,

        "extracted_symptoms":
            [],

        "matched_symptoms":
            [],

        "unmatched_symptoms":
            [],

        "red_flag_result":
            None,

        "top_predictions":
            [],

        "possible_symptoms":
            [],

        "negated_symptoms":
            [],

        "symptom_extraction":
            {
                "intent":
                    "explanation_request",

                "explanation_language":
                    language,

                "skip_prediction":
                    True,

                # True ONLY when Gemini actually generated
                # this explanation.
                "gemini_used":
                    explanation_source
                    == "gemini",

                # Gemini may be configured even when a
                # particular request falls back.
                "gemini_available":
                    gemini_symptom_engine
                    is not None,

                "explanation_source":
                    explanation_source,
            },
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    extracted_symptoms = []
    extraction_details = None
    possible_symptoms: List[Dict[str, Any]] = []
    negated_symptoms: List[str] = []

    # ========================================================
    # PREVIOUS RESULT EXPLANATION
    # ========================================================
    #
    # Known Bangla/Banglish/English explanation phrases are
    # handled BEFORE symptom extraction.
    #
    # This prevents:
    # "amk bujai dao"
    # from accidentally becoming a new symptom prediction.

    explanation_language = (
        detect_explanation_request_language(
            request.message or ""
        )
    )

    if explanation_language:

        return (
            build_previous_result_explanation_response(
                previous_result=
                    request.previous_result,

                user_message=
                    request.message or "",

                language=
                    explanation_language,
            )
        )

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

    # GEMINI_EXPLANATION_REQUEST
    if (
        extraction_details
        and extraction_details.get(
            "intent"
        )
        == "explanation_request"
    ):

        detected_language = (
            extraction_details.get(
                "language",
                "english",
            )
        )

        language = (
            "bangla"
            if detected_language
            in {
                "bangla",
                "banglish",
                "mixed",
            }
            else "english"
        )

        return (
            build_previous_result_explanation_response(
                previous_result=
                    request.previous_result,

                user_message=
                    request.message or "",

                language=
                    language,
            )
        )


    # HYBRID_INTENT_GATE
    if (
        extraction_details
        and extraction_details.get(
            "skip_prediction",
            False,
        )
    ):
        return {
            "status":
                "non_symptom",

            "red_flag":
                False,

            "message":
                extraction_details.get(
                    "assistant_message",
                    "No new symptom description was detected.",
                ),

            "extracted_symptoms":
                [],

            "matched_symptoms":
                [],

            "unmatched_symptoms":
                [],

            "red_flag_result":
                None,

            "top_predictions":
                [],

            "possible_symptoms":
                [],

            "negated_symptoms":
                [],

            "symptom_extraction":
                extraction_details,
        }


    # GEMINI_CLARIFICATION
    if (
        extraction_details
        and extraction_details.get(
            "clarification_needed",
            False,
        )
    ):
        # Gemini says the message is too ambiguous for a safe
        # canonical mapping.
        #
        # Do not allow fuzzy/semantic guesses from the old engine
        # to trigger a disease prediction.
        #
        # Safety exception:
        # if the current extraction already triggers an actual
        # CareAI red-flag rule, preserve the safety pathway.

        clarification_safety_check = (
            check_red_flag_rule(
                extracted_symptoms
            )
            if extracted_symptoms
            else {
                "red_flag": False
            }
        )

        if not clarification_safety_check.get(
            "red_flag",
            False,
        ):
            question = (
                extraction_details.get(
                    "follow_up_question"
                )
                or (
                    "Please describe your symptom "
                    "a little more clearly."
                )
            )

            # Keep these only for debugging.
            # They are NOT sent into disease prediction.
            extraction_details[
                "discarded_for_clarification"
            ] = list(
                extracted_symptoms
            )

            extraction_details[
                "clarification_overrode_local_matches"
            ] = bool(
                extracted_symptoms
            )

            extraction_details[
                "model_input"
            ] = []

            return {
                "status":
                    "clarification_needed",

                "red_flag":
                    False,

                "message":
                    question,

                "extracted_symptoms":
                    [],

                "matched_symptoms":
                    [],

                "unmatched_symptoms":
                    [],

                "red_flag_result":
                    None,

                "top_predictions":
                    [],

                "possible_symptoms":
                    possible_symptoms,

                "negated_symptoms":
                    negated_symptoms,

                "symptom_extraction":
                    extraction_details,
            }


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
