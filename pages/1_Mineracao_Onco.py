import json
import html
import hashlib
import os
import re
import sqlite3
import unicodedata
from difflib import SequenceMatcher
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import requests
import streamlit as st

try:
    from fuzzywuzzy import fuzz
except Exception:
    fuzz = None


MODEL_OPTIONS = {
    "OpenAI": ["gpt-4o-mini", "gpt-4.1-mini", "gpt-5-mini"],
    "DeepSeek": ["deepseek-chat", "deepseek-reasoner"],
    "Gemini": ["gemini-2.5-flash", "gemini-2.5-pro"],
    "LLM Studio": ["local-model"],
}

PROVIDER_ENV_KEYS = {
    "OpenAI": "OPENAI_API_KEY",
    "DeepSeek": "DEEPSEEK_API_KEY",
    "Gemini": "GEMINI_API_KEY",
    "LLM Studio": "",
}

PROVIDER_BASE_URLS = {
    "OpenAI": "https://api.openai.com/v1",
    "DeepSeek": "https://api.deepseek.com",
    "LLM Studio": os.getenv("LLM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
}

URGENCY_ORDER = ["CRITICA", "MUITO ALTA", "ALTA", "MODERADA", "BAIXA"]
ROW_BG_BY_URGENCY = {
    "CRITICA": "#4a1a1a",
    "MUITO ALTA": "#5a320f",
    "ALTA": "#5a510f",
    "MODERADA": "#163a59",
    "BAIXA": "#143d31",
}

SPECIALTY_BUCKETS = [
    "Neurologia",
    "Cabeca e pescoco",
    "Torax",
    "Abdome e pelve / urologia (medicina interna)",
    "Ginecologico (utero/ovarios)",
    "Mama (mamas e axilas)",
    "Obstetrico",
    "Musculoesqueletico",
]

EXCEL_NAME_CANDIDATES = ["nome", "nome_paciente", "paciente_nome_social", "paciente"]
EXCEL_CONVENIO_CANDIDATES = ["convenio", "plano_convenio"]
EXCEL_TELEFONE_CANDIDATES = ["telefone", "fone", "tel", "celular"]
EXCEL_SETOR_CANDIDATES = ["setor", "tipo_atendimento", "setor_executante", "departamento"]
EXCEL_SAME_CANDIDATES = ["same", "same_id", "sameid"]
APP_SETTINGS_FILENAME = "onco_app_settings.json"
CORRELATION_UPLOAD_STEM = "convenios_correlation_source"


def init_model_cache():
    if "models_by_provider" not in st.session_state:
        st.session_state["models_by_provider"] = {
            provider: list(models) for provider, models in MODEL_OPTIONS.items()
        }


def get_cached_models(provider):
    init_model_cache()
    return st.session_state["models_by_provider"].get(provider, list(MODEL_OPTIONS.get(provider, [])))


def set_cached_models(provider, models):
    init_model_cache()
    if not models:
        st.session_state["models_by_provider"][provider] = list(MODEL_OPTIONS.get(provider, []))
    else:
        unique_models = []
        seen = set()
        for model in models:
            model_name = normalize_text(model)
            if model_name and model_name not in seen:
                unique_models.append(model_name)
                seen.add(model_name)
        st.session_state["models_by_provider"][provider] = unique_models or list(MODEL_OPTIONS.get(provider, []))


def specialty_chip(name):
    key = ascii_fold(name).lower()
    if "neurolog" in key:
        return "🧠"
    if "cabeca" in key or "pescoco" in key:
        return "🗣️"
    if "torax" in key:
        return "🫁"
    if "abdome" in key or "pelve" in key or "urologia" in key:
        return "🟠"
    if "ginecologico" in key:
        return "🩺"
    if "mama" in key:
        return "🌸"
    if "obstetrico" in key:
        return "🤰"
    if "musculoesqueletico" in key:
        return "🦴"
    if "onco" in key:
        return "🔴"
    return "⚪"


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip()


def esc(value):
    return html.escape(normalize_text(value))


def ascii_fold(value):
    text = normalize_text(value)
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def get_db_path():
    env_path = os.getenv("DB_PATH", "").strip()
    if env_path:
        return env_path
    return str(Path.cwd() / "tumor_findings_patients.db")


def hash_password(password):
    return hashlib.sha256(normalize_text(password).encode("utf-8")).hexdigest()


def now_display():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


def get_app_storage_dir(db_path):
    return Path(db_path).expanduser().resolve().parent


def get_app_settings_path(db_path):
    return get_app_storage_dir(db_path) / APP_SETTINGS_FILENAME


def default_app_settings():
    default_provider = "OpenAI"
    default_model = MODEL_OPTIONS[default_provider][0]
    env_key = PROVIDER_ENV_KEYS.get(default_provider, "")
    default_api_key = os.getenv(env_key, "") if env_key else ""
    return {
        "users": {
            "admin": {
                "username": "admin",
                "password_hash": hash_password(os.getenv("ONCO_ADMIN_PASSWORD", "admin123")),
                "role": "admin",
                "label": "Administrador",
                "aliases": ["admin", "administrador", "admin123"],
            },
            "common": {
                "username": "usuario",
                "password_hash": hash_password(os.getenv("ONCO_COMMON_PASSWORD", "usuario123")),
                "role": "common",
                "label": "Usuario comum",
                "aliases": ["usuario", "usuario123", "comum"],
            },
        },
        "admin_runtime": {
            "provider": default_provider,
            "api_key": default_api_key,
            "validated_at": "",
            "available_models": {},
        },
        "common_ai_config": {
            "provider": default_provider,
            "model": default_model,
            "api_key": default_api_key,
            "configured_at": "",
            "configured_by": "",
        },
        "correlation": {
            "stored_excel_path": "",
            "original_name": "",
            "uploaded_at": "",
            "uploaded_by": "",
            "threshold": 70,
        },
    }


def merge_settings_with_defaults(current, defaults):
    if not isinstance(current, dict):
        return defaults

    merged = {}
    for key, default_value in defaults.items():
        current_value = current.get(key)
        if isinstance(default_value, dict):
            merged[key] = merge_settings_with_defaults(current_value, default_value)
        else:
            merged[key] = current_value if current_value not in (None, "") else default_value

    for key, value in current.items():
        if key not in merged:
            merged[key] = value
    return merged


def load_app_settings(db_path):
    settings_path = get_app_settings_path(db_path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    defaults = default_app_settings()

    if settings_path.exists():
        try:
            loaded = json.loads(settings_path.read_text(encoding="utf-8"))
        except Exception:
            loaded = {}
    else:
        loaded = {}

    settings = merge_settings_with_defaults(loaded, defaults)
    if settings != loaded:
        save_app_settings(db_path, settings)
    return settings


def save_app_settings(db_path, settings):
    settings_path = get_app_settings_path(db_path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, ensure_ascii=True, indent=2), encoding="utf-8")


def authenticate_user(settings, username, password):
    username_norm = normalize_text(username).lower()
    password_hash = hash_password(password)
    for user in settings.get("users", {}).values():
        aliases = [normalize_text(user.get("username")).lower()]
        aliases.extend(normalize_text(alias).lower() for alias in user.get("aliases", []))
        aliases = [alias for alias in aliases if alias]
        if username_norm not in aliases:
            continue
        if normalize_text(user.get("password_hash")) != password_hash:
            return None
        return {
            "username": user.get("username", ""),
            "role": user.get("role", "common"),
            "label": user.get("label", user.get("username", "")),
        }
    return None


def init_auth_state():
    if "auth_user" not in st.session_state:
        st.session_state["auth_user"] = None


def render_login_screen(db_path):
    init_auth_state()
    if st.session_state.get("auth_user"):
        return st.session_state["auth_user"]

    settings = load_app_settings(db_path)
    left, center, right = st.columns([1.2, 1.4, 1.2])
    with center:
        st.markdown("### Login")
        with st.form("onco_login_form", clear_on_submit=False):
            username = st.text_input("Usuario")
            password = st.text_input("Senha", type="password")
            submitted = st.form_submit_button("Entrar", use_container_width=True)

        if submitted:
            user = authenticate_user(settings, username, password)
            if user is None:
                st.error("Usuario ou senha invalidos.")
            else:
                st.session_state["auth_user"] = user
                st.rerun()

    st.stop()


def logout_user():
    st.session_state["auth_user"] = None
    st.rerun()


def get_saved_common_ai_config(settings):
    return settings.get("common_ai_config", {})


def get_saved_correlation_info(settings):
    return settings.get("correlation", {})


def get_stored_correlation_file_path(settings, db_path):
    stored_path = normalize_text(get_saved_correlation_info(settings).get("stored_excel_path"))
    if not stored_path:
        return None
    path = Path(stored_path)
    if not path.is_absolute():
        path = get_app_storage_dir(db_path) / path
    return path if path.exists() else None


def store_correlation_excel(db_path, uploaded_excel, uploaded_by, threshold):
    storage_dir = get_app_storage_dir(db_path)
    storage_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(normalize_text(getattr(uploaded_excel, "name", "")) or ".xlsx").suffix or ".xlsx"
    target = storage_dir / f"{CORRELATION_UPLOAD_STEM}{suffix}"
    target.write_bytes(uploaded_excel.getvalue())

    settings = load_app_settings(db_path)
    settings["correlation"] = {
        "stored_excel_path": str(target),
        "original_name": normalize_text(getattr(uploaded_excel, "name", target.name)),
        "uploaded_at": now_display(),
        "uploaded_by": normalize_text(uploaded_by),
        "threshold": int(threshold),
    }
    save_app_settings(db_path, settings)
    return settings["correlation"]


def correlate_patients_with_stored_excel(db_path, settings):
    stored_path = get_stored_correlation_file_path(settings, db_path)
    if stored_path is None:
        raise RuntimeError("Nenhuma planilha armazenada pelo administrador.")

    threshold = int(get_saved_correlation_info(settings).get("threshold", 70))
    with open(stored_path, "rb") as fp:
        return correlate_patients_with_excel_upload(db_path, fp, threshold)


def ensure_schema(db_path):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                same_id TEXT PRIMARY KEY,
                patient_name TEXT,
                birth_date TEXT,
                age TEXT,
                last_exam_date TEXT,
                last_file TEXT,
                context TEXT,
                full_text TEXT,
                ai_analysis TEXT,
                ai_model TEXT,
                is_eligible INTEGER DEFAULT 0,
                convenio TEXT,
                telefone TEXT,
                setor TEXT,
                endereco TEXT,
                exam_title TEXT,
                exam_modality TEXT,
                medical_specialty TEXT,
                tumor_findings TEXT,
                tumor_location TEXT,
                tumor_characteristics TEXT,
                malignancy_score INTEGER DEFAULT 0,
                urgency_level TEXT DEFAULT 'BAIXA',
                urgency_reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TRIGGER IF NOT EXISTS update_patients_timestamp
            AFTER UPDATE ON patients
            BEGIN
                UPDATE patients SET updated_at = CURRENT_TIMESTAMP
                WHERE same_id = NEW.same_id;
            END;
            """
        )
        conn.commit()
    finally:
        conn.close()


def get_pdf_reader_class():
    try:
        from pypdf import PdfReader

        return PdfReader
    except Exception:
        try:
            from PyPDF2 import PdfReader

            return PdfReader
        except Exception as exc:
            raise RuntimeError(
                "Nenhum leitor de PDF instalado. Instale pypdf ou PyPDF2 e faca redeploy."
            ) from exc


def extract_pdf_text(uploaded_file):
    PdfReader = get_pdf_reader_class()
    reader = PdfReader(uploaded_file)
    chunks = []
    for page in reader.pages:
        text = page.extract_text() or ""
        chunks.append(text)
    return "\n".join(chunks).strip()


def parse_same_id_fallback(text):
    patterns = [
        r"\bSAME\s*[:\-]?\s*([A-Z0-9]{4,})",
        r"\bID\s*PACIENTE\s*[:\-]?\s*([A-Z0-9]{4,})",
        r"\bPRONTUARIO\s*[:\-]?\s*([A-Z0-9]{4,})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return f"AUTO-{uuid4().hex[:12].upper()}"


def parse_patient_name_fallback(text):
    patterns = [
        r"PACIENTE\s*[:\-]\s*([A-ZA-Z\s]{6,})",
        r"NOME\s*[:\-]\s*([A-ZA-Z\s]{6,})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = re.sub(r"\s+", " ", match.group(1)).strip(" -:")
            if len(name) >= 3:
                return name.title()
    return "Paciente nao identificado"


def parse_exam_date_fallback(text):
    match = re.search(r"\b(\d{2}[\/\-]\d{2}[\/\-]\d{4})\b", text)
    return format_exam_date(match.group(1)) if match else datetime.utcnow().strftime("%d/%m/%Y")


def parse_exam_datetime(value):
    if isinstance(value, pd.Timestamp):
        return value.normalize()

    text = normalize_text(value)
    if not text:
        return pd.NaT

    for fmt in (
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d.%m.%Y",
        "%Y-%m-%d",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ):
        try:
            return pd.Timestamp(datetime.strptime(text, fmt)).normalize()
        except ValueError:
            continue

    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.notna(parsed):
        return pd.Timestamp(parsed).normalize()
    return pd.NaT


def format_exam_date(value):
    parsed = parse_exam_datetime(value)
    if pd.notna(parsed):
        return parsed.strftime("%d/%m/%Y")
    return normalize_text(value)


def parse_age_fallback(text):
    match = re.search(r"\b(\d{1,3})\s*ANOS\b", ascii_fold(text).upper())
    if match:
        return match.group(1)
    return ""


def infer_modality_fallback(text):
    upper = ascii_fold(text).upper()
    if "RESSON" in upper or re.search(r"\bRM\b", upper):
        return "RM"
    if "TOMOGRAF" in upper or re.search(r"\bTC\b", upper):
        return "TC"
    if "PET" in upper:
        return "PET"
    if "MAMOGRAF" in upper:
        return "MAMOGRAFIA"
    return "RADIOLOGIA"


def infer_specialty_fallback(text):
    return canonical_specialty("", text)


def canonical_specialty(raw_specialty, source_text=""):
    text = ascii_fold(f"{raw_specialty} {source_text}").upper()

    if any(t in text for t in ["OBST", "GESTA", "FETO", "PLACENTA", "GRAVID"]):
        return "Obstetrico"
    if any(t in text for t in ["MAMA", "MAMAR", "AXILA", "BIRADS", "BI-RADS"]):
        return "Mama (mamas e axilas)"
    if any(t in text for t in ["UTERO", "OVARIO", "ENDOMET", "ADNEX", "GINECO", "ANEXIAL"]):
        return "Ginecologico (utero/ovarios)"
    if any(t in text for t in ["ENCEF", "CEREBR", "NEURO", "INTRACRAN", "CRANIO", "SNC"]):
        return "Neurologia"
    if any(t in text for t in ["PESCOCO", "LARING", "OROFAR", "NASO", "SEIOS PARANASAIS", "TIREOID", "MANDIB", "FACE"]):
        return "Cabeca e pescoco"
    if any(t in text for t in ["TORAX", "PULMAO", "PULMON", "MEDIAST", "PLEURA", "CARDIO", "MAMARIA INTERNA"]):
        return "Torax"
    if any(t in text for t in ["OSSO", "ARTIC", "MUSCUL", "TEND", "LIGAMENT", "COLUNA", "ORTOP", "ESQUELET"]):
        return "Musculoesqueletico"
    if any(t in text for t in ["ABDOM", "PELVE", "HEPAT", "FIGADO", "PANCREA", "RENAL", "RIM", "PROSTAT", "BEXIGA", "URO", "GASTRO", "INTEST", "BACO", "ADREN", "VESIC"]):
        return "Abdome e pelve / urologia (medicina interna)"

    # Fallback para garantir somente os grupos definidos.
    return "Abdome e pelve / urologia (medicina interna)"


def normalize_urgency(value):
    text = ascii_fold(value).upper()
    alias = {
        "CRITICA": "CRITICA",
        "MUITO ALTA": "MUITO ALTA",
        "ALTA": "ALTA",
        "MODERADA": "MODERADA",
        "BAIXA": "BAIXA",
    }
    return alias.get(text, "MODERADA")


def normalize_score(value):
    try:
        score = int(float(value))
    except Exception:
        score = 0
    return max(0, min(5, score))


def normalize_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return int(value) == 1
    text = ascii_fold(value).lower()
    return text in {"1", "true", "sim", "yes", "y"}


def normalize_excel_column_name(value):
    text = ascii_fold(value).lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalize_same(value):
    text = normalize_text(value)
    if not text:
        return ""
    only_digits = re.sub(r"\D", "", text)
    if len(only_digits) >= 4:
        return only_digits
    folded = ascii_fold(text).upper()
    return re.sub(r"[^A-Z0-9]", "", folded)


def normalize_name_for_match(value):
    text = ascii_fold(value).upper().strip()
    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    text = " ".join(text.split())
    return text


def token_set_similarity(a, b):
    if not a or not b:
        return 0.0
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    if intersection == 0:
        return 0.0
    return (2.0 * intersection / (len(tokens_a) + len(tokens_b))) * 100.0


def partial_similarity(a, b):
    if not a or not b:
        return 0.0
    small, big = (a, b) if len(a) <= len(b) else (b, a)
    if small in big:
        return 100.0
    return SequenceMatcher(None, small, big).ratio() * 100.0


def similarity_scores(a, b):
    if fuzz is not None:
        return (
            float(fuzz.ratio(a, b)),
            float(fuzz.token_set_ratio(a, b)),
            float(fuzz.partial_ratio(a, b)),
        )
    return (
        SequenceMatcher(None, a, b).ratio() * 100.0,
        token_set_similarity(a, b),
        partial_similarity(a, b),
    )


def find_best_name_match(patient_name, excel_names_normalized, threshold):
    clean_name = normalize_name_for_match(patient_name)
    if not clean_name:
        return None

    for idx, excel_name in enumerate(excel_names_normalized):
        if excel_name and excel_name == clean_name:
            return idx, 100.0

    best_idx = None
    best_score = -1.0
    for idx, excel_name in enumerate(excel_names_normalized):
        if not excel_name:
            continue
        score_ratio, score_token, score_partial = similarity_scores(clean_name, excel_name)
        score = max(score_ratio, score_token, score_partial)
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx is not None and best_score >= float(threshold):
        return best_idx, best_score
    return None


def read_excel_for_correlation(uploaded_excel):
    name = normalize_text(getattr(uploaded_excel, "name", ""))
    lower = name.lower()
    if lower.endswith(".xls"):
        try:
            return pd.read_excel(uploaded_excel, engine="xlrd")
        except Exception:
            uploaded_excel.seek(0)
    return pd.read_excel(uploaded_excel)


def normalize_excel_for_correlation(df_excel):
    if df_excel is None or df_excel.empty:
        raise RuntimeError("Arquivo Excel vazio.")

    normalized_columns = [normalize_excel_column_name(c) for c in df_excel.columns]
    if len(set(normalized_columns)) != len(normalized_columns):
        used = {}
        deduped = []
        for col in normalized_columns:
            count = used.get(col, 0) + 1
            used[col] = count
            deduped.append(col if count == 1 else f"{col}_{count}")
        normalized_columns = deduped

    work = df_excel.copy()
    work.columns = normalized_columns

    def pick_col(candidates):
        return next((c for c in candidates if c in work.columns), None)

    name_col = pick_col(EXCEL_NAME_CANDIDATES)
    convenio_col = pick_col(EXCEL_CONVENIO_CANDIDATES)
    telefone_col = pick_col(EXCEL_TELEFONE_CANDIDATES)
    setor_col = pick_col(EXCEL_SETOR_CANDIDATES)
    same_col = pick_col(EXCEL_SAME_CANDIDATES)

    if not name_col:
        raise RuntimeError(
            "Coluna de nome nao encontrada no Excel. Use uma das colunas: "
            + ", ".join(EXCEL_NAME_CANDIDATES)
        )

    if not convenio_col and not telefone_col and not setor_col:
        raise RuntimeError(
            "Nenhuma coluna de convenio/telefone/setor encontrada no Excel."
        )

    work["nome"] = work[name_col]
    work["convenio"] = work[convenio_col] if convenio_col else None
    if "plano_convenio" in work.columns:
        convenio_text = work["convenio"].astype(str).str.strip()
        fallback_mask = work["convenio"].isna() | (convenio_text == "")
        work.loc[fallback_mask, "convenio"] = work.loc[fallback_mask, "plano_convenio"]
    work["telefone"] = work[telefone_col] if telefone_col else None
    work["setor"] = work[setor_col] if setor_col else None
    work["same"] = work[same_col] if same_col else None

    work["nome_norm"] = work["nome"].apply(normalize_name_for_match)
    work["same_norm"] = work["same"].apply(normalize_same)
    work = work[(work["nome_norm"] != "") | (work["same_norm"] != "")].reset_index(drop=True)
    if work.empty:
        raise RuntimeError("Arquivo Excel sem linhas validas para correlacao.")
    return work


def correlate_patients_with_excel_upload(db_path, uploaded_excel, threshold=70):
    ensure_schema(db_path)
    df_excel_raw = read_excel_for_correlation(uploaded_excel)
    df_excel = normalize_excel_for_correlation(df_excel_raw)

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT same_id, patient_name, convenio, telefone, setor FROM patients")
        patients = cursor.fetchall()

        total_patients = len(patients)
        if total_patients == 0:
            return {
                "success": True,
                "total_patients": 0,
                "matched_patients": 0,
                "updated_patients": 0,
                "unmatched_patients": 0,
                "match_rate": 0.0,
                "updated_counts": {"convenio": 0, "telefone": 0, "setor": 0},
            }

        same_index_map = {}
        for idx, same_norm in enumerate(df_excel["same_norm"].tolist()):
            if same_norm and same_norm not in same_index_map:
                same_index_map[same_norm] = idx

        excel_names_normalized = df_excel["nome_norm"].tolist()
        matched_patients = 0
        updated_patients = 0
        unmatched_patients = 0
        updated_counts = {"convenio": 0, "telefone": 0, "setor": 0}

        for same_id, patient_name, current_convenio, current_telefone, current_setor in patients:
            if not normalize_text(patient_name):
                unmatched_patients += 1
                continue

            best_idx = None

            same_norm = normalize_same(same_id)
            if same_norm and same_norm in same_index_map:
                best_idx = same_index_map[same_norm]
            else:
                name_match = find_best_name_match(patient_name, excel_names_normalized, threshold)
                if name_match is not None:
                    best_idx, _ = name_match

            if best_idx is None:
                unmatched_patients += 1
                continue

            matched_patients += 1
            excel_row = df_excel.iloc[best_idx]
            excel_convenio = normalize_text(excel_row.get("convenio"))
            excel_telefone = normalize_text(excel_row.get("telefone"))
            excel_setor = normalize_text(excel_row.get("setor"))

            update_fields = {}
            if excel_convenio and normalize_text(current_convenio) != excel_convenio:
                update_fields["convenio"] = excel_convenio
                updated_counts["convenio"] += 1
            if excel_telefone and normalize_text(current_telefone) != excel_telefone:
                update_fields["telefone"] = excel_telefone
                updated_counts["telefone"] += 1
            if excel_setor and normalize_text(current_setor) != excel_setor:
                update_fields["setor"] = excel_setor
                updated_counts["setor"] += 1

            if not update_fields:
                continue

            query = "UPDATE patients SET " + ", ".join([f"{k} = ?" for k in update_fields.keys()]) + " WHERE same_id = ?"
            values = list(update_fields.values()) + [same_id]
            cursor.execute(query, values)
            updated_patients += 1

        conn.commit()

        return {
            "success": True,
            "total_patients": total_patients,
            "matched_patients": matched_patients,
            "updated_patients": updated_patients,
            "unmatched_patients": unmatched_patients,
            "match_rate": (matched_patients / total_patients) * 100.0 if total_patients else 0.0,
            "updated_counts": updated_counts,
        }
    finally:
        conn.close()


def extract_json_block(text):
    if not text:
        return None
    cleaned = text.strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, flags=re.IGNORECASE)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1).strip())
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return None


def list_openai_compatible_models(base_url, api_key):
    endpoint = base_url.rstrip("/") + "/models"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.get(endpoint, headers=headers, timeout=60)
    if response.status_code >= 400:
        raise RuntimeError(f"Falha ao listar modelos ({response.status_code}): {response.text[:400]}")

    payload = response.json()
    data = payload.get("data", [])
    models = []
    for item in data:
        if isinstance(item, dict):
            model_id = normalize_text(item.get("id", ""))
            if model_id:
                model_id_lower = model_id.lower()
                blocked_tokens = [
                    "embedding",
                    "whisper",
                    "tts",
                    "transcribe",
                    "moderation",
                    "omni-moderation",
                    "dall",
                    "image",
                    "audio",
                ]
                if any(token in model_id_lower for token in blocked_tokens):
                    continue
                models.append(model_id)
    return sorted(set(models))


def list_gemini_models(api_key):
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY nao informado.")

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    response = requests.get(endpoint, timeout=60)
    if response.status_code >= 400:
        raise RuntimeError(f"Falha ao listar modelos Gemini ({response.status_code}): {response.text[:400]}")

    payload = response.json()
    models = []
    for item in payload.get("models", []):
        if not isinstance(item, dict):
            continue
        methods = item.get("supportedGenerationMethods", []) or []
        if methods and "generateContent" not in methods:
            continue
        name = normalize_text(item.get("name", ""))
        if name.startswith("models/"):
            name = name.split("/", 1)[1]
        if name:
            models.append(name)
    return sorted(set(models))


def fetch_models_for_provider(provider, api_key):
    if provider == "Gemini":
        return list_gemini_models(api_key)

    if provider in ("OpenAI", "DeepSeek", "LLM Studio"):
        if provider != "LLM Studio" and not api_key:
            raise RuntimeError("API key obrigatoria para listar modelos.")
        return list_openai_compatible_models(PROVIDER_BASE_URLS[provider], api_key)

    raise RuntimeError(f"Provider nao suportado: {provider}")


def ai_system_prompt():
    return (
        "Voce e um especialista em radiologia oncologica. "
        "Extraia dados do laudo e responda EXCLUSIVAMENTE em JSON valido, sem markdown. "
        "Campos obrigatorios: same_id, patient_name, age, last_exam_date, exam_modality, "
        "medical_specialty, tumor_findings, tumor_location, tumor_characteristics, "
        "malignancy_score, urgency_level, urgency_reason, is_eligible. "
        "medical_specialty deve ser UMA destas opcoes: "
        "Neurologia, Cabeca e pescoco, Torax, Abdome e pelve / urologia (medicina interna), "
        "Ginecologico (utero/ovarios), Mama (mamas e axilas), Obstetrico, Musculoesqueletico. "
        "Regras: malignancy_score deve ser inteiro 0-5; urgency_level deve ser CRITICA, MUITO ALTA, ALTA, MODERADA ou BAIXA; "
        "is_eligible deve ser booleano. Se dado ausente, use string vazia."
    )


def call_openai_compatible(base_url, api_key, model, report_text):
    endpoint = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "temperature": 1,
        "messages": [
            {"role": "system", "content": ai_system_prompt()},
            {
                "role": "user",
                "content": (
                    "Analise o laudo abaixo e retorne apenas JSON.\n\n"
                    + report_text[:120000]
                ),
            },
        ],
        "response_format": {"type": "json_object"},
    }

    response = None
    for _ in range(4):
        response = requests.post(endpoint, headers=headers, json=payload, timeout=150)
        if response.status_code < 400:
            break

        error_txt = response.text.lower()
        changed = False

        if "temperature" in error_txt and "temperature" in payload:
            payload.pop("temperature", None)
            changed = True

        if "response_format" in error_txt and "response_format" in payload:
            payload.pop("response_format", None)
            changed = True

        if not changed:
            break

    if response is None or response.status_code >= 400:
        status_code = response.status_code if response is not None else "sem_status"
        error_body = response.text[:500] if response is not None else "sem resposta"
        raise RuntimeError(f"Falha na API ({status_code}): {error_body}")

    data = response.json()
    return data["choices"][0]["message"]["content"]


def call_gemini(api_key, model, report_text):
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY nao informado.")

    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )
    prompt = (
        ai_system_prompt()
        + "\n\nLAUDO:\n"
        + report_text[:120000]
        + "\n\nRetorne apenas JSON valido."
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 1},
    }

    response = requests.post(endpoint, json=payload, timeout=150)
    if response.status_code >= 400 and "temperature" in response.text.lower():
        payload.pop("generationConfig", None)
        response = requests.post(endpoint, json=payload, timeout=150)

    if response.status_code >= 400:
        raise RuntimeError(f"Falha na API Gemini ({response.status_code}): {response.text[:500]}")

    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini retornou sem candidates.")

    parts = candidates[0].get("content", {}).get("parts", [])
    text_parts = [normalize_text(part.get("text", "")) for part in parts if isinstance(part, dict)]
    raw = "\n".join([t for t in text_parts if t])
    if not raw:
        raise RuntimeError("Gemini retornou resposta vazia.")
    return raw


def call_ai(provider, model, api_key, report_text):
    if provider == "Gemini":
        return call_gemini(api_key, model, report_text)

    if provider in ("OpenAI", "DeepSeek", "LLM Studio"):
        base_url = PROVIDER_BASE_URLS[provider]
        return call_openai_compatible(base_url, api_key, model, report_text)

    raise RuntimeError(f"Provider nao suportado: {provider}")


def build_ai_analysis_from_payload(payload):
    conclusion = "ELEGIVEL" if payload["is_eligible"] else "NAO ELEGIVEL"
    return (
        f"**MODALIDADE DO EXAME**: {payload['exam_modality']}\n"
        f"**ESPECIALIDADE MEDICA**: {payload['medical_specialty']}\n"
        f"**ACHADOS**: {payload['tumor_findings']}\n"
        f"**ESCORE DE MALIGNIDADE**: {payload['malignancy_score']}\n"
        f"URGENCIA: {payload['urgency_level']}\n"
        f"MOTIVO DA URGENCIA: {payload['urgency_reason']}\n"
        f"CONCLUSAO: {conclusion}"
    )


def normalize_ai_payload(ai_dict, source_text, file_name, provider, model):
    same_id = normalize_text(ai_dict.get("same_id")) or parse_same_id_fallback(source_text)
    patient_name = normalize_text(ai_dict.get("patient_name")) or parse_patient_name_fallback(source_text)
    age = normalize_text(ai_dict.get("age")) or parse_age_fallback(source_text)
    last_exam_date = format_exam_date(
        normalize_text(ai_dict.get("last_exam_date")) or parse_exam_date_fallback(source_text)
    )
    exam_modality = normalize_text(ai_dict.get("exam_modality")) or infer_modality_fallback(source_text)
    medical_specialty_raw = normalize_text(ai_dict.get("medical_specialty")) or infer_specialty_fallback(source_text)
    medical_specialty = canonical_specialty(medical_specialty_raw, source_text)
    tumor_findings = normalize_text(ai_dict.get("tumor_findings"))
    tumor_location = normalize_text(ai_dict.get("tumor_location"))
    tumor_characteristics = normalize_text(ai_dict.get("tumor_characteristics"))
    malignancy_score = normalize_score(ai_dict.get("malignancy_score"))
    urgency_level = normalize_urgency(ai_dict.get("urgency_level"))
    urgency_reason = normalize_text(ai_dict.get("urgency_reason"))

    is_eligible = normalize_bool(ai_dict.get("is_eligible"))
    if "is_eligible" not in ai_dict:
        is_eligible = malignancy_score >= 2

    payload = {
        "same_id": same_id,
        "patient_name": patient_name,
        "age": age,
        "last_exam_date": last_exam_date,
        "last_file": file_name,
        "context": json.dumps(
            {
                "source": "streamlit_mineracao_onco_ai",
                "provider": provider,
                "model": model,
                "processed_at": datetime.utcnow().isoformat(),
            },
            ensure_ascii=True,
        ),
        "full_text": source_text[:250000],
        "ai_analysis": "",
        "ai_model": f"{provider}:{model}",
        "is_eligible": 1 if is_eligible else 0,
        "exam_title": file_name,
        "exam_modality": exam_modality,
        "medical_specialty": medical_specialty,
        "tumor_findings": tumor_findings,
        "tumor_location": tumor_location,
        "tumor_characteristics": tumor_characteristics,
        "malignancy_score": malignancy_score,
        "urgency_level": urgency_level,
        "urgency_reason": urgency_reason,
    }
    payload["ai_analysis"] = build_ai_analysis_from_payload(payload)
    return payload


def upsert_patient(db_path, data):
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO patients (
                same_id, patient_name, age, last_exam_date, last_file, context,
                full_text, ai_analysis, ai_model, is_eligible, exam_title,
                exam_modality, medical_specialty, tumor_findings, tumor_location,
                tumor_characteristics, malignancy_score, urgency_level, urgency_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(same_id) DO UPDATE SET
                patient_name = excluded.patient_name,
                age = excluded.age,
                last_exam_date = excluded.last_exam_date,
                last_file = excluded.last_file,
                context = excluded.context,
                full_text = excluded.full_text,
                ai_analysis = excluded.ai_analysis,
                ai_model = excluded.ai_model,
                is_eligible = excluded.is_eligible,
                exam_title = excluded.exam_title,
                exam_modality = excluded.exam_modality,
                medical_specialty = excluded.medical_specialty,
                tumor_findings = excluded.tumor_findings,
                tumor_location = excluded.tumor_location,
                tumor_characteristics = excluded.tumor_characteristics,
                malignancy_score = excluded.malignancy_score,
                urgency_level = excluded.urgency_level,
                urgency_reason = excluded.urgency_reason,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                data["same_id"],
                data["patient_name"],
                data["age"],
                data["last_exam_date"],
                data["last_file"],
                data["context"],
                data["full_text"],
                data["ai_analysis"],
                data["ai_model"],
                data["is_eligible"],
                data["exam_title"],
                data["exam_modality"],
                data["medical_specialty"],
                data["tumor_findings"],
                data["tumor_location"],
                data["tumor_characteristics"],
                data["malignancy_score"],
                data["urgency_level"],
                data["urgency_reason"],
            ),
        )
        conn.commit()
    finally:
        conn.close()


def process_pdf_with_ai(uploaded_file, db_path, provider, model, api_key):
    report_text = extract_pdf_text(uploaded_file)
    if not report_text:
        raise RuntimeError("PDF sem texto extraivel.")

    raw_ai = call_ai(provider, model, api_key, report_text)
    ai_dict = extract_json_block(raw_ai)
    if not isinstance(ai_dict, dict):
        raise RuntimeError(f"IA retornou formato invalido: {raw_ai[:500]}")

    payload = normalize_ai_payload(ai_dict, report_text, uploaded_file.name, provider, model)
    upsert_patient(db_path, payload)

    return {
        "urgencia": payload["urgency_level"],
        "score": f"{payload['malignancy_score']}/5",
        "same": payload["same_id"],
        "nome": payload["patient_name"],
        "idade": payload["age"],
        "data_exame": format_exam_date(payload["last_exam_date"]),
        "modalidade": payload["exam_modality"],
        "especialidade": payload["medical_specialty"],
        "modelo_ia": payload["ai_model"],
        "elegivel": "sim" if payload["is_eligible"] else "nao",
    }


def save_uploaded_db(uploaded_file, db_path):
    target = Path(db_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(f"{db_path}.tmp")
    with open(tmp_path, "wb") as fp:
        fp.write(uploaded_file.getvalue())

    conn = None
    try:
        conn = sqlite3.connect(str(tmp_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='patients'")
        if cursor.fetchone() is None:
            raise RuntimeError("Arquivo invalido: tabela 'patients' nao encontrada.")
    finally:
        if conn is not None:
            conn.close()

    os.replace(str(tmp_path), str(target))


@st.cache_data(show_spinner=False)
def load_patients_from_db(db_path, mtime_token):
    _ = mtime_token
    conn = sqlite3.connect(db_path)
    try:
        query = """
            SELECT
                same_id,
                patient_name,
                age,
                last_exam_date,
                exam_modality,
                medical_specialty,
                malignancy_score,
                urgency_level,
                ai_model,
                is_eligible,
                ai_analysis,
                full_text,
                tumor_findings,
                tumor_location,
                tumor_characteristics,
                urgency_reason,
                convenio,
                setor,
                last_file,
                context,
                updated_at
            FROM patients
            ORDER BY updated_at DESC
        """
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


def build_results_dataframe(df, only_eligible):
    if df.empty:
        return pd.DataFrame(
            columns=[
                "URGENCIA",
                "SCORE MALIG.",
                "SAME",
                "NOME",
                "IDADE",
                "DATA EXAME",
                "MODALIDADE",
                "CONVENIO",
                "SETOR",
                "ESPECIALIDADE",
                "MODELO IA",
            ]
        )

    work = df.copy()
    work["is_eligible"] = work["is_eligible"].apply(normalize_bool)
    if only_eligible:
        work = work[work["is_eligible"]].copy()

    work["URGENCIA"] = work["urgency_level"].apply(normalize_urgency)
    work["SCORE_NUM"] = work["malignancy_score"].apply(normalize_score)
    work["SCORE MALIG."] = work["SCORE_NUM"].apply(lambda x: f"{x}/5")
    work["SAME"] = work["same_id"].fillna("").astype(str)
    work["NOME"] = work["patient_name"].fillna("").astype(str)
    work["IDADE"] = work["age"].fillna("").astype(str)
    work["DATA EXAME"] = work["last_exam_date"].apply(parse_exam_datetime)
    work["MODALIDADE"] = work["exam_modality"].fillna("").astype(str)
    work["CONVENIO"] = work["convenio"].fillna("").astype(str)
    work["SETOR"] = work["setor"].fillna("").astype(str)
    work["ESPECIALIDADE"] = work.apply(
        lambda r: canonical_specialty(
            r.get("medical_specialty", ""),
            f"{r.get('tumor_findings', '')} {r.get('tumor_location', '')} {r.get('tumor_characteristics', '')} {r.get('exam_modality', '')}",
        ),
        axis=1,
    )
    work["MODELO IA"] = work["ai_model"].fillna("").astype(str)
    work = work.sort_values("DATA EXAME", ascending=False, na_position="last")

    cols = [
        "URGENCIA",
        "SCORE MALIG.",
        "SAME",
        "NOME",
        "IDADE",
        "DATA EXAME",
        "MODALIDADE",
        "CONVENIO",
        "SETOR",
        "ESPECIALIDADE",
        "MODELO IA",
        "same_id",
        "ai_analysis",
        "full_text",
        "tumor_findings",
        "tumor_location",
        "tumor_characteristics",
        "urgency_reason",
        "convenio",
        "setor",
        "last_file",
        "context",
        "updated_at",
    ]
    return work[cols]


def style_patient_table(df):
    def row_style(row):
        urgency = normalize_urgency(row.get("URGENCIA", ""))
        bg = ROW_BG_BY_URGENCY.get(urgency, "#1a2230")
        styles = [f"background-color: {bg}; color: #dfe9f3;"] * len(row)
        return styles

    styler = df.style.apply(row_style, axis=1)
    styler = styler.set_properties(subset=["URGENCIA", "SCORE MALIG.", "SAME"], **{"font-weight": "700"})
    styler = styler.format({"DATA EXAME": lambda value: format_exam_date(value)})
    return styler


def render_specialty_tabs(df):
    if df.empty:
        st.info("Nenhum paciente minerado ainda.")
        return

    table_cols = ["URGENCIA", "SCORE MALIG.", "SAME", "NOME", "IDADE", "DATA EXAME", "MODALIDADE", "CONVENIO", "SETOR"]
    specialty_counts = df["ESPECIALIDADE"].value_counts().to_dict()
    ordered_specialties = [name for name in SPECIALTY_BUCKETS if specialty_counts.get(name, 0) > 0]

    labels = ["⚪ Todos ({})".format(len(df))]
    labels.extend([f"{specialty_chip(name)} {name} ({specialty_counts.get(name, 0)})" for name in ordered_specialties])
    tabs = st.tabs(labels)

    with tabs[0]:
        render_clickable_patient_table(df, table_cols, "tab_all")

    for idx, name in enumerate(ordered_specialties, start=1):
        with tabs[idx]:
            filtered = df[df["ESPECIALIDADE"] == name].copy()
            render_clickable_patient_table(filtered, table_cols, f"tab_{idx}_{ascii_fold(name)}")


def render_clickable_patient_table(df, table_cols, table_key):
    if df.empty:
        st.info("Sem pacientes nesta aba.")
        return

    view_df = df[table_cols].copy()
    selected_same = normalize_text(st.session_state.get(f"{table_key}_selected_same"))
    view_df["DATA EXAME"] = view_df["DATA EXAME"].apply(format_exam_date)
    view_df.insert(0, "SELECIONAR", view_df["SAME"].astype(str) == selected_same)

    edited_df = st.data_editor(
        view_df,
        use_container_width=True,
        height=620,
        hide_index=True,
        disabled=table_cols,
        num_rows="fixed",
        column_config={
            "SELECIONAR": st.column_config.CheckboxColumn(
                "SEL",
                help="Marque a linha do paciente para abrir a analise detalhada.",
                default=False,
            )
        },
    )

    selected_rows = edited_df[edited_df["SELECIONAR"]]
    if len(selected_rows) > 1:
        st.info("Mais de uma linha foi marcada. O sistema usara apenas a primeira.")

    selected_same = normalize_text(selected_rows.iloc[0]["SAME"]) if not selected_rows.empty else ""
    st.session_state[f"{table_key}_selected_same"] = selected_same

    action_cols = st.columns([1.2, 3])
    if action_cols[0].button("Abrir analise detalhada", key=f"{table_key}_open_btn", type="primary"):
        if not selected_same:
            st.warning("Selecione uma linha na tabela antes de abrir a analise detalhada.")
        else:
            st.session_state["detail_same_id"] = selected_same
            st.session_state["open_detail_dialog"] = True
            st.rerun()
    action_cols[1].caption("Selecione a linha do paciente diretamente na tabela e abra os detalhes.")


def extract_ai_field(ai_text, keys):
    text = normalize_text(ai_text)
    if not text:
        return ""

    for key in keys:
        pattern = rf"{re.escape(key)}\\s*:\\s*(.+?)(?:\\n[A-Z_\\*\\s]+:|$)"
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return normalize_text(match.group(1))
    return ""


def extract_achados_clinicos(row):
    direct = normalize_text(row.get("tumor_findings"))
    if direct:
        return direct
    return extract_ai_field(
        row.get("ai_analysis"),
        ["ACHADOS", "**ACHADOS**", "ACHADOS CLINICOS PRINCIPAIS"],
    )


def extract_localizacao(row):
    direct = normalize_text(row.get("tumor_location"))
    if direct:
        return direct
    return extract_ai_field(
        row.get("ai_analysis"),
        ["LOCALIZACAO", "LOCALIZACAO E CARACTERISTICAS", "TUMOR_LOCATION"],
    )


def extract_caracteristicas(row):
    direct = normalize_text(row.get("tumor_characteristics"))
    if direct:
        return direct
    return extract_ai_field(
        row.get("ai_analysis"),
        ["CARACTERISTICAS", "TUMOR_CHARACTERISTICS"],
    )


def urgency_badge_color(urgency):
    palette = {
        "CRITICA": "#c1121f",
        "MUITO ALTA": "#e85d04",
        "ALTA": "#f59f00",
        "MODERADA": "#1971c2",
        "BAIXA": "#2b8a3e",
    }
    return palette.get(normalize_urgency(urgency), "#3b3f46")


@st.dialog("Analise Detalhada do Paciente")
def show_patient_detail_dialog(row):
    patient_name = normalize_text(row.get("NOME")) or "Paciente"
    urgency = normalize_urgency(row.get("URGENCIA"))
    score_txt = normalize_text(row.get("SCORE MALIG."))
    score_num = normalize_score(score_txt.split("/")[0] if "/" in score_txt else score_txt)
    stars = "*" * score_num + "." * (5 - score_num)
    urg_color = urgency_badge_color(urgency)

    modalidade = normalize_text(row.get("MODALIDADE"))
    especialidade = normalize_text(row.get("ESPECIALIDADE"))
    achados = extract_achados_clinicos(row) or "Sem descricao de achados."
    localizacao = extract_localizacao(row) or "Nao informado."
    caracteristicas = extract_caracteristicas(row) or "Nao informado."
    urgency_reason = normalize_text(row.get("urgency_reason")) or "Nao informado."
    model_ia = normalize_text(row.get("MODELO IA")) or "Nao informado."

    top_cols = st.columns([4.5, 1.2])
    with top_cols[0]:
        st.markdown(
            f"""
            <div class="detail-header-block">
                <div class="detail-patient-name">{esc(patient_name)}</div>
                <div class="detail-patient-meta">
                    <span>SAME: {esc(row.get('SAME'))}</span>
                    <span>Idade: {esc(row.get('IDADE'))}</span>
                    <span>Data: {esc(format_exam_date(row.get('DATA EXAME')))}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_cols[1]:
        st.markdown(
            f"""
            <div class="detail-side-stack">
                <div class="detail-side-card" style="border-color:{urg_color};">
                    <div class="detail-side-title">URGENCIA</div>
                    <div class="detail-side-value">{esc(urgency)}</div>
                </div>
                <div class="detail-side-card">
                    <div class="detail-side-title">MALIGNIDADE</div>
                    <div class="detail-side-stars">{esc(stars)}</div>
                    <div class="detail-side-value">{esc(score_txt)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    tab_ai, tab_full, tab_details = st.tabs(["Analise IA", "Texto Completo", "Detalhes"])

    with tab_ai:
        st.markdown(
            f"""
            <div class="detail-main-panel">
                <div class="detail-panel-title">ANALISE DA INTELIGENCIA ARTIFICIAL</div>
                <div class="detail-section">
                    <div class="detail-section-title">INFORMACOES DO EXAME</div>
                    <div class="detail-line"><b>Modalidade:</b> {esc(modalidade)}</div>
                    <div class="detail-line"><b>Especialidade:</b> {esc(especialidade)}</div>
                    <div class="detail-line"><b>Modelo IA:</b> {esc(model_ia)}</div>
                </div>
                <div class="detail-section">
                    <div class="detail-section-title">ACHADOS CLINICOS PRINCIPAIS</div>
                    <div class="detail-content">{esc(achados)}</div>
                </div>
                <div class="detail-section">
                    <div class="detail-section-title">LOCALIZACAO E CARACTERISTICAS</div>
                    <div class="detail-content"><b>Localizacao:</b> {esc(localizacao)}</div>
                    <div class="detail-content"><b>Caracteristicas:</b> {esc(caracteristicas)}</div>
                </div>
                <div class="detail-section">
                    <div class="detail-section-title">JUSTIFICATIVA DE URGENCIA</div>
                    <div class="detail-content">{esc(urgency_reason)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.text_area(
            "Bloco original da IA",
            value=normalize_text(row.get("ai_analysis")) or "Sem analise IA salva.",
            height=150,
            disabled=True,
            key=f"ai_detail_raw_{normalize_text(row.get('same_id'))}",
        )

    with tab_full:
        st.text_area(
            "Texto completo do laudo",
            value=normalize_text(row.get("full_text")) or "Sem texto completo salvo.",
            height=360,
            disabled=True,
            key=f"full_detail_{normalize_text(row.get('same_id'))}",
        )

    with tab_details:
        details_df = pd.DataFrame(
            [
                {"Campo": "Especialidade", "Valor": normalize_text(row.get("ESPECIALIDADE"))},
                {"Campo": "Modelo IA", "Valor": normalize_text(row.get("MODELO IA"))},
                {"Campo": "Achados", "Valor": normalize_text(row.get("tumor_findings"))},
                {"Campo": "Localizacao", "Valor": normalize_text(row.get("tumor_location"))},
                {"Campo": "Caracteristicas", "Valor": normalize_text(row.get("tumor_characteristics"))},
                {"Campo": "Motivo urgencia", "Valor": normalize_text(row.get("urgency_reason"))},
                {"Campo": "Arquivo", "Valor": normalize_text(row.get("last_file"))},
                {"Campo": "Atualizado em", "Valor": normalize_text(row.get("updated_at"))},
            ]
        )
        st.dataframe(details_df, use_container_width=True, hide_index=True)


def render_pending_detail_dialog(display_df):
    if not st.session_state.get("open_detail_dialog"):
        return

    same_id = normalize_text(st.session_state.get("detail_same_id"))
    st.session_state["open_detail_dialog"] = False
    if not same_id:
        return

    matched = display_df[display_df["same_id"].astype(str) == same_id]
    if matched.empty:
        return

    row = matched.iloc[0].to_dict()
    show_patient_detail_dialog(row)


def render_css():
    st.markdown(
        """
        <style>
        .onco-toolbar {
            background: linear-gradient(180deg, #2a2f37, #1f252e);
            border: 1px solid #4a4f58;
            border-radius: 10px;
            padding: 10px 12px;
            margin-bottom: 12px;
        }
        .onco-toolbar-title {
            font-size: 14px;
            color: #ced4da;
            font-weight: 600;
            margin-bottom: 4px;
        }
        .onco-status {
            color: #b5c1cf;
            font-size: 14px;
            margin-bottom: 6px;
        }
        [data-testid="stDialog"] > div[role="dialog"] {
            width: min(96vw, 1500px) !important;
            max-width: min(96vw, 1500px) !important;
        }
        [data-baseweb="tab-list"] {
            gap: 6px;
            flex-wrap: wrap;
        }
        [data-baseweb="tab-list"] button {
            border: 1px solid #3d4652 !important;
            border-radius: 8px 8px 0 0 !important;
            background: #1c2430 !important;
            color: #cdd7e1 !important;
            padding: 6px 10px !important;
        }
        [data-baseweb="tab-list"] button[aria-selected="true"] {
            border-color: #228be6 !important;
            background: #0b2e57 !important;
            color: #f1f3f5 !important;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(180deg, #d90429, #9d0208) !important;
            border: 1px solid #ff4d6d !important;
            color: #fff5f5 !important;
            font-weight: 700 !important;
        }
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(180deg, #ef233c, #c1121f) !important;
        }
        .detail-header-block {
            background: linear-gradient(180deg, #27241f, #221f1a);
            border: 1px solid #3a332b;
            border-radius: 10px;
            padding: 12px 14px;
            margin-bottom: 8px;
        }
        .detail-patient-name {
            font-size: clamp(34px, 3.6vw, 52px);
            font-weight: 900;
            color: #f1f3f5;
            line-height: 1.05;
            margin-bottom: 8px;
            word-break: break-word;
        }
        .detail-patient-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            color: #ced4da;
            font-size: 15px;
            font-weight: 600;
        }
        .detail-side-stack {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .detail-side-card {
            background: linear-gradient(180deg, #2b2214, #1f1b16);
            border: 2px solid #5c4c2a;
            border-radius: 10px;
            padding: 8px;
            text-align: center;
        }
        .detail-side-title {
            font-size: 12px;
            font-weight: 800;
            color: #ffa94d;
            letter-spacing: 0.5px;
            margin-bottom: 3px;
        }
        .detail-side-value {
            font-size: clamp(22px, 2.3vw, 34px);
            font-weight: 900;
            color: #ffe066;
            line-height: 1.05;
        }
        .detail-side-stars {
            font-size: clamp(16px, 1.6vw, 26px);
            letter-spacing: 1px;
            color: #ff6b6b;
            margin: 2px 0;
        }
        .detail-main-panel {
            background: linear-gradient(180deg, #071a2f, #051224);
            border: 1px solid #173a63;
            border-radius: 10px;
            padding: 14px;
            margin-bottom: 10px;
            max-height: 50vh;
            overflow-y: auto;
        }
        .detail-panel-title {
            font-size: clamp(28px, 3vw, 42px);
            font-weight: 900;
            color: #f1f3f5;
            margin-bottom: 10px;
        }
        .detail-section {
            background: rgba(2, 28, 60, 0.55);
            border: 1px solid #123c6d;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 9px;
        }
        .detail-section-title {
            font-size: clamp(20px, 2.1vw, 30px);
            font-weight: 900;
            margin-bottom: 6px;
            color: #4dabf7;
            line-height: 1.1;
        }
        .detail-line, .detail-content {
            color: #dee2e6;
            font-size: clamp(17px, 1.5vw, 24px);
            line-height: 1.35;
            margin-bottom: 2px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def test_ai_connection(provider, model, api_key):
    test_text = (
        "PACIENTE: TESTE SISTEMA\n"
        "SAME: 123456\n"
        "EXAME: TC de torax\n"
        "Achado: nodulo pulmonar suspeito para malignidade."
    )
    raw = call_ai(provider, model, api_key, test_text)
    parsed = extract_json_block(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Resposta nao retornou JSON valido: {raw[:500]}")
    return parsed


def main():
    render_css()

    st.markdown(
        '<div style="color:#8ce99a; font-weight:800; font-size:2.1rem; line-height:1.1; letter-spacing:0.3px; margin-bottom:0.45rem;">ACSC-CSSJ/Grupo Oncoclinicas</div>',
        unsafe_allow_html=True,
    )
    st.title("Minerador de Achados Oncológicos - CDI")
    st.caption("Mineracao com IA + lista de pacientes minerados por especialidade no mesmo banco do dashboard.")
    default_db_path = st.session_state.get("selected_db_path", get_db_path())
    auth_user = render_login_screen(default_db_path)
    is_admin = auth_user.get("role") == "admin"

    if is_admin:
        db_path = st.sidebar.text_input("DB_PATH (SQLite)", value=default_db_path)
        st.session_state["selected_db_path"] = db_path
    else:
        db_path = st.session_state.get("selected_db_path", get_db_path())
        st.sidebar.text_input("DB_PATH (SQLite)", value=db_path, disabled=True)

    settings = load_app_settings(db_path)
    admin_runtime = settings.get("admin_runtime", {})
    common_ai_config = get_saved_common_ai_config(settings)
    correlation_info = get_saved_correlation_info(settings)

    st.sidebar.caption(f"Usuario logado: {auth_user.get('username', '')} ({auth_user.get('label', '')})")
    if st.sidebar.button("Sair", use_container_width=True):
        logout_user()

    only_eligible = st.sidebar.checkbox("Mostrar somente elegiveis", value=False)

    if is_admin:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Inicializar/Substituir banco")
        uploaded_db = st.sidebar.file_uploader("Upload banco SQLite", type=["db", "sqlite", "sqlite3"], key="db_upload")
        if uploaded_db is not None and st.sidebar.button("Salvar banco em DB_PATH"):
            try:
                save_uploaded_db(uploaded_db, db_path)
                st.sidebar.success("Banco salvo com sucesso.")
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.sidebar.error(str(exc))

    st.sidebar.markdown("---")
    st.sidebar.subheader("Correlacionar setor/convenio")
    stored_corr_path = get_stored_correlation_file_path(settings, db_path)
    if stored_corr_path is not None:
        st.sidebar.caption(
            "Planilha ativa: "
            f"{normalize_text(correlation_info.get('original_name')) or stored_corr_path.name}"
        )
        st.sidebar.caption(
            "Ultima atualizacao: "
            f"{normalize_text(correlation_info.get('uploaded_at')) or 'Nao informada'}"
        )
    else:
        st.sidebar.caption("Nenhuma planilha de correlacao armazenada.")

    if is_admin:
        uploaded_corr_excel = st.sidebar.file_uploader(
            "Upload Excel (XLS/XLSX)",
            type=["xls", "xlsx"],
            key="corr_excel_upload",
        )
        corr_threshold = st.sidebar.slider(
            "Similaridade minima (%)",
            min_value=60,
            max_value=100,
            value=int(correlation_info.get("threshold", 70)),
            step=1,
            key="corr_threshold_slider",
        )
        if st.sidebar.button("Salvar/Atualizar planilha no sistema", use_container_width=True):
            if uploaded_corr_excel is None:
                st.sidebar.error("Selecione um arquivo Excel antes de salvar.")
            else:
                try:
                    info = store_correlation_excel(db_path, uploaded_corr_excel, auth_user.get("username", ""), corr_threshold)
                    st.sidebar.success(
                        "Planilha armazenada com sucesso. "
                        f"Atualizada em {info.get('uploaded_at', '')}."
                    )
                    st.rerun()
                except Exception as exc:
                    st.sidebar.error(f"Falha ao armazenar planilha: {exc}")

    if st.sidebar.button("Correlacionar Excel com pacientes", use_container_width=True):
        try:
            result = correlate_patients_with_stored_excel(db_path, settings)
            st.session_state["excel_correlation_result"] = result
            st.cache_data.clear()
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"Falha na correlacao: {exc}")

    corr_result = st.session_state.get("excel_correlation_result")
    if isinstance(corr_result, dict) and corr_result.get("success"):
        updated_counts = corr_result.get("updated_counts", {})
        st.sidebar.success(
            (
                "Correlacao concluida: "
                f"{corr_result.get('updated_patients', 0)} pacientes atualizados "
                f"de {corr_result.get('matched_patients', 0)} correlacionados."
            )
        )
        st.sidebar.caption(
            (
                f"Campos atualizados -> convenio: {int(updated_counts.get('convenio', 0))}, "
                f"telefone: {int(updated_counts.get('telefone', 0))}, "
                f"setor: {int(updated_counts.get('setor', 0))}"
            )
        )

    configured_provider = normalize_text(common_ai_config.get("provider")) or "OpenAI"
    if configured_provider not in MODEL_OPTIONS:
        configured_provider = "OpenAI"
    configured_model = normalize_text(common_ai_config.get("model")) or MODEL_OPTIONS[configured_provider][0]
    configured_api_key = normalize_text(common_ai_config.get("api_key"))

    if is_admin:
        provider_state_key = "admin_provider_selected"
        if st.session_state.get(provider_state_key) not in MODEL_OPTIONS:
            runtime_provider = normalize_text(admin_runtime.get("provider")) or configured_provider
            st.session_state[provider_state_key] = runtime_provider if runtime_provider in MODEL_OPTIONS else "OpenAI"

        provider_cols = st.columns([1.0, 1.25, 1.25, 1.35, 1.15])
        provider = provider_cols[0].selectbox("API", list(MODEL_OPTIONS.keys()), key=provider_state_key)

        env_key_name = PROVIDER_ENV_KEYS.get(provider, "")
        runtime_api_key = ""
        if normalize_text(admin_runtime.get("provider")) == provider:
            runtime_api_key = normalize_text(admin_runtime.get("api_key"))
        default_api_key = runtime_api_key or (os.getenv(env_key_name, "") if env_key_name else "")
        api_key_state_key = f"api_key_{provider}"
        if api_key_state_key not in st.session_state:
            st.session_state[api_key_state_key] = default_api_key

        api_key = provider_cols[3].text_input(
            "API Key",
            type="password",
            value=st.session_state.get(api_key_state_key, default_api_key),
            help=f"Variavel de ambiente: {env_key_name}" if env_key_name else "Nao requer chave para uso local.",
        )
        st.session_state[api_key_state_key] = api_key

        update_models_clicked = provider_cols[4].button(
            "Validar chave / Atualizar modelos",
            use_container_width=True,
        )

        if update_models_clicked:
            try:
                fresh_models = fetch_models_for_provider(provider, api_key)
                if not fresh_models:
                    raise RuntimeError("Nenhum modelo retornado para este provider.")
                set_cached_models(provider, fresh_models)
                settings["admin_runtime"]["provider"] = provider
                settings["admin_runtime"]["api_key"] = api_key
                settings["admin_runtime"]["validated_at"] = now_display()
                settings["admin_runtime"].setdefault("available_models", {})
                settings["admin_runtime"]["available_models"][provider] = fresh_models
                save_app_settings(db_path, settings)
                st.session_state[f"model_selected_{provider}"] = fresh_models[0]
                st.success(f"API validada. {len(fresh_models)} modelos carregados para {provider}.")
                st.rerun()
            except Exception as exc:
                st.error(f"Falha ao validar chave/modelos: {exc}")

        provider_models = get_cached_models(provider)
        if not provider_models:
            provider_models = settings.get("admin_runtime", {}).get("available_models", {}).get(provider, [])
        if not provider_models:
            provider_models = list(MODEL_OPTIONS.get(provider, []))
        if provider == configured_provider and configured_model and configured_model not in provider_models:
            provider_models = [configured_model] + provider_models

        model_state_key = f"model_selected_{provider}"
        if st.session_state.get(model_state_key) not in provider_models:
            st.session_state[model_state_key] = configured_model if provider == configured_provider else (provider_models[0] if provider_models else "")

        selected_model = provider_cols[1].selectbox(
            "Modelo",
            provider_models,
            key=model_state_key,
        )
        custom_model_key = f"admin_custom_model_{provider}"
        default_custom_model = configured_model if provider == configured_provider and configured_model not in provider_models else ""
        if custom_model_key not in st.session_state:
            st.session_state[custom_model_key] = default_custom_model
        custom_model = provider_cols[2].text_input("Modelo custom", value=st.session_state.get(custom_model_key, default_custom_model))
        st.session_state[custom_model_key] = custom_model
        effective_model = custom_model.strip() or selected_model

        st.markdown('<div class="onco-toolbar">', unsafe_allow_html=True)
        st.markdown('<div class="onco-toolbar-title">Controles de Mineracao</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="onco-status">Status: Pronto | Provider: {provider} | Modelo: {effective_model}</div>', unsafe_allow_html=True)

        configured_label = (
            f"IA atual do usuario comum: {configured_provider} / {configured_model}"
            if configured_model else
            "IA do usuario comum ainda nao configurada."
        )
        st.info(configured_label)
        if normalize_text(common_ai_config.get("configured_at")):
            st.caption(
                "Configurada por "
                f"{normalize_text(common_ai_config.get('configured_by')) or 'admin'} em "
                f"{normalize_text(common_ai_config.get('configured_at'))}"
            )

        if st.button("Salvar esta IA para o usuario comum", use_container_width=True):
            try:
                if provider != "LLM Studio" and not api_key:
                    raise RuntimeError("Informe uma API key valida antes de salvar a configuracao.")

                runtime_matches = (
                    normalize_text(settings.get("admin_runtime", {}).get("provider")) == provider
                    and normalize_text(settings.get("admin_runtime", {}).get("api_key")) == api_key
                    and normalize_text(settings.get("admin_runtime", {}).get("validated_at"))
                )
                if not runtime_matches:
                    fresh_models = fetch_models_for_provider(provider, api_key)
                    if fresh_models:
                        set_cached_models(provider, fresh_models)
                        settings["admin_runtime"].setdefault("available_models", {})
                        settings["admin_runtime"]["available_models"][provider] = fresh_models

                settings["admin_runtime"]["provider"] = provider
                settings["admin_runtime"]["api_key"] = api_key
                settings["admin_runtime"]["validated_at"] = now_display()
                settings["common_ai_config"] = {
                    "provider": provider,
                    "model": effective_model,
                    "api_key": api_key,
                    "configured_at": now_display(),
                    "configured_by": auth_user.get("username", "admin"),
                }
                save_app_settings(db_path, settings)
                st.success(f"Configuracao salva. Usuarios comuns usarao {provider} / {effective_model}.")
                st.rerun()
            except Exception as exc:
                st.error(f"Falha ao salvar configuracao da IA: {exc}")
    else:
        provider = configured_provider
        effective_model = configured_model
        api_key = configured_api_key

        st.markdown('<div class="onco-toolbar">', unsafe_allow_html=True)
        st.markdown('<div class="onco-toolbar-title">Controles de Mineracao</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="onco-status">Modo usuario comum | IA configurada pelo administrador: {provider} / {effective_model}</div>',
            unsafe_allow_html=True,
        )
        if normalize_text(common_ai_config.get("configured_at")):
            st.caption(
                "Configuracao ativa desde "
                f"{normalize_text(common_ai_config.get('configured_at'))} "
                f"por {normalize_text(common_ai_config.get('configured_by')) or 'admin'}."
            )
        if provider != "LLM Studio" and not api_key:
            st.error("A IA do usuario comum ainda nao foi configurada por um administrador.")

    control_cols = st.columns([1.6, 1.2, 1.2, 1.0])
    uploaded_files = control_cols[0].file_uploader(
        "Selecionar arquivos PDF",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    start_clicked = control_cols[1].button("Iniciar Processamento", use_container_width=True)
    test_clicked = control_cols[2].button("Testar IA", use_container_width=True)
    clear_clicked = control_cols[3].button("Limpar Cache", use_container_width=True)

    if clear_clicked:
        st.cache_data.clear()
        st.rerun()

    if test_clicked:
        try:
            if not effective_model:
                raise RuntimeError("Nenhum modelo configurado para teste.")
            if provider != "LLM Studio" and not api_key:
                raise RuntimeError("API key obrigatoria para o provider selecionado.")
            parsed = test_ai_connection(provider, effective_model, api_key)
            if is_admin:
                try:
                    refreshed = fetch_models_for_provider(provider, api_key)
                    if refreshed:
                        set_cached_models(provider, refreshed)
                except Exception:
                    pass
            st.success("Teste de IA concluido com sucesso.")
            st.json(parsed)
        except Exception as exc:
            st.error(f"Falha no teste de IA: {exc}")

    run_rows = []
    run_errors = []
    if start_clicked:
        if not uploaded_files:
            st.warning("Selecione pelo menos um PDF.")
        elif not effective_model:
            st.error("Nenhum modelo de IA configurado para processamento.")
        elif provider != "LLM Studio" and not api_key:
            st.error("API key obrigatoria para o provider selecionado.")
        else:
            ensure_schema(db_path)
            progress = st.progress(0)
            status_box = st.empty()
            total = len(uploaded_files)

            for idx, uploaded in enumerate(uploaded_files, start=1):
                try:
                    row = process_pdf_with_ai(uploaded, db_path, provider, effective_model, api_key)
                    run_rows.append(row)
                except Exception as exc:
                    run_errors.append({"arquivo": uploaded.name, "erro": str(exc)})

                progress.progress(int(idx * 100 / total))
                status_box.text(f"Processados {idx}/{total}")

            st.cache_data.clear()
            st.success(f"Processamento concluido. Sucesso: {len(run_rows)} | Erros: {len(run_errors)}")

    st.markdown("</div>", unsafe_allow_html=True)

    if run_rows:
        st.subheader("Resumo da execucao")
        run_df = pd.DataFrame(run_rows)
        if "data_exame" in run_df.columns:
            run_df["data_exame"] = run_df["data_exame"].apply(format_exam_date)
        st.dataframe(run_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Exportar resultados da execucao (CSV)",
            run_df.to_csv(index=False).encode("utf-8"),
            file_name="resultado_mineracao_execucao.csv",
            mime="text/csv",
        )

    if run_errors:
        st.error("Arquivos com falha:")
        st.dataframe(pd.DataFrame(run_errors), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Resultados")

    if not os.path.exists(db_path):
        st.warning("Banco nao encontrado no DB_PATH atual.")
        st.stop()

    try:
        mtime = os.path.getmtime(db_path)
        raw_df = load_patients_from_db(db_path, mtime)
    except Exception as exc:
        st.error(f"Erro ao ler banco: {exc}")
        st.stop()

    display_df = build_results_dataframe(raw_df, only_eligible)

    tab_resumo, tab_detalhado, tab_analise = st.tabs([
        "Resumo",
        "Resultados Detalhados",
        "Analise Detalhada",
    ])

    with tab_resumo:
        total = len(display_df)
        metric_cols = st.columns(len(URGENCY_ORDER) + 1)
        metric_cols[0].metric("Total", total)
        for i, urg in enumerate(URGENCY_ORDER, start=1):
            count = int((display_df["URGENCIA"] == urg).sum()) if not display_df.empty else 0
            metric_cols[i].metric(urg, count)

        if not display_df.empty:
            counts_map = display_df["ESPECIALIDADE"].value_counts().to_dict()
            rows = []
            for spec in SPECIALTY_BUCKETS:
                rows.append({"Especialidade": spec, "Pacientes": int(counts_map.get(spec, 0))})
            sp_counts = pd.DataFrame(rows)
            st.dataframe(sp_counts, use_container_width=True, hide_index=True)

    with tab_detalhado:
        render_specialty_tabs(display_df)
        render_pending_detail_dialog(display_df)

    with tab_analise:
        if display_df.empty:
            st.info("Sem dados para analise.")
        else:
            st.dataframe(display_df[["SAME", "NOME", "ESPECIALIDADE", "MODELO IA", "URGENCIA"]], use_container_width=True, hide_index=True)
            st.caption("Coluna MODELO IA comprova qual modelo executou a mineracao.")
            render_pending_detail_dialog(display_df)


if __name__ == "__main__":
    main()
