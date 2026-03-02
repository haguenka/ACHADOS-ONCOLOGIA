import json
import os
import re
import sqlite3
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import requests
import streamlit as st


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


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip()


def ascii_fold(value):
    text = normalize_text(value)
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def get_db_path():
    env_path = os.getenv("DB_PATH", "").strip()
    if env_path:
        return env_path
    return str(Path.cwd() / "tumor_findings_patients.db")


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
    return match.group(1) if match else datetime.utcnow().strftime("%d/%m/%Y")


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
    upper = ascii_fold(text).upper()
    if any(t in upper for t in ["MAMA", "MAMARIO", "BIRADS"]):
        return "Oncologia"
    if any(t in upper for t in ["PULMAO", "TORAX", "NODULO PULMONAR"]):
        return "Pneumologia"
    if any(t in upper for t in ["PROSTATA", "PIRADS", "RIM", "RENAL"]):
        return "Urologia"
    if any(t in upper for t in ["UTERO", "OVARIO", "GINECO", "PELV"]):
        return "Ginecologia"
    if any(t in upper for t in ["FIGADO", "HEPAT", "PANCREAS", "ABDOMEN"]):
        return "Gastroenterologia"
    if any(t in upper for t in ["OSSO", "ORTOP", "MUSCULO"]):
        return "Ortopedia"
    return "Outro"


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
    last_exam_date = normalize_text(ai_dict.get("last_exam_date")) or parse_exam_date_fallback(source_text)
    exam_modality = normalize_text(ai_dict.get("exam_modality")) or infer_modality_fallback(source_text)
    medical_specialty = normalize_text(ai_dict.get("medical_specialty")) or infer_specialty_fallback(source_text)
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
        "data_exame": payload["last_exam_date"],
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
    work["DATA EXAME"] = work["last_exam_date"].fillna("").astype(str)
    work["MODALIDADE"] = work["exam_modality"].fillna("").astype(str)
    work["ESPECIALIDADE"] = work["medical_specialty"].fillna("Outro").astype(str)
    work["MODELO IA"] = work["ai_model"].fillna("").astype(str)

    cols = [
        "URGENCIA",
        "SCORE MALIG.",
        "SAME",
        "NOME",
        "IDADE",
        "DATA EXAME",
        "MODALIDADE",
        "ESPECIALIDADE",
        "MODELO IA",
        "same_id",
        "ai_analysis",
        "full_text",
        "tumor_findings",
        "tumor_location",
        "tumor_characteristics",
        "urgency_reason",
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
    return styler


def render_specialty_tabs(df):
    if df.empty:
        st.info("Nenhum paciente minerado ainda.")
        return

    table_cols = ["URGENCIA", "SCORE MALIG.", "SAME", "NOME", "IDADE", "DATA EXAME", "MODALIDADE"]
    specialty_counts = df["ESPECIALIDADE"].value_counts()
    labels = [f"Todos ({len(df)})"] + [f"{name} ({count})" for name, count in specialty_counts.items()]
    tabs = st.tabs(labels)

    with tabs[0]:
        render_clickable_patient_table(df, table_cols, "tab_all")

    for idx, (name, _count) in enumerate(specialty_counts.items(), start=1):
        with tabs[idx]:
            filtered = df[df["ESPECIALIDADE"] == name].copy()
            render_clickable_patient_table(filtered, table_cols, f"tab_{idx}_{ascii_fold(name)}")


def extract_selected_rows(event):
    if event is None:
        return []
    try:
        rows = event.selection.rows
        if isinstance(rows, list):
            return rows
    except Exception:
        pass

    if isinstance(event, dict):
        selection = event.get("selection", {})
        rows = selection.get("rows", [])
        if isinstance(rows, list):
            return rows

    return []


def register_double_click(table_key, row_idx, threshold=0.8):
    now = time.time()
    signature = f"{table_key}:{row_idx}"
    last_signature = st.session_state.get("last_click_signature")
    last_ts = st.session_state.get("last_click_ts", 0.0)

    st.session_state["last_click_signature"] = signature
    st.session_state["last_click_ts"] = now

    return signature == last_signature and (now - last_ts) <= threshold


def render_clickable_patient_table(df, table_cols, table_key):
    if df.empty:
        st.info("Sem pacientes nesta aba.")
        return

    view_df = df[table_cols].copy()
    event = st.dataframe(
        style_patient_table(view_df),
        use_container_width=True,
        height=620,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"{table_key}_grid",
    )

    selected_rows = extract_selected_rows(event)
    if not selected_rows:
        return

    row_idx = selected_rows[0]
    selected_row = None
    if isinstance(row_idx, int) and 0 <= row_idx < len(df):
        selected_row = df.iloc[row_idx]
    elif row_idx in df.index:
        selected_row = df.loc[row_idx]
    else:
        return

    if register_double_click(table_key, row_idx):
        same_id = normalize_text(selected_row["same_id"])
        if same_id:
            st.session_state["detail_same_id"] = same_id
            st.session_state["open_detail_dialog"] = True


@st.dialog("Analise Detalhada do Paciente")
def show_patient_detail_dialog(row):
    patient_name = normalize_text(row.get("NOME")) or "Paciente"
    urgency = normalize_urgency(row.get("URGENCIA"))
    score_txt = normalize_text(row.get("SCORE MALIG."))

    st.subheader(patient_name)
    meta_cols = st.columns([2, 1, 1, 1])
    meta_cols[0].markdown(
        f"**SAME:** {normalize_text(row.get('SAME'))}  \n"
        f"**Idade:** {normalize_text(row.get('IDADE'))}  \n"
        f"**Data:** {normalize_text(row.get('DATA EXAME'))}"
    )
    meta_cols[1].metric("Urgencia", urgency)
    meta_cols[2].metric("Malignidade", score_txt)
    meta_cols[3].metric("Modalidade", normalize_text(row.get("MODALIDADE")))

    tab_ai, tab_full, tab_details = st.tabs(["Analise IA", "Texto Completo", "Detalhes"])

    with tab_ai:
        st.text_area(
            "Analise gerada pela IA",
            value=normalize_text(row.get("ai_analysis")) or "Sem analise IA salva.",
            height=360,
            disabled=True,
            key=f"ai_detail_{normalize_text(row.get('same_id'))}",
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

    st.title("Minerador de Achados Suspeitos de Tumor")
    st.caption("Mineracao com IA + lista de pacientes minerados por especialidade no mesmo banco do dashboard.")

    db_path = st.sidebar.text_input("DB_PATH (SQLite)", value=get_db_path())
    only_eligible = st.sidebar.checkbox("Mostrar somente elegiveis", value=False)

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

    provider_cols = st.columns([1.0, 1.25, 1.25, 1.35, 1.15])
    provider = provider_cols[0].selectbox("API", list(MODEL_OPTIONS.keys()), index=0)

    env_key_name = PROVIDER_ENV_KEYS.get(provider, "")
    default_api_key = os.getenv(env_key_name, "") if env_key_name else ""
    api_key = provider_cols[3].text_input(
        "API Key",
        type="password",
        value=st.session_state.get(f"api_key_{provider}", default_api_key),
        help=f"Variavel de ambiente: {env_key_name}" if env_key_name else "Nao requer chave para uso local.",
    )
    st.session_state[f"api_key_{provider}"] = api_key

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
            st.session_state[f"model_selected_{provider}"] = fresh_models[0]
            st.success(f"API validada. {len(fresh_models)} modelos carregados para {provider}.")
            st.rerun()
        except Exception as exc:
            st.error(f"Falha ao validar chave/modelos: {exc}")

    provider_models = get_cached_models(provider)
    if not provider_models:
        provider_models = list(MODEL_OPTIONS.get(provider, []))

    model_state_key = f"model_selected_{provider}"
    if st.session_state.get(model_state_key) not in provider_models:
        st.session_state[model_state_key] = provider_models[0] if provider_models else ""

    selected_model = provider_cols[1].selectbox(
        "Modelo",
        provider_models,
        key=model_state_key,
    )
    custom_model = provider_cols[2].text_input("Modelo custom", value="")
    effective_model = custom_model.strip() or selected_model

    st.markdown('<div class="onco-toolbar">', unsafe_allow_html=True)
    st.markdown('<div class="onco-toolbar-title">Controles de Mineracao</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="onco-status">Status: Pronto | Provider: {provider} | Modelo: {effective_model}</div>', unsafe_allow_html=True)

    control_cols = st.columns([1.6, 1.2, 1.2, 1.0])
    uploaded_files = control_cols[0].file_uploader(
        "Selecionar arquivos PDF",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    start_clicked = control_cols[1].button("Iniciar Processamento", type="primary", use_container_width=True)
    test_clicked = control_cols[2].button("Testar IA", use_container_width=True)
    clear_clicked = control_cols[3].button("Limpar Cache", use_container_width=True)

    if clear_clicked:
        st.cache_data.clear()
        st.rerun()

    if test_clicked:
        try:
            if provider != "LLM Studio" and not api_key:
                raise RuntimeError("API key obrigatoria para o provider selecionado.")
            parsed = test_ai_connection(provider, effective_model, api_key)
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
            sp_counts = display_df["ESPECIALIDADE"].value_counts().reset_index()
            sp_counts.columns = ["Especialidade", "Pacientes"]
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


if __name__ == "__main__":
    main()
