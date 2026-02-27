import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st
from pypdf import PdfReader


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


def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    chunks = []
    for page in reader.pages:
        text = page.extract_text() or ""
        chunks.append(text)
    return "\n".join(chunks).strip()


def parse_same_id(text):
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


def parse_patient_name(text):
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


def parse_exam_date(text):
    match = re.search(r"\b(\d{2}[\/\-]\d{2}[\/\-]\d{4})\b", text)
    return match.group(1) if match else datetime.utcnow().strftime("%d/%m/%Y")


def infer_modality(text_upper):
    if "RESSON" in text_upper or re.search(r"\bRM\b", text_upper):
        return "RESSONANCIA MAGNETICA"
    if "TOMOGRAF" in text_upper or re.search(r"\bTC\b", text_upper):
        return "TOMOGRAFIA COMPUTADORIZADA"
    if "PET" in text_upper:
        return "PET-CT"
    if "MAMOGRAF" in text_upper:
        return "MAMOGRAFIA"
    return "RADIOLOGIA"


def infer_specialty(text_upper):
    if any(t in text_upper for t in ["MAMA", "MAMARIO", "BIRADS"]):
        return "ONCOLOGIA MAMARIA"
    if any(t in text_upper for t in ["PULMAO", "TORAX", "NODULO PULMONAR"]):
        return "ONCOLOGIA TORACICA"
    if any(t in text_upper for t in ["FIGADO", "HEPATIC", "LIRADS"]):
        return "ONCOLOGIA ABDOMINAL"
    if any(t in text_upper for t in ["PROSTATA", "PIRADS"]):
        return "ONCOLOGIA UROLOGICA"
    if any(t in text_upper for t in ["CEREBRO", "ENCEFALO", "CRANIO"]):
        return "NEURO-ONCOLOGIA"
    return "ONCOLOGIA RADIOLOGICA"


def infer_location(text_upper):
    locations = [
        ("PULMAO", "pulmao"),
        ("MAMA", "mama"),
        ("FIGADO", "figado"),
        ("PROSTATA", "prostata"),
        ("RIM", "rim"),
        ("PANCREAS", "pancreas"),
        ("CEREBRO", "cerebro"),
        ("OSSO", "osso"),
    ]
    found = [label for token, label in locations if token in text_upper]
    return ", ".join(found) if found else "nao especificado"


def evaluate_oncology_risk(text):
    text_upper = text.upper()
    weighted_terms = {
        "METASTASE": 4,
        "METASTATICO": 4,
        "CARCINOMA": 4,
        "ADENOCARCINOMA": 4,
        "LINFOMA": 4,
        "NEOPLASIA": 3,
        "MALIGN": 3,
        "NODULO": 2,
        "MASSA": 2,
        "LESAO": 2,
        "BIRADS 4": 2,
        "BIRADS 5": 3,
        "PIRADS 4": 2,
        "PIRADS 5": 3,
        "LIRADS 4": 2,
        "LIRADS 5": 3,
        "BIOPSIA": 1,
    }

    found_terms = []
    total_weight = 0
    for term, weight in weighted_terms.items():
        if term in text_upper:
            found_terms.append(term)
            total_weight += weight

    if total_weight >= 9:
        score = 5
        urgency = "CRITICA"
    elif total_weight >= 6:
        score = 4
        urgency = "MUITO ALTA"
    elif total_weight >= 4:
        score = 3
        urgency = "ALTA"
    elif total_weight >= 2:
        score = 2
        urgency = "MODERADA"
    elif total_weight >= 1:
        score = 1
        urgency = "BAIXA"
    else:
        score = 0
        urgency = "BAIXA"

    eligible = score >= 2
    reason = ", ".join(found_terms[:8]) if found_terms else "sem termos oncologicos relevantes"
    return score, urgency, eligible, reason, found_terms


def build_analysis_text(modality, specialty, score, urgency, reason):
    conclusion = "ELEGIVEL" if score >= 2 else "NAO ELEGIVEL"
    return (
        f"**MODALIDADE DO EXAME**: {modality}\n"
        f"**ESPECIALIDADE MEDICA**: {specialty}\n"
        f"**ACHADOS**: Mineracao automatica por regras.\n"
        f"**ESCORE DE MALIGNIDADE**: {score}\n"
        f"URGENCIA: {urgency}\n"
        f"MOTIVO DA URGENCIA: {reason}\n"
        f"CONCLUSAO: {conclusion}"
    )


def upsert_patient(db_path, data):
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO patients (
                same_id, patient_name, last_exam_date, last_file, context,
                full_text, ai_analysis, ai_model, is_eligible, exam_title,
                exam_modality, medical_specialty, tumor_findings, tumor_location,
                tumor_characteristics, malignancy_score, urgency_level, urgency_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(same_id) DO UPDATE SET
                patient_name = excluded.patient_name,
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


def process_pdf(uploaded_file, db_path):
    text = extract_pdf_text(uploaded_file)
    if not text:
        raise ValueError("PDF sem texto extraivel.")

    text_upper = text.upper()
    same_id = parse_same_id(text)
    patient_name = parse_patient_name(text)
    exam_date = parse_exam_date(text)
    modality = infer_modality(text_upper)
    specialty = infer_specialty(text_upper)
    location = infer_location(text_upper)
    score, urgency, eligible, reason, terms = evaluate_oncology_risk(text)

    ai_analysis = build_analysis_text(modality, specialty, score, urgency, reason)
    tumor_findings = ", ".join(terms[:10]) if terms else "sem achados relevantes"

    payload = {
        "same_id": same_id,
        "patient_name": patient_name,
        "last_exam_date": exam_date,
        "last_file": uploaded_file.name,
        "context": '{"source":"streamlit_mineracao_onco"}',
        "full_text": text[:250000],
        "ai_analysis": ai_analysis,
        "ai_model": "RULES_MINER_RENDER_V1",
        "is_eligible": 1 if eligible else 0,
        "exam_title": uploaded_file.name,
        "exam_modality": modality,
        "medical_specialty": specialty,
        "tumor_findings": tumor_findings,
        "tumor_location": location,
        "tumor_characteristics": reason,
        "malignancy_score": score,
        "urgency_level": urgency,
        "urgency_reason": reason,
    }
    upsert_patient(db_path, payload)

    return {
        "arquivo": uploaded_file.name,
        "same_id": same_id,
        "paciente": patient_name,
        "modalidade": modality,
        "especialidade": specialty,
        "score": score,
        "urgencia": urgency,
        "elegivel": "sim" if eligible else "nao",
        "achados": tumor_findings,
    }


def main():
    st.title("Mineracao Oncologica")
    st.caption("Upload de PDFs e gravacao na mesma base usada pelo dashboard.")

    db_path = st.sidebar.text_input("DB_PATH (SQLite)", value=get_db_path())
    uploaded_files = st.file_uploader(
        "Selecione PDFs para minerar",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Minerar e salvar no banco", type="primary"):
        if not uploaded_files:
            st.warning("Selecione pelo menos um PDF.")
            st.stop()

        ensure_schema(db_path)
        progress = st.progress(0)
        status = st.empty()
        rows = []
        errors = []

        total = len(uploaded_files)
        for idx, uploaded in enumerate(uploaded_files, start=1):
            try:
                rows.append(process_pdf(uploaded, db_path))
            except Exception as exc:
                errors.append({"arquivo": uploaded.name, "erro": str(exc)})
            progress.progress(int(idx * 100 / total))
            status.text(f"Processados {idx}/{total}")

        st.success(f"Mineracao concluida. Sucesso: {len(rows)} | Erros: {len(errors)}")

        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "Baixar resumo CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="resumo_mineracao_onco.csv",
                mime="text/csv",
            )

        if errors:
            st.error("Alguns arquivos falharam:")
            st.dataframe(pd.DataFrame(errors), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.info("Apos minerar, abra a pagina principal do Dashboard para ver os indicadores atualizados.")


if __name__ == "__main__":
    main()
