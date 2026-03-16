import json
import os
import re
import sqlite3
import unicodedata
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


st.set_page_config(
    page_title="Achados Onco Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


URGENCY_ORDER = ["CRITICA", "MUITO ALTA", "ALTA", "MODERADA", "BAIXA"]
URGENCY_COLORS = {
    "CRITICA": "#dc2626",
    "MUITO ALTA": "#ea580c",
    "ALTA": "#f59e0b",
    "MODERADA": "#10b981",
    "BAIXA": "#6b7280",
}
APP_SETTINGS_FILENAME = "onco_app_settings.json"
PROVIDER_BASE_URLS = {
    "OpenAI": "https://api.openai.com/v1",
    "DeepSeek": "https://api.deepseek.com",
    "LLM Studio": os.getenv("LLM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
}


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip()


def ascii_fold(value):
    text = normalize_text(value)
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def is_true_value(value):
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return int(value) == 1
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y", "sim"}


def extract_from_analysis(analysis, field):
    if not analysis:
        return ""
    analysis_ascii = ascii_fold(analysis)

    if field == "specialty":
        patterns = [
            r"\*\*ESPECIALIDADE\s+MEDICA\*\*:?\s*([^\n*]+)",
            r"ESPECIALIDADE\s+MEDICA:\s*([^\n]+)",
        ]
    elif field == "modality":
        patterns = [
            r"\*\*MODALIDADE\s+DO\s+EXAME\*\*:?\s*([^\n*]+)",
            r"MODALIDADE\s+DO\s+EXAME:\s*([^\n]+)",
        ]
    else:
        return ""

    for pattern in patterns:
        match = re.search(pattern, analysis_ascii, re.IGNORECASE)
        if match:
            return normalize_text(match.group(1))
    return ""


def extract_urgency(analysis):
    if not analysis:
        return ""
    analysis_ascii = ascii_fold(analysis)
    match = re.search(
        r"URGENCIA:\s*(CRITICA|MUITO ALTA|ALTA|MODERADA|BAIXA)",
        analysis_ascii,
        re.IGNORECASE,
    )
    if not match:
        return ""
    return normalize_urgency(match.group(1))


def extract_malignancy_score(analysis):
    if not analysis:
        return None
    patterns = [
        r"ESCORE DE MALIGNIDADE:\s*([0-5])",
        r"MALIGNANCY SCORE:\s*([0-5])",
    ]
    for pattern in patterns:
        match = re.search(pattern, analysis, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def normalize_urgency(value):
    text = ascii_fold(value).upper()
    if text == "CRITICA":
        return "CRITICA"
    if text in URGENCY_ORDER:
        return text
    return text


def default_db_candidates():
    cwd = Path.cwd()
    local_db = cwd / "tumor_findings_patients.db"
    parent_db = cwd.parent / "tumor_findings_patients.db"
    env_db = os.getenv("DB_PATH", "")
    candidates = []
    if env_db:
        candidates.append(Path(env_db))
    candidates.extend([local_db, parent_db])
    return candidates


def get_app_settings_path(db_path):
    return Path(db_path).expanduser().resolve().parent / APP_SETTINGS_FILENAME


def load_app_settings(db_path):
    settings_path = get_app_settings_path(db_path)
    if not settings_path.exists():
        return {}
    try:
        return json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_dashboard_ai_config(db_path):
    settings = load_app_settings(db_path)
    config = settings.get("common_ai_config", {}) or {}
    provider = normalize_text(config.get("provider"))
    model = normalize_text(config.get("model"))
    api_key = normalize_text(config.get("api_key"))
    configured_at = normalize_text(config.get("configured_at"))
    configured_by = normalize_text(config.get("configured_by"))
    return {
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "configured_at": configured_at,
        "configured_by": configured_by,
    }


def call_openai_compatible(base_url, api_key, model, prompt):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Voce gera resumos gerenciais objetivos em portugues do Brasil."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=90,
    )
    response.raise_for_status()
    data = response.json()
    return normalize_text(data["choices"][0]["message"]["content"])


def call_gemini(api_key, model, prompt):
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2},
    }
    response = requests.post(endpoint, json=payload, timeout=90)
    response.raise_for_status()
    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "\n".join(normalize_text(part.get("text")) for part in parts if isinstance(part, dict)).strip()


def build_dashboard_summary_payload(df, data_df, total_patients, total_eligible, eligible_rate, avg_score, high_risk):
    urgency_counts = count_series(data_df, "urgency_final")
    top_specialties = count_series(data_df, "specialty_final").head(5).to_dict()
    top_modalities = count_series(data_df, "modality_final").head(5).to_dict()
    top_convenios = count_series(data_df, "convenio").head(5).to_dict()
    top_setores = count_series(data_df, "setor").head(5).to_dict()

    return {
        "total_pacientes_banco": int(total_patients),
        "total_elegiveis_banco": int(total_eligible),
        "taxa_elegibilidade_banco": round(float(eligible_rate), 1),
        "registros_visualizacao_atual": int(len(data_df)),
        "score_medio_visualizacao": round(float(avg_score), 1),
        "alto_risco_score_maior_igual_4": int(high_risk),
        "urgencia": {key: int(value) for key, value in urgency_counts.to_dict().items()},
        "top_especialidades": {key: int(value) for key, value in top_specialties.items()},
        "top_modalidades": {key: int(value) for key, value in top_modalities.items()},
        "top_convenios": {key: int(value) for key, value in top_convenios.items()},
        "top_setores": {key: int(value) for key, value in top_setores.items()},
    }


def fallback_operational_summary(summary_payload):
    urg = summary_payload.get("urgencia", {})
    top_specs = summary_payload.get("top_especialidades", {})
    top_mods = summary_payload.get("top_modalidades", {})
    top_convs = summary_payload.get("top_convenios", {})
    top_sets = summary_payload.get("top_setores", {})

    def first_label(data):
        return next(iter(data.items())) if data else ("Nao informado", 0)

    spec_name, spec_total = first_label(top_specs)
    mod_name, mod_total = first_label(top_mods)
    conv_name, conv_total = first_label(top_convs)
    setor_name, setor_total = first_label(top_sets)

    return (
        f"- Base analisada: {summary_payload['registros_visualizacao_atual']} registros na visualizacao atual e "
        f"{summary_payload['total_elegiveis_banco']} elegiveis no banco.\n"
        f"- Taxa de elegibilidade global: {summary_payload['taxa_elegibilidade_banco']}%.\n"
        f"- Pacientes de alto risco (score >= 4): {summary_payload['alto_risco_score_maior_igual_4']}.\n"
        f"- Especialidade com maior volume: {spec_name} ({spec_total}).\n"
        f"- Modalidade predominante: {mod_name} ({mod_total}).\n"
        f"- Maior concentracao operacional atual: convenio {conv_name} ({conv_total}) e setor {setor_name} ({setor_total}).\n"
        f"- Distribuicao de urgencia: {', '.join(f'{k} {v}' for k, v in urg.items()) if urg else 'sem dados relevantes'}."
    )


def generate_operational_summary_with_ai(db_path, summary_payload):
    ai_config = get_dashboard_ai_config(db_path)
    provider = ai_config.get("provider")
    model = ai_config.get("model")
    api_key = ai_config.get("api_key")

    if not provider or not model:
        return fallback_operational_summary(summary_payload), "Resumo local", ai_config
    if provider != "LLM Studio" and not api_key:
        return fallback_operational_summary(summary_payload), "Resumo local", ai_config

    prompt = (
        "Gere um resumo operacional/gerencial executivo em portugues do Brasil para um dashboard de achados oncologicos.\n"
        "Regras:\n"
        "- Use somente os dados agregados fornecidos.\n"
        "- Nao cite nomes de pacientes, exemplos individuais nem dados sensiveis.\n"
        "- Estruture em 5 a 7 bullets curtos.\n"
        "- Traga leitura operacional, gargalos provaveis, prioridades e uma recomendacao de gestao.\n\n"
        f"DADOS AGREGADOS:\n{json.dumps(summary_payload, ensure_ascii=False, indent=2)}"
    )

    try:
        if provider == "Gemini":
            summary = call_gemini(api_key, model, prompt)
        else:
            summary = call_openai_compatible(PROVIDER_BASE_URLS[provider], api_key, model, prompt)
        summary = normalize_text(summary)
        if not summary:
            raise RuntimeError("Resumo vazio retornado pela IA.")
        return summary, f"{provider} / {model}", ai_config
    except Exception:
        return fallback_operational_summary(summary_payload), "Resumo local", ai_config


def get_cached_dashboard_summary(db_path, summary_payload, only_eligible):
    ai_config = get_dashboard_ai_config(db_path)
    cache_key_payload = {
        "db_path": str(db_path),
        "only_eligible": bool(only_eligible),
        "summary_payload": summary_payload,
        "provider": ai_config.get("provider"),
        "model": ai_config.get("model"),
        "configured_at": ai_config.get("configured_at"),
    }
    fingerprint = json.dumps(cache_key_payload, ensure_ascii=False, sort_keys=True)
    cache_bucket = st.session_state.setdefault("dashboard_ai_summary_cache", {})
    cached = cache_bucket.get(fingerprint)
    if cached:
        return cached

    generated = generate_operational_summary_with_ai(db_path, summary_payload)
    cache_bucket[fingerprint] = generated
    return generated


def save_uploaded_db(uploaded_file, db_path):
    target = Path(db_path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return f"Nao foi possivel criar diretorio de destino: {exc}"

    tmp_path = Path(f"{db_path}.tmp")
    try:
        with open(tmp_path, "wb") as fp:
            fp.write(uploaded_file.getvalue())
    except Exception as exc:
        return f"Falha ao gravar arquivo temporario: {exc}"

    conn = None
    try:
        conn = sqlite3.connect(str(tmp_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='patients'")
        if cursor.fetchone() is None:
            return "Arquivo invalido: tabela 'patients' nao encontrada."
    except Exception as exc:
        return f"SQLite invalido: {exc}"
    finally:
        if conn is not None:
            conn.close()

    try:
        os.replace(str(tmp_path), str(target))
    except Exception as exc:
        return f"Falha ao mover banco para destino final: {exc}"

    return None


@st.cache_data(show_spinner=False)
def load_patients_from_db(db_path, mtime_token):
    _ = mtime_token
    conn = sqlite3.connect(db_path)
    try:
        query = """
            SELECT
                same_id,
                patient_name,
                ai_analysis,
                is_eligible,
                convenio,
                setor,
                exam_modality,
                medical_specialty,
                malignancy_score,
                urgency_level,
                created_at,
                updated_at
            FROM patients
        """
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    return df


def build_dataframe(df):
    work = df.copy()
    work["ai_analysis"] = work["ai_analysis"].fillna("")
    work["eligible_bool"] = work["is_eligible"].apply(is_true_value)

    work["specialty_final"] = work.apply(
        lambda r: normalize_text(r.get("medical_specialty")) or extract_from_analysis(r["ai_analysis"], "specialty"),
        axis=1,
    )
    work["modality_final"] = work.apply(
        lambda r: normalize_text(r.get("exam_modality")) or extract_from_analysis(r["ai_analysis"], "modality"),
        axis=1,
    )
    work["urgency_final"] = work.apply(
        lambda r: normalize_urgency(normalize_text(r.get("urgency_level")) or extract_urgency(r["ai_analysis"])),
        axis=1,
    )

    def resolve_score(row):
        value = row.get("malignancy_score")
        if value is not None and str(value).strip() != "":
            try:
                return int(value)
            except Exception:
                pass
        return extract_malignancy_score(row.get("ai_analysis"))

    work["malignancy_score_final"] = work.apply(resolve_score, axis=1)
    work["convenio"] = work["convenio"].fillna("").astype(str).str.strip()
    work["setor"] = work["setor"].fillna("").astype(str).str.strip()
    work = work[~work["setor"].str.lower().str.contains("auditoria", na=False)].copy()
    return work


def count_series(df, column):
    if column not in df.columns:
        return pd.Series(dtype="int64")
    data = df[column].fillna("").astype(str).str.strip()
    data = data[(data != "") & (~data.str.upper().isin(["NULL", "NONE", "N/A", "NAN"]))]
    return data.value_counts()


def plot_bar_counts(counts, title, color):
    if counts.empty:
        st.info(f"Sem dados para {title.lower()}.")
        return
    chart_df = counts.reset_index()
    chart_df.columns = ["categoria", "total"]
    fig = px.bar(
        chart_df,
        x="total",
        y="categoria",
        orientation="h",
        title=title,
        color_discrete_sequence=[color],
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), yaxis_title=None, xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)


def plot_urgency(urgency_counts):
    ordered = [(u, int(urgency_counts.get(u, 0))) for u in URGENCY_ORDER if int(urgency_counts.get(u, 0)) > 0]
    if not ordered:
        st.info("Sem dados para niveis de urgencia.")
        return
    chart_df = pd.DataFrame(ordered, columns=["urgency", "total"])
    fig = px.bar(
        chart_df,
        x="urgency",
        y="total",
        title="Niveis de Urgencia",
        color="urgency",
        color_discrete_map=URGENCY_COLORS,
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), xaxis_title=None, yaxis_title=None, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def plot_plan_vs_urgency(df):
    filtered = df[(df["convenio"] != "") & (df["urgency_final"].isin(URGENCY_ORDER))]
    if filtered.empty:
        st.info("Sem dados para correlacao criticidade x planos.")
        return

    grouped = (
        filtered.groupby(["convenio", "urgency_final"], dropna=False)
        .size()
        .reset_index(name="total")
    )
    total_per_plan = grouped.groupby("convenio")["total"].sum().sort_values(ascending=False)
    top_plans = total_per_plan.head(8).index
    grouped = grouped[grouped["convenio"].isin(top_plans)]

    fig = px.bar(
        grouped,
        x="convenio",
        y="total",
        color="urgency_final",
        title="Criticidade x Planos de Saude (Top 8)",
        category_orders={"urgency_final": URGENCY_ORDER},
        color_discrete_map=URGENCY_COLORS,
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=30), xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("Dashboard Linha de Cuidados - Achados Oncológicos CDI")
    st.caption("Fonte: tabela patients do app pdf_tumor_findings_miner.py")

    candidates = default_db_candidates()
    db_path = str(candidates[0]) if candidates else "tumor_findings_patients.db"

    st.sidebar.header("Configuracao")
    only_eligible = st.sidebar.checkbox("Somente elegiveis", value=False)

    if st.sidebar.button("Recarregar dados"):
        st.cache_data.clear()

    if not os.path.exists(db_path):
        st.error(
            "Banco nao encontrado no caminho configurado pela aplicacao. "
            "Inicialize ou substitua o banco na pagina Mineracao Onco."
        )
        st.stop()

    try:
        mtime = os.path.getmtime(db_path)
        raw_df = load_patients_from_db(db_path, mtime)
    except Exception as exc:
        st.error(f"Erro ao ler banco: {exc}")
        st.stop()

    df = build_dataframe(raw_df)

    data_df = df[df["eligible_bool"]] if only_eligible else df
    total_patients = int(len(df))
    total_eligible = int(df["eligible_bool"].sum())
    eligible_rate = (total_eligible / total_patients * 100.0) if total_patients else 0.0

    scores = data_df["malignancy_score_final"].dropna().astype(int)
    avg_score = float(scores.mean()) if len(scores) else 0.0
    high_risk = int((scores >= 4).sum()) if len(scores) else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total pacientes", total_patients)
    c2.metric("Elegiveis", total_eligible)
    c3.metric("Taxa elegibilidade", f"{eligible_rate:.1f}%")
    c4.metric("Score medio", f"{avg_score:.1f}")

    a, b, c = st.columns(3)
    with a:
        plot_bar_counts(count_series(data_df, "specialty_final").head(8), "Especialidades Medicas", "#38bdf8")
    with b:
        plot_urgency(count_series(data_df, "urgency_final"))
    with c:
        plot_bar_counts(count_series(data_df, "modality_final").head(8), "Modalidades de Exame", "#06b6d4")

    d, e = st.columns(2)
    with d:
        plot_bar_counts(count_series(data_df, "convenio").head(8), "Convenios Medicos", "#10b981")
    with e:
        plot_bar_counts(count_series(data_df, "setor").head(8), "Setores Hospitalares", "#f59e0b")

    plot_plan_vs_urgency(data_df)

    st.subheader("Resumo Operacional/Gerencial")
    summary_payload = build_dashboard_summary_payload(
        df=df,
        data_df=data_df,
        total_patients=total_patients,
        total_eligible=total_eligible,
        eligible_rate=eligible_rate,
        avg_score=avg_score,
        high_risk=high_risk,
    )
    summary_text, summary_source, ai_config = get_cached_dashboard_summary(db_path, summary_payload, only_eligible)

    st.caption(
        "Gerado por: "
        f"{summary_source}"
        + (
            f" | Configurado em {ai_config.get('configured_at')}"
            if normalize_text(ai_config.get("configured_at"))
            else ""
        )
    )
    st.markdown(summary_text.replace("\n", "  \n"))


if __name__ == "__main__":
    main()
