import os
import re
import sqlite3
import unicodedata
from pathlib import Path

import pandas as pd
import plotly.express as px
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
    data = data[(data != "") & (~data.str.upper().isin(["NULL", "NONE", "N/A"]))]
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
    st.title("Dashboard Oncologico - Tumor Findings Miner")
    st.caption("Fonte: tabela patients do app pdf_tumor_findings_miner.py")

    candidates = default_db_candidates()
    suggested_path = str(candidates[0]) if candidates else "tumor_findings_patients.db"

    st.sidebar.header("Configuracao")
    db_path = st.sidebar.text_input("DB_PATH (SQLite)", value=suggested_path)
    only_eligible = st.sidebar.checkbox("Somente elegiveis", value=False)

    if st.sidebar.button("Recarregar dados"):
        st.cache_data.clear()

    if not db_path:
        st.error("Informe o caminho do banco SQLite.")
        return

    if not os.path.exists(db_path):
        st.error(
            "Banco nao encontrado. Ajuste DB_PATH no painel da esquerda "
            "ou configure a variavel DB_PATH no Render."
        )
        st.sidebar.markdown("---")
        st.sidebar.subheader("Inicializar banco")
        uploaded_db = st.sidebar.file_uploader(
            "Upload do arquivo SQLite",
            type=["db", "sqlite", "sqlite3"],
            accept_multiple_files=False,
        )
        if uploaded_db is not None and st.sidebar.button("Salvar banco em DB_PATH"):
            error = save_uploaded_db(uploaded_db, db_path)
            if error:
                st.sidebar.error(error)
            else:
                st.sidebar.success("Banco salvo com sucesso. Recarregando...")
                st.cache_data.clear()
                st.rerun()
        st.stop()

    try:
        mtime = os.path.getmtime(db_path)
        raw_df = load_patients_from_db(db_path, mtime)
    except Exception as exc:
        st.error(f"Erro ao ler banco: {exc}")
        st.sidebar.markdown("---")
        st.sidebar.subheader("Substituir banco")
        uploaded_db = st.sidebar.file_uploader(
            "Upload do arquivo SQLite",
            type=["db", "sqlite", "sqlite3"],
            accept_multiple_files=False,
            key="replace_db",
        )
        if uploaded_db is not None and st.sidebar.button("Substituir banco em DB_PATH"):
            error = save_uploaded_db(uploaded_db, db_path)
            if error:
                st.sidebar.error(error)
            else:
                st.sidebar.success("Banco substituido. Recarregando...")
                st.cache_data.clear()
                st.rerun()
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

    st.subheader("Resumo")
    st.write(
        f"Pacientes alto risco (score >= 4): **{high_risk}** "
        f"| Registros na visualizacao atual: **{len(data_df)}**"
    )

    with st.expander("Visualizar dados (amostra)"):
        cols = [
            "same_id",
            "patient_name",
            "eligible_bool",
            "specialty_final",
            "modality_final",
            "urgency_final",
            "malignancy_score_final",
            "convenio",
            "setor",
            "updated_at",
        ]
        cols = [c for c in cols if c in data_df.columns]
        st.dataframe(data_df[cols].head(500), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
