
import io
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Barras Agrupadas com Erro", layout="wide")

st.title("Gráfico de Barras Agrupadas com Barras de Erro")
st.caption(
    "Carregue seus dados e gere um gráfico no estilo Plotly. "
    "Funciona com dados em **formato longo** (Categoria, Série, Valor, Erro) "
    "ou **formato largo** (uma coluna por série e, opcionalmente, colunas de erro com sufixo `_err`)."
)

# ------------------------
# Dados de exemplo
# ------------------------
example_long = pd.DataFrame({
    "Categoria": ["10 mg/L","25 mg/L","50 mg/L","75 mg/L","100 mg/L","125 mg/L","150 mg/L","300 mg/L"] * 2,
    "Série": (["MFC/2B@C"] * 8) + (["MFC/8B@C"] * 8),
    "Valor": [62, 44, 32, 30, 28, 26, 22, 18, 68, 56, 43, 41, 33, 31, 29, 20],
    "Erro":  [14,  6,  4,  3,  4,  4,  4,  3,  4,  5,  9,  5,  4,  4,  6,  2],
})

example_wide = pd.DataFrame({
    "Categoria": ["10 mg/L","25 mg/L","50 mg/L","75 mg/L","100 mg/L","125 mg/L","150 mg/L","300 mg/L"],
    "MFC/2B@C":  [62, 44, 32, 30, 28, 26, 22, 18],
    "MFC/2B@C_err": [14, 6, 4, 3, 4, 4, 4, 3],
    "MFC/8B@C":  [68, 56, 43, 41, 33, 31, 29, 20],
    "MFC/8B@C_err": [4, 5, 9, 5, 4, 4, 6, 2],
})

with st.sidebar:
    st.header("Entrada de Dados")
    uploaded = st.file_uploader("CSV ou Excel (.csv, .xlsx)", type=["csv","xlsx"])
    layout_kind = st.radio(
        "Formato dos dados",
        ["Longo (Categoria, Série, Valor, Erro opcional)",
         "Largo (uma coluna por série, erro em <serie>_err)"],
        index=0
    )
    st.markdown("**Baixar exemplos:**")
    st.download_button("📥 Exemplo (LONGO).csv", example_long.to_csv(index=False).encode("utf-8"), file_name="exemplo_longo.csv", mime="text/csv")
    st.download_button("📥 Exemplo (LARGO).csv", example_wide.to_csv(index=False).encode("utf-8"), file_name="exemplo_largo.csv", mime="text/csv")

    st.header("Aparência")
    y_min = st.number_input("Y mínimo (%)", value=0, step=1)
    y_max = st.number_input("Y máximo (%)", value=100, step=1)
    rotate_x = st.slider("Rotação dos rótulos do eixo X (°)", 0, 90, 45)
    show_grid = st.checkbox("Mostrar grade horizontal", value=True)
    legend_title = st.text_input("Título da legenda", value="Oven")

# ---------------------------------
# Carregar dados
# ---------------------------------
def read_data(file):
    if file is None:
        return None
    name = (file.name or "").lower()
    if name.endswith(".xlsx"):
        return pd.read_excel(file)
    return pd.read_csv(file)

df_in = read_data(uploaded)

if df_in is None:
    st.info("Sem arquivo carregado. Usando **dados de exemplo** (formato LONGO).")
    df_in = example_long.copy()

st.subheader("Pré-visualização dos dados")
st.dataframe(df_in, use_container_width=True)

# ---------------------------------
# Mapeamento de colunas
# ---------------------------------
def to_numeric_safe(s):
    return pd.to_numeric(s, errors="coerce")

fig = go.Figure()

if layout_kind.startswith("Longo"):
    # Colunas esperadas (flexíveis via UI)
    cols = df_in.columns.tolist()
    col_cat = st.selectbox("Coluna de CATEGORIA", cols, index=min(0, len(cols)-1))
    col_series = st.selectbox("Coluna de SÉRIE", cols, index=min(1, len(cols)-1))
    col_value = st.selectbox("Coluna de VALOR (%)", cols, index=min(2, len(cols)-1))
    # Erro é opcional
    col_err = st.selectbox("Coluna de ERRO (opcional, pontos %)", ["<nenhuma>"] + cols, index=0)

    # Coagir tipos
    df_plot = df_in.copy()
    df_plot[col_value] = to_numeric_safe(df_plot[col_value])
    if col_err != "<nenhuma>":
        df_plot[col_err] = to_numeric_safe(df_plot[col_err])

    categories = df_plot[col_cat].astype(str).tolist()
    series_names = list(pd.unique(df_plot[col_series].astype(str)))

    for serie in series_names:
        sub = df_plot[df_plot[col_series].astype(str) == serie]
        x = sub[col_cat].astype(str).tolist()
        y = to_numeric_safe(sub[col_value]).tolist()
        if col_err != "<nenhuma>":
            err = to_numeric_safe(sub[col_err]).tolist()
            error_y = dict(type="data", array=err, visible=True)
        else:
            error_y = None
        fig.add_bar(name=str(serie), x=x, y=y, error_y=error_y)

else:
    # Largo: escolher colunas
    cols = df_in.columns.tolist()
    if len(cols) < 2:
        st.error("Para formato largo, é necessário ao menos uma coluna de categorias e uma de série.")
        st.stop()

    col_cat = st.selectbox("Coluna de CATEGORIA", cols, index=0)
    # Sugerir séries (excluir coluna de categoria e colunas *_err)
    candidate_series = [c for c in cols if c != col_cat and not c.endswith("_err")]
    series_chosen = st.multiselect("Colunas de SÉRIE", candidate_series, default=candidate_series)

    # Mapear colunas de erro por série (opcional)
    err_map = {}
    for s_name in series_chosen:
        default_err = f"{s_name}_err" if f"{s_name}_err" in cols else None
        options = ["<nenhuma>"] + cols
        idx = options.index(default_err) if default_err in options else 0
        err_col = st.selectbox(f"Erro para '{s_name}' (opcional)", options, index=idx, key=f"err_{s_name}")
        err_map[s_name] = (None if err_col == "<nenhuma>" else err_col)

    # Construir barras
    df_plot = df_in.copy()
    cats = df_plot[col_cat].astype(str).tolist()
    for s_name in series_chosen:
        y = to_numeric_safe(df_plot[s_name]).tolist()
        err_col = err_map.get(s_name)
        if err_col:
            err = to_numeric_safe(df_plot[err_col]).tolist()
            error_y = dict(type="data", array=err, visible=True)
        else:
            error_y = None
        fig.add_bar(name=str(s_name), x=cats, y=y, error_y=error_y)

# ---------------------------------
# Layout e renderização
# ---------------------------------
fig.update_layout(
    barmode="group",
    bargap=0.2,
    bargroupgap=0.08,
    xaxis_title="Concentração (mg/L)",
    yaxis_title="Removal (%)",
    legend_title=legend_title,
    template="plotly_white",
    margin=dict(l=60, r=30, t=60, b=80),
)
fig.update_yaxes(range=[y_min, y_max], ticksuffix="%", showgrid=show_grid, gridcolor="rgba(0,0,0,0.1)")
fig.update_xaxes(tickangle=rotate_x)

st.subheader("Gráfico")
config = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "filename": "grafico_barras"}}
st.plotly_chart(fig, use_container_width=True, config=config)

st.info("💡 Dica: Use o botão da barra de ferramentas do gráfico (câmera) para baixar o PNG.")

# ---------------------------------
# Downloads (PNG/HTML) com fallback
# ---------------------------------
col1, col2 = st.columns(2)
with col1:
    try:
        # Tentativa de gerar PNG no servidor (requer 'kaleido' + Chrome/Chromium dependendo da versão)
        png_bytes = fig.to_image(format="png", scale=2)
        st.download_button("⬇️ Baixar PNG (servidor)", data=png_bytes, file_name="grafico_barras.png", mime="image/png")
    except Exception as e:
        st.warning("Exportação PNG no servidor indisponível neste ambiente. Use o botão da barra do gráfico (câmera) ou baixe o HTML.")
with col2:
    html_bytes = fig.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
    st.download_button("⬇️ Baixar HTML interativo", data=html_bytes, file_name="grafico_barras.html", mime="text/html")

st.caption("Compatível com dados em formato longo ou largo. Para erros em formato largo, nomeie a coluna como '<serie>_err'.")
