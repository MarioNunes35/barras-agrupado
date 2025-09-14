import io
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =============================================
# Helper: Color Palettes (incl. colorblind-safe)
# =============================================
OKABE_ITO = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
]

PALETTES = {
    "Plotly (qualitative)": px.colors.qualitative.Plotly,
    "D3": px.colors.qualitative.D3,
    "Bold": px.colors.qualitative.Bold,
    "Pastel": px.colors.qualitative.Pastel,
    "Prism": px.colors.qualitative.Prism,
    "Set1": px.colors.qualitative.Set1,
    "Safe (Okabe-Ito)": OKABE_ITO,
    "Viridis (seq)": px.colors.sequential.Viridis,
    "Cividis (seq)": px.colors.sequential.Cividis,
}

# ============================
# Streamlit Page Configuration
# ============================
st.set_page_config(
    page_title="Barras Agrupadas • Padrão Científico",
    layout="wide",
    page_icon="📊",
)

# ==================
# Session State Util
# ==================
if "preset" not in st.session_state:
    st.session_state.preset = "Científico"
if "config" not in st.session_state:
    st.session_state.config = {}

# =====================
# Utility: Safe to Image
# =====================
def fig_to_image_bytes(fig: go.Figure, fmt: str = "png", scale: float = 2.0) -> Optional[bytes]:
    """Return bytes for image export using kaleido if available. Fallback: None."""
    try:
        return fig.to_image(format=fmt, scale=scale)  # requires kaleido
    except Exception as e:
        st.warning(
            "Não consegui exportar imagem com o motor atual (Kaleido/Chrome). "
            "Baixe como HTML ou tente PNG novamente após instalar o Kaleido.")
        st.caption(f"Erro técnico: {e}")
        return None

# =====================
# Data Loading & Parsing
# =====================
def read_any_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    # Try several CSV-like options
    content = uploaded_file.read()
    for sep in [",", ";", "\t", " "]:
        for decimal in [".", ","]:
            try:
                df = pd.read_csv(io.BytesIO(content), sep=sep, decimal=decimal)
                if df.shape[1] >= 2:
                    return df
            except Exception:
                continue
    # Final attempt using python engine
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file, engine="python", sep=None)


def infer_long_format(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Try to infer columns for Category, Series, Value, Error (optional)."""
    cols = [c.lower() for c in df.columns]
    mapping = {lower: orig for lower, orig in zip(cols, df.columns)}

    cat = mapping.get("categoria") or mapping.get("category") or mapping.get("x")
    series = mapping.get("serie") or mapping.get("series") or mapping.get("grupo") or mapping.get("group")
    val = mapping.get("valor") or mapping.get("value") or mapping.get("y")
    err = mapping.get("erro") or mapping.get("error") or mapping.get("yerr")

    return cat, series, val, err


def to_long_format(df: pd.DataFrame, cat_col: str, value_cols: List[str], error_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Convert wide to long. value_cols are series; optional error columns aligned by order."""
    long_rows = []
    for i, vcol in enumerate(value_cols):
        ecol = error_cols[i] if error_cols and i < len(error_cols) else None
        for _, row in df[[cat_col, vcol] + ([ecol] if ecol else [])].iterrows():
            long_rows.append({
                "Category": row[cat_col],
                "Series": vcol,
                "Value": row[vcol],
                "Error": row[ecol] if ecol else np.nan,
            })
    return pd.DataFrame(long_rows)


# ======================
# Presets (look & feel)
# ======================
PRESETS = {
    "Científico": {
        "font": {"family": "Inter, Arial, sans-serif", "title": 20, "axes": 14, "ticks": 12, "legend": 12},
        "bargap": 0.15,
        "bargroupgap": 0.06,
        "legend": {"orientation": "h", "x": 0.5, "y": 1.15, "xanchor": "center", "yanchor": "bottom"},
        "palette": "Safe (Okabe-Ito)",
        "showgrid": True,
    },
    "Apresentação": {
        "font": {"family": "Inter, Arial, sans-serif", "title": 26, "axes": 18, "ticks": 14, "legend": 14},
        "bargap": 0.2,
        "bargroupgap": 0.1,
        "legend": {"orientation": "h", "x": 0.5, "y": 1.1, "xanchor": "center", "yanchor": "bottom"},
        "palette": "Plotly (qualitative)",
        "showgrid": False,
    },
    "Compacto": {
        "font": {"family": "Inter, Arial, sans-serif", "title": 16, "axes": 12, "ticks": 10, "legend": 10},
        "bargap": 0.1,
        "bargroupgap": 0.02,
        "legend": {"orientation": "v", "x": 1.02, "y": 1.0, "xanchor": "left", "yanchor": "top"},
        "palette": "D3",
        "showgrid": True,
    },
}

# ============================
# Sidebar: Layout & Controls
# ============================
left, right = st.columns([1, 3], gap="large")

with left:
    st.header("⚙️ Controles")
    st.subheader("Dados")

    uploaded = st.file_uploader(
        "Envie dados (CSV, XLSX, TXT)", type=["csv", "xlsx", "xls", "txt"], accept_multiple_files=False
    )

    df_raw = None
    if uploaded is not None:
        uploaded.seek(0)
        df_raw = read_any_table(uploaded)
        st.caption(f"Dimensões: {df_raw.shape[0]} linhas × {df_raw.shape[1]} colunas")
        st.dataframe(df_raw.head(), use_container_width=True)

    st.divider()

    preset = st.selectbox("Preset visual", list(PRESETS.keys()), index=list(PRESETS.keys()).index(st.session_state.preset))
    st.session_state.preset = preset
    preset_cfg = PRESETS[preset]

    with st.expander("🎨 Estilo do gráfico", expanded=True):
        pal_name = st.selectbox("Paleta de cores", list(PALETTES.keys()), index=list(PALETTES.keys()).index(preset_cfg["palette"]))
        opacity = st.slider("Transparência das barras", 0.2, 1.0, 1.0, 0.05)
        outline = st.slider("Espessura do contorno", 0.0, 3.0, 0.0, 0.1)
        bargap = st.slider("Espaço entre grupos (bargap)", 0.0, 0.4, float(preset_cfg["bargap"]), 0.01)
        bargroupgap = st.slider("Espaço entre barras do grupo (bargroupgap)", 0.0, 0.4, float(preset_cfg["bargroupgap"]), 0.01)

    with st.expander("🧭 Texto & Eixos", expanded=False):
        title = st.text_input("Título do gráfico", value="")
        x_label = st.text_input("Rótulo do eixo X", value="")
        y_label = st.text_input("Rótulo do eixo Y", value="")

        col_font1, col_font2 = st.columns(2)
        with col_font1:
            title_size = st.slider("Fonte do título", 10, 36, int(preset_cfg["font"]["title"]))
            axes_size = st.slider("Fonte dos eixos", 8, 28, int(preset_cfg["font"]["axes"]))
        with col_font2:
            tick_size = st.slider("Fonte dos ticks", 6, 24, int(preset_cfg["font"]["ticks"]))
            legend_size = st.slider("Fonte da legenda", 6, 24, int(preset_cfg["font"]["legend"]))

        x_rotate = st.slider("Rotação dos rótulos do X", 0, 90, 0, 5)
        y_scale = st.selectbox("Escala do eixo Y", ["linear", "log"], index=0)
        y_min, y_max = st.text_input("Limite Y mínimo (opcional)", value=""), st.text_input("Limite Y máximo (opcional)", value="")
        show_grid = st.checkbox("Mostrar gridlines", value=bool(preset_cfg["showgrid"]))

    with st.expander("🏷️ Legenda & Rótulos", expanded=False):
        legend_orientation = st.selectbox("Orientação da legenda", ["h", "v"], index=0)
        legend_pos = st.selectbox("Posição da legenda", [
            "top_outside", "right_outside", "bottom_inside", "top_inside"
        ], index=0)
        show_values = st.checkbox("Mostrar valor sobre as barras", value=False)
        value_fmt = st.text_input("Formato do valor (ex.: .2f, .1% )", value=".2f")
        value_size = st.slider("Tamanho da fonte do valor", 8, 24, 12)

    with st.expander("📦 Exportação", expanded=False):
        dpi = st.slider("DPI (escala)", 1, 6, 3)
        export_formats = st.multiselect("Formatos", ["png", "svg", "pdf", "html"], default=["png", "html"])

    with st.expander("💾 Presets & Config", expanded=False):
        if st.button("Salvar configuração atual (JSON)"):
            cfg = {
                "preset": preset,
                "palette": pal_name,
                "opacity": opacity,
                "outline": outline,
                "bargap": bargap,
                "bargroupgap": bargroupgap,
                "title": title,
                "x_label": x_label,
                "y_label": y_label,
                "fonts": {
                    "title": title_size,
                    "axes": axes_size,
                    "ticks": tick_size,
                    "legend": legend_size,
                },
                "x_rotate": x_rotate,
                "y_scale": y_scale,
                "y_min": y_min,
                "y_max": y_max,
                "show_grid": show_grid,
                "legend": {"orientation": legend_orientation, "position": legend_pos},
                "labels": {"show": show_values, "fmt": value_fmt, "size": value_size},
                "dpi": dpi,
                "export_formats": export_formats,
            }
            st.session_state.config = cfg
            b = io.BytesIO(json.dumps(cfg, indent=2).encode("utf-8"))
            st.download_button("⬇️ Baixar JSON de configuração", b, file_name="config_barras.json", mime="application/json")

        cfg_file = st.file_uploader("Carregar JSON de configuração", type=["json"], key="cfg")
        if cfg_file is not None:
            cfg_loaded = json.load(cfg_file)
            # Do minimal live apply
            st.info("Configuração carregada. Ajuste os controles acima conforme necessário.")
            # We don't force-override every widget to avoid UX jumps.

    st.divider()

    st.subheader("Mapeamento de colunas")
    long_mode = st.radio("Formato dos dados", ["Longo (Category, Series, Value, Error opcional)", "Largo (uma coluna por série)"])

    df_long = None
    error_col = None
    series_order: List[str] = []

    if df_raw is not None:
        if long_mode.startswith("Longo"):
            cat_guess, series_guess, val_guess, err_guess = infer_long_format(df_raw)
            cat_col = st.selectbox("Categoria (X)", df_raw.columns, index=(df_raw.columns.get_loc(cat_guess) if (cat_guess in df_raw.columns) else 0))
            series_col = st.selectbox("Série (grupo)", df_raw.columns, index=(df_raw.columns.get_loc(series_guess) if (series_guess in df_raw.columns) else 1))
            value_col = st.selectbox("Valor (Y)", df_raw.columns, index=(df_raw.columns.get_loc(val_guess) if (val_guess in df_raw.columns) else 2))
            error_col = st.selectbox("Erro (opcional)", ["<nenhum>"] + list(df_raw.columns), index=( ["<nenhum>"] + list(df_raw.columns) ).index(err_guess) if (err_guess in df_raw.columns) else 0)
            if error_col == "<nenhum>":
                error_col = None
            df_long = df_raw.rename(columns={cat_col: "Category", series_col: "Series", value_col: "Value"}).copy()
            if error_col:
                df_long["Error"] = df_raw[error_col]
            else:
                df_long["Error"] = np.nan
            series_order = list(df_long["Series"].unique())
        else:
            # Wide → select cat, value columns, optional error columns in parallel
            cat_col = st.selectbox("Categoria (X)", df_raw.columns, index=0, key="cat_wide")
            value_cols = st.multiselect("Colunas de valor (cada uma será uma série)", [c for c in df_raw.columns if c != cat_col])
            use_err = st.checkbox("Tenho colunas de erro correspondentes?", value=False)
            error_cols = []
            if use_err and value_cols:
                st.caption("Selecione na mesma ordem das séries.")
                error_cols = st.multiselect("Colunas de erro (mesma ordem)", [c for c in df_raw.columns if c not in [cat_col] + value_cols])
                if error_cols and len(error_cols) != len(value_cols):
                    st.warning("O número de colunas de erro deve coincidir com o número de séries.")
            if value_cols:
                df_long = to_long_format(df_raw, cat_col, value_cols, error_cols if use_err else None)
                series_order = value_cols

    manual_colors: Dict[str, str] = {}
    if df_long is not None:
        with st.expander("🎯 Cores por série (manual)", expanded=False):
            series_sorted = sorted(series_order) if series_order else sorted(df_long["Series"].unique())
            for s in series_sorted:
                manual_colors[s] = st.color_picker(f"Cor para {s}", value=None) or ""

# =======================
# Figure Builder Function (SAFE LAYOUT)
# =======================
def build_grouped_bar(
    df_long: pd.DataFrame,
    palette_name: str,
    opacity: float,
    outline: float,
    bargap: float,
    bargroupgap: float,
    title: str,
    x_label: str,
    y_label: str,
    font_sizes: Dict[str, int],
    x_rotate: int,
    y_scale: str,
    y_min: str,
    y_max: str,
    show_grid: bool,
    legend_orientation: str,
    legend_pos: str,
    show_values: bool,
    value_fmt: str,
    value_size: int,
    manual_colors: Optional[Dict[str, str]] = None,
) -> go.Figure:
    pal = PALETTES.get(palette_name, px.colors.qualitative.Plotly)

    # Mapa de cores (respeita manual)
    series = list(pd.unique(df_long["Series"]))
    color_map = {}
    i = 0
    for s in series:
        if manual_colors and manual_colors.get(s):
            color_map[s] = manual_colors[s]
        else:
            color_map[s] = pal[i % len(pal)]
            i += 1

    fig = go.Figure()

    for s in series:
        sub = df_long[df_long["Series"] == s]
        y_vals = sub["Value"].to_numpy(dtype=float)
        err = sub["Error"].to_numpy(dtype=float) if "Error" in sub.columns else np.full_like(y_vals, np.nan)
        has_err = np.isfinite(err).any()

        text_vals = None
        if show_values:
            try:
                text_vals = [f"{v:{value_fmt}}" for v in y_vals]
            except Exception:
                text_vals = [f"{v:.2f}" for v in y_vals]

        fig.add_trace(
            go.Bar(
                name=str(s),
                x=sub["Category"],
                y=y_vals,
                marker_color=color_map.get(s),
                marker_line_width=outline,
                opacity=opacity,
                error_y=dict(type="data", array=err, visible=bool(has_err)) if bool(has_err) else None,
                text=text_vals,
                textposition="outside" if show_values else None,
                textfont=dict(size=value_size) if show_values else None,
            )
        )

    # Posição da legenda
    legend = {"orientation": legend_orientation}
    if legend_pos == "top_outside":
        legend.update(x=0.5, y=1.15, xanchor="center", yanchor="bottom")
    elif legend_pos == "right_outside":
        legend.update(x=1.02, y=1.0, xanchor="left", yanchor="top")
    elif legend_pos == "bottom_inside":
        legend.update(x=0.5, y=-0.2, xanchor="center", yanchor="top")
    elif legend_pos == "top_inside":
        legend.update(x=0.5, y=1.02, xanchor="center", yanchor="bottom")

    # Escala Y segura
    y_scale_final = y_scale
    if y_scale == "log" and (df_long["Value"] <= 0).any():
        y_scale_final = "linear"
        try:
            st.warning("Valores ≤ 0 detectados. Escala Y ajustada para 'linear'.")
        except Exception:
            pass

    # Limites Y: só se ambos forem válidos
    y_range = None
    try:
        y0 = float(y_min) if y_min not in (None, "") else None
        y1 = float(y_max) if y_max not in (None, "") else None
        if (y0 is not None) and (y1 is not None):
            if not (y_scale_final == "log" and (y0 <= 0 or y1 <= 0)):
                y_range = [y0, y1]
    except Exception:
        y_range = None

    # Monta kwargs sem None
    layout_kwargs = dict(
        barmode="group",
        bargap=bargap,
        bargroupgap=bargroupgap,
        xaxis=dict(
            title=x_label if x_label else None,
            tickangle=x_rotate,
            titlefont=dict(size=font_sizes["axes"]),
            tickfont=dict(size=font_sizes["ticks"]),
            showgrid=show_grid,
        ),
        yaxis=dict(
            title=y_label if y_label else None,
            type=y_scale_final,
            titlefont=dict(size=font_sizes["axes"]),
            tickfont=dict(size=font_sizes["ticks"]),
            showgrid=show_grid,
        ),
        legend=legend,
        margin=dict(l=40, r=40, t=80, b=80),
    )

    if title:
        layout_kwargs["title"] = dict(text=title, x=0.5, font=dict(size=font_sizes["title"]))
    if y_range is not None:
        layout_kwargs["yaxis"]["range"] = y_range

    # Remove chaves None nos eixos
    for ax in ("xaxis", "yaxis"):
        layout_kwargs[ax] = {k: v for k, v in layout_kwargs[ax].items() if v is not None}

    fig.update_layout(**layout_kwargs)
    return fig

# ==================
# Main Plot Section
# ==================
with right:
    st.header("📊 Gráfico de Barras Agrupadas")
    if 'df_raw' not in locals() or df_raw is None:
        st.info("Envie um arquivo de dados na barra lateral para começar.")
    else:
        if 'df_long' not in locals() or df_long is None:
            st.warning("Configure o mapeamento de colunas para gerar o gráfico.")
        else:
            fig = build_grouped_bar(
                df_long=df_long,
                palette_name=pal_name,
                opacity=opacity,
                outline=outline,
                bargap=bargap,
                bargroupgap=bargroupgap,
                title=title,
                x_label=x_label,
                y_label=y_label,
                font_sizes={"title": title_size, "axes": axes_size, "ticks": tick_size, "legend": legend_size},
                x_rotate=x_rotate,
                y_scale=y_scale,
                y_min=y_min,
                y_max=y_max,
                show_grid=show_grid,
                legend_orientation=legend_orientation,
                legend_pos=legend_pos,
                show_values=show_values,
                value_fmt=value_fmt,
                value_size=value_size,
                manual_colors=manual_colors,
            )

            st.plotly_chart(fig, use_container_width=True, config={
                "displaylogo": False,
                "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"],
            })

            # ============
            # Export Area
            # ============
            st.subheader("Exportar")

            col_exp1, col_exp2, col_exp3 = st.columns(3)

            # Image export
            with col_exp1:
                if any(f in export_formats for f in ("png", "svg", "pdf")):
                    fmt = st.selectbox("Formato imagem", [f for f in export_formats if f in ("png", "svg", "pdf")], index=0)
                    img_bytes = fig_to_image_bytes(fig, fmt=fmt, scale=float(dpi))
                    if img_bytes:
                        fname = f"barras_agrupadas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{fmt}"
                        st.download_button("⬇️ Baixar imagem", data=img_bytes, file_name=fname, mime=f"image/{fmt}")

            # HTML export (no kaleido needed)
            with col_exp2:
                if "html" in export_formats:
                    html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
                    fname = f"barras_agrupadas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                    st.download_button("⬇️ Baixar HTML interativo", data=html_bytes, file_name=fname, mime="text/html")

            # Data export
            with col_exp3:
                df_proc = df_long.copy()
                csv_bytes = df_proc.to_csv(index=False).encode("utf-8")
                xlsx_buf = io.BytesIO()
                with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
                    df_proc.to_excel(writer, index=False, sheet_name="dados")
                st.download_button("⬇️ Baixar CSV (dados)", data=csv_bytes, file_name="dados_barras.csv", mime="text/csv")
                st.download_button("⬇️ Baixar XLSX (dados)", data=xlsx_buf.getvalue(), file_name="dados_barras.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ==============
# Footer / Notes
# ==============
st.caption("Dica: salve um JSON de configuração e reutilize em outros apps para padronizar estilos e exportações.")

# =====================
# requirements.txt (sugestão)
# =====================
# Coloque isto em um arquivo separado chamado requirements.txt no seu repositório:
#
# streamlit>=1.36
# plotly==5.22.0
# kaleido==0.2.1
# pandas>=2.1
# numpy>=1.26
# openpyxl>=3.1
# xlsxwriter>=3.1


