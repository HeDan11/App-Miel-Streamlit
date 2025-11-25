import os
import glob
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# =========================================
# ========== CONFIGURACIÓN GENERAL ========
# =========================================

st.set_page_config(
    page_title="Detector de Adulteración de Miel",
    page_icon="🍯",
    layout="wide"
)

# Nombre del CSV de ejemplo dentro de la carpeta de modelos
DEMO_FILENAME = "adulteration_dataset_26_08_2021.csv"

# === CSS temática miel / ámbar ===
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #FFF8E1 0%, #FFF3CD 40%, #FFE0B2 100%);
    }

    h1, h2, h3 {
        color: #6D4C41;
        font-family: "Helvetica", "Segoe UI", sans-serif;
    }

    section[data-testid="stSidebar"] {
        background-color: #FFF3CD;
        border-right: 1px solid #FFECB3;
    }

    .honey-card {
        padding: 1rem 1.2rem;
        border-radius: 0.8rem;
        background-color: #FFF8E1;
        border: 1px solid #FFECB3;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.08);
    }

    .honey-card-strong {
        padding: 1.3rem 1.4rem;
        border-radius: 1rem;
        background: linear-gradient(135deg, #FFECB3, #FFE082);
        border: 1px solid #FBC02D;
        box-shadow: 0px 3px 10px rgba(0,0,0,0.15);
    }

    .honey-pill {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        background-color: #FFFDE7;
        border: 1px solid #FFE082;
        font-size: 0.8rem;
        color: #8D6E63;
        margin-right: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🍯 Detector de Adulteración de Miel (Clases combinadas)")
st.markdown(
    "Esta app carga **modelos preentrenados** y te permite evaluar **espectros de miel** "
    "para estimar si la muestra está adulterada y en qué nivel."
)

# =========================================
# ============ UTILIDADES BÁSICAS =========
# =========================================

@st.cache_data(show_spinner=False)
def list_model_pkls(models_dir: str):
    paths = sorted(glob.glob(os.path.join(models_dir, "*.pkl")))
    names = [os.path.basename(p) for p in paths]
    return names, paths

@st.cache_resource(show_spinner=True)
def load_model_payload(pkl_path: str):
    payload = joblib.load(pkl_path)
    model = payload["model"]
    meta  = payload.get("metadata", {})
    model_name   = meta.get("model_name", "UnknownModel")
    version      = meta.get("version", "v1")
    task_type    = meta.get("task_type", "multiclass")
    target_name  = meta.get("target_name", None)
    classes      = np.array(meta.get("classes", []))
    feature_cols = meta.get("feature_names", [])
    return model, {
        "model_name": model_name,
        "version": version,
        "task_type": task_type,      # "binary" | "multiclass"
        "target_name": target_name,  # nombre de la columna real en tu CSV (si la incluyes)
        "classes": classes,
        "feature_names": feature_cols,
        "raw_meta": meta,
    }

@st.cache_data(show_spinner=False)
def load_example_dataset(models_dir: str) -> pd.DataFrame:
    """
    Carga el CSV de ejemplo desde la misma carpeta donde viven los modelos.
    Espera encontrar: models_dir / DEMO_FILENAME
    """
    demo_path = os.path.join(models_dir, DEMO_FILENAME)
    if not os.path.exists(demo_path):
        raise FileNotFoundError(
            f"No se encontró el dataset de ejemplo en: {demo_path}"
        )
    df = pd.read_csv(demo_path)
    return df

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            continue
        out[c] = pd.to_numeric(out[c], errors="ignore")
    return out

def align_features(df: pd.DataFrame, feature_names: list):
    missing = [c for c in feature_names if c not in df.columns]
    extra   = [c for c in df.columns if c not in feature_names]
    ok = len(missing) == 0
    return ok, missing, extra, df.reindex(columns=feature_names, fill_value=np.nan)

def predict_with_any(model, X_row_2d: np.ndarray, classes: np.ndarray):
    """
    Retorna (y_pred_label, proba_dict)
    - Usa predict_proba si existe,
    - Si no, usa decision_function y convierte (sigmoid/softmax).
    """
    y_pred = model.predict(X_row_2d)
    y_pred = np.asarray(y_pred).ravel()
    pred_label = y_pred[0] if len(y_pred) > 0 else None

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            p = model.predict_proba(X_row_2d)
            proba = p[0]
        except Exception:
            proba = None
    if proba is None and hasattr(model, "decision_function"):
        scores = model.decision_function(X_row_2d)
        scores = np.atleast_2d(scores)
        if scores.shape[1] == 1:
            scores = np.concatenate([-scores, scores], axis=1)
        if classes is not None and len(classes) == 2:
            s = scores[0, -1]
            p1 = _sigmoid(s)
            proba = np.array([1.0 - p1, p1])
        else:
            proba = _softmax(scores, axis=1)[0]

    proba_dict = None
    if proba is not None and classes is not None and len(classes) == len(proba):
        proba_dict = dict(zip(map(str, classes), map(float, proba)))
    return pred_label, proba_dict

def extract_wavelengths(feature_names):
    """
    Intenta convertir nombres tipo '400nm', '450nm' -> 400, 450.
    Si no puede, regresa índices 0,1,2,...
    """
    wls = []
    for i, c in enumerate(feature_names):
        wl = None
        if isinstance(c, str) and "nm" in c:
            try:
                wl = float(c.replace("nm", "").strip())
            except Exception:
                wl = None
        if wl is None:
            wl = float(i)
        wls.append(wl)
    return np.array(wls)

def interpret_honey_message(pred_label):
    """
    Convierte la clase GANADORA en un mensaje amigable.
    Convención:
      - clase 0  -> muestra NO adulterada
      - clase !=0 -> muestra adulterada
    """
    try:
        val = float(pred_label)
    except Exception:
        # fallback textual
        label_str = str(pred_label)
        if label_str in ["0", "pura", "no adulterada"]:
            return "🟢 La muestra de miel **no está adulterada**.", "no_adulterada"
        else:
            return "🟠 La muestra de miel **presenta signos de adulteración**.", "adulterada"

    if abs(val) < 1e-9:  # 0
        msg = "🟢 La muestra de miel **no está adulterada** (clase 0)."
        return msg, "no_adulterada"
    else:
        msg = (
            "🟠 La muestra de miel **presenta adulteración**.\n\n"
            f"La clase ganadora es **{pred_label}**, asociada a un nivel de adulteración distinto de cero, especificamente la muestra de miel seleccionada tiene un **{pred_label}**% de adulteracion."
        )
        return msg, "adulterada"


# =========================================
# =============== SIDEBAR =================
# =========================================

st.sidebar.header("⚙️ Configuración del modelo")

models_dir = st.sidebar.text_input("Carpeta de modelos (.pkl):", value="models")
model_files, model_paths = list_model_pkls(models_dir)

if not model_files:
    st.sidebar.warning("No se encontraron `.pkl` en la carpeta indicada.")
    st.info(
        "Coloca tus archivos `.pkl` y el CSV de ejemplo "
        f"`{DEMO_FILENAME}` en una carpeta llamada **models/** "
        "(junto a este `app.py`), o cambia la ruta en el sidebar."
    )
    st.stop()

model_choice = st.sidebar.selectbox("Selecciona el modelo preentrenado", model_files, index=0)
model_path   = model_paths[model_files.index(model_choice)]
model, meta  = load_model_payload(model_path)

with st.sidebar.expander("ℹ️ Metadata del modelo"):
    st.write(f"**Nombre:** `{meta['model_name']}`")
    st.write(f"**Versión:** `{meta['version']}`")
    st.write(f"**Tarea:** `{meta['task_type']}`")
    st.write(f"**Target esperado:** `{meta['target_name']}`")
    st.write("**Clases del modelo:**")
    st.write(", ".join(map(str, meta["classes"])))
    st.caption("`feature_names` define el orden de columnas que el modelo espera para predecir.")

st.sidebar.markdown("---")
auto_random = st.sidebar.checkbox("Elegir muestra aleatoria al cargar dataset", value=False)

# =========================================
# =========== CARGA DEL DATASET ===========
# =========================================

st.header("1️⃣ Elige la fuente de datos de miel")

data_mode = st.radio(
    "¿De dónde quieres tomar las muestras?",
    ["Usar dataset de ejemplo (recomendado)", "Subir mi propio CSV"],
    horizontal=True
)

df = None

if data_mode == "Usar dataset de ejemplo (recomendado)":
    try:
        df = load_example_dataset(models_dir)
        st.success(
            f"Usando el **dataset de ejemplo** `{DEMO_FILENAME}` desde la carpeta de modelos."
        )
        st.caption(
            "Este dataset de ejemplo contiene espectros de miel ya medidos en el laboratorio. "
            "Puedes explorar muestras, ver su espectro y el diagnóstico de adulteración."
        )
    except FileNotFoundError as e:
        st.error(
            f"No encontré el archivo de ejemplo `{DEMO_FILENAME}` en la carpeta de modelos.\n\n"
            f"Ruta buscada: `{os.path.join(models_dir, DEMO_FILENAME)}`\n\n"
            "Asegúrate de que exista ese archivo o selecciona la opción de subir tu propio CSV."
        )
        st.stop()
else:
    uploaded = st.file_uploader(
        "Sube un CSV con las columnas espectrales (y, si quieres, la columna de clase real).",
        type=["csv"]
    )

    if uploaded is None:
        st.info("Sube un archivo **CSV** o cambia a la opción de dataset de ejemplo.")
        st.stop()

    try:
        df = pd.read_csv(uploaded)
        st.success("CSV cargado correctamente.")
    except Exception as e:
        st.error(f"No pude leer el CSV: {e}")
        st.stop()

df = ensure_numeric(df)
st.write("🔍 Vista previa del dataset:")
st.dataframe(df.head())

# Alineamos columnas con las columnas esperadas por el modelo
ok, missing, extra, df_aligned = align_features(df, meta["feature_names"])

if not ok:
    st.error(
        "⚠️ Faltan columnas necesarias para este modelo:\n\n"
        + "\n".join(f"- {c}" for c in missing)
        + "\n\nRevisa que tu CSV (o el dataset de ejemplo) incluya **exactamente** las columnas esperadas por el modelo."
    )
    if extra:
        st.warning(
            "Estas columnas sobran para el modelo (no pasa nada, solo no se usan):\n\n"
            + "\n".join(f"- {c}" for c in extra)
        )
    st.stop()

# =========================================
# === TARGET REAL (para comparación) ======
# =========================================

target_col  = meta.get("target_name", None)
compare_col = target_col  # por defecto

if target_col and target_col in df.columns:
    model_classes = set()
    try:
        model_classes = set(map(int, meta["classes"])) if len(meta["classes"]) else set()
    except Exception:
        try:
            model_classes = set(int(c) for c in meta["classes"])
        except Exception:
            model_classes = set(meta["classes"])

    # Si el modelo no tiene clase 5 pero sí 0, fusiona 5→0 para comparar
    if (5 not in model_classes) and (0 in model_classes):
        compare_col = f"{target_col}_merged"
        df[compare_col] = df[target_col].replace({5: 0})
        st.caption(
            f"ℹ️ Se creó la columna **{compare_col}** para comparar con el modelo, "
            "fusionando la clase 5 dentro de la clase 0 (clases combinadas)."
        )

# =========================================
# ======== SELECCIÓN DE INSTANCIA =========
# =========================================

st.header("2️⃣ Selecciona la muestra a evaluar")

# Reset de aleatorio cuando cambia dataset
if "last_n_rows" not in st.session_state or st.session_state.last_n_rows != len(df_aligned):
    st.session_state.row_index = 0
    st.session_state.already_random = False
    st.session_state.last_n_rows = len(df_aligned)

if "row_index" not in st.session_state:
    st.session_state.row_index = 0

max_idx = max(0, len(df_aligned) - 1)
if st.session_state.row_index > max_idx:
    st.session_state.row_index = max_idx

c1, c2 = st.columns([3, 1])

with c1:
    mode_choice = st.radio(
        "¿Cómo quieres elegir la muestra?",
        ["Elegir índice manualmente", "Elegir una muestra aleatoria"],
        horizontal=True
    )

    if mode_choice == "Elegir índice manualmente":
        st.session_state.row_index = st.number_input(
            "Índice de fila (0 = primera fila del CSV)",
            min_value=0, max_value=max_idx,
            value=int(st.session_state.row_index),
            step=1
        )
    else:
        if auto_random and st.session_state.get("already_random", False) is False:
            st.session_state.row_index = int(np.random.randint(0, len(df_aligned)))
            st.session_state.already_random = True

        if st.button("🎲 Elegir muestra aleatoria ahora"):
            st.session_state.row_index = int(np.random.randint(0, len(df_aligned)))

with c2:
    show_meta_body = st.checkbox("Mostrar metadata técnica al final", value=False)

row_index = int(st.session_state.row_index)

st.markdown(f"**Muestra seleccionada:** índice `{row_index}` (de 0 a {max_idx})")

X_row = df_aligned.iloc[[row_index]]  # 2D align
X_row_display = df.iloc[[row_index]]  # tal como viene del CSV

st.markdown("#### Detalle de la muestra")
st.dataframe(X_row_display)

# =========================================
# ========== PREDICCIÓN Y MENSAJES ========
# =========================================

st.header("3️⃣ Resultado del modelo")

pred_label, proba_dict = predict_with_any(
    model,
    X_row.to_numpy(dtype=float),
    meta["classes"]
)

left_col, right_col = st.columns([1.4, 1.6])

with left_col:
    st.markdown('<div class="honey-card-strong">', unsafe_allow_html=True)
    st.subheader("🔮 Diagnóstico para esta muestra")

    if pred_label is None:
        st.error("No se pudo obtener una predicción del modelo.")
    else:
        msg, status = interpret_honey_message(pred_label)
        st.markdown(msg)

        st.markdown("---")
        st.markdown(f"**Clase predicha por el modelo:** `{pred_label}`")

        if compare_col and compare_col in df.columns:
            true_label = df.loc[df.index[row_index], compare_col]
            match = (str(true_label) == str(pred_label))
            if match:
                st.markdown(f"**Clase real (comparada):** `{true_label}` ✅ (coincide con el modelo)")
            else:
                st.markdown(f"**Clase real (comparada):** `{true_label}` ❌ (no coincide con el modelo)")

            if target_col and compare_col != target_col:
                orig_lbl = df.loc[df.index[row_index], target_col]
                if str(orig_lbl) != str(true_label):
                    st.caption(
                        f"En el CSV original, la clase era `{orig_lbl}`. "
                        "Para este modelo de clases combinadas se compara como `0` (fusión 5→0)."
                    )
        else:
            st.caption(
                "Tu dataset no incluye la columna de clase esperada, "
                "por eso no se compara con un valor 'real' de referencia."
            )

    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="honey-card">', unsafe_allow_html=True)
    st.subheader("📊 Probabilidades por clase")

    if proba_dict is None:
        st.info(
            "El modelo no expone probabilidades. "
            "Si internamente tiene `decision_function`, intento aproximar probabilidades; "
            "si no, sólo se muestra la clase ganadora."
        )
    else:
        proba_items = sorted(proba_dict.items(), key=lambda kv: kv[1], reverse=True)
        proba_df = pd.DataFrame(proba_items, columns=["Clase", "Probabilidad"])

        # Clase ganadora
        top_class = proba_df.loc[0, "Clase"]

        chart = (
            alt.Chart(proba_df)
            .mark_bar()
            .encode(
                x=alt.X("Clase:N", sort=None, title="Clase"),
                y=alt.Y("Probabilidad:Q", title="Probabilidad"),
                color=alt.condition(
                    alt.datum.Clase == top_class,
                    alt.value("#F9A825"),   # ámbar intenso para ganadora
                    alt.value("#FFE082")    # amarillo claro para el resto
                ),
                tooltip=["Clase", alt.Tooltip("Probabilidad:Q", format=".4f")]
            )
            .properties(height=260)
        )
        st.altair_chart(chart, use_container_width=True)

        with st.expander("Ver valores numéricos"):
            st.dataframe(proba_df.style.format({"Probabilidad": "{:.4f}"}))

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================
# ========== ESPECTRO DE LA MUESTRA =======
# =========================================

st.header("4️⃣ Espectro de la muestra seleccionada")

wavelengths = extract_wavelengths(meta["feature_names"])
intensities = X_row.to_numpy(dtype=float).ravel()

spec_df = pd.DataFrame(
    {
        "Longitud de onda (nm)": wavelengths,
        "Intensidad": intensities,
    }
)

spec_chart = (
    alt.Chart(spec_df)
    .mark_line(point=True)
    .encode(
        x=alt.X("Longitud de onda (nm):Q", title="Longitud de onda (nm)"),
        y=alt.Y("Intensidad:Q", title="Intensidad (unidades arbitrarias)"),
        tooltip=[
            alt.Tooltip("Longitud de onda (nm):Q", format=".0f"),
            alt.Tooltip("Intensidad:Q", format=".4f"),
        ],
    )
    .properties(height=320)
)

st.markdown(
    "El siguiente gráfico muestra el **espectro de la muestra seleccionada**, "
    "tal como lo ve el modelo (una medida por longitud de onda):"
)
st.altair_chart(spec_chart, use_container_width=True)

# =========================================
# ========== INFO TÉCNICA EXTRA ===========
# =========================================

if show_meta_body:
    st.header("🔧 Metadata técnica del modelo")
    st.json(meta["raw_meta"])
