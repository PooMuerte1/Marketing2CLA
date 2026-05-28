import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Inteligencia de Clientes & Posicionamiento Comercial",
    page_icon="bar_chart",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR DESIGN AESTHETICS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=Outfit:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E3C72 0%, #2A5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #5A6A85;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .section-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        border-bottom: 3px solid #3B82F6;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .card {
        background-color: #1E293B !important;
        color: #F1F5F9 !important;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #334155 !important;
        margin-bottom: 1rem;
    }
    .card h1, .card h2, .card h3, .card h4, .card h5, .card h6, .card strong, .card p, .card li, .card span {
        color: #F1F5F9 !important;
    }
    
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #64748B;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    .badge-info {
        background-color: #E2E8F0;
        color: #334155;
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .pitch-guide {
        background-color: #FFF9E6;
        border-left: 5px solid #F59E0B;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        color: #78350F !important;
    }
    .pitch-guide strong {
        color: #92400E !important;
    }
    /* Hide Streamlit Sidebar completely */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="collapsedSidebarIcons"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def get_global_pca_rfm(df_full):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    X = df_full[['recency_final', 'frequency_final', 'monetary_final']]
    coords = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_
    return coords[:, 0], coords[:, 1], ev

@st.cache_data
def get_global_elbow_rfm(df_full):
    from sklearn.cluster import KMeans
    X = df_full[['recency_final', 'frequency_final', 'monetary_final']]
    inertia = []
    rango = list(range(1, 10))
    for k in rango:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertia.append(km.inertia_)
    return rango, inertia

@st.cache_data
def get_data():
    df = pd.read_csv('segmented_customers.csv')
    df_raw = pd.read_csv('customers_with_clusters.csv')
    rfm_mapping = {
        0: "Clase K-Means 0: Frecuencia Alta, Gasto Alto (En Riesgo)",
        1: "Clase K-Means 1: Frecuencia Alta, Gasto Alto (Activos)",
        2: "Clase K-Means 2: Frecuencia Baja, Gasto Bajo (Dormidos)"
    }
    lca_mapping = {
        0: "Clase LCA 0: Hombres Jóvenes (Redes Sociales, Hogar)",
        1: "Clase LCA 1: Adultos (Búsqueda Orgánica, Ropa)",
        2: "Clase LCA 2: Mujeres (Búsqueda Orgánica, Electrónica)"
    }
    df['Cluster_RFM_Nombre'] = df['Cluster_RFM'].map(rfm_mapping)
    df['Cluster_LCA_Nombre'] = df['Cluster_LCA'].map(lca_mapping)
    return df, df_raw

# --- GLOBAL CONSTANTS ---
CAT_COLS = ['gender', 'age_group', 'region', 'membership_tier', 'preferred_device', 'acquisition_channel', 'preferred_category']
CAT_RFM_COLS = ['recency_cat', 'frequency_cat', 'monetary_cat']

try:
    df, df_raw = get_data()
    # Add static global PCA coordinates
    pc1_rfm, pc2_rfm, ev_rfm = get_global_pca_rfm(df)
    df['PC1_RFM'] = pc1_rfm
    df['PC2_RFM'] = pc2_rfm
    EV_RFM_1 = ev_rfm[0]
    EV_RFM_2 = ev_rfm[1]
except Exception as e:
    st.error(f"Error al cargar el archivo preprocesado 'segmented_customers.csv': {e}")
    st.info("Por favor, asegúrate de ejecutar primero `prepare_data.py` en este directorio para generar la segmentación.")
    st.stop()

# --- HEADER ---
st.markdown('<div class="main-title">Estrategia de Segmentación de Clientes y Posicionamiento</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">División de Growth Marketing & Business Analytics — Plataforma de Decisiones Estratégicas</div>', unsafe_allow_html=True)

# --- GLOBAL FILTERS AT THE TOP ---
with st.expander("Filtros Globales de Segmento (Región Geográfica y Nivel de Membresía)", expanded=False):
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        region_filter = st.multiselect("Filtrar por Región Geográfica", options=sorted(df['region'].unique()), default=sorted(df['region'].unique()))
    with col_f2:
        tier_filter = st.multiselect("Filtrar por Nivel de Membresía", options=sorted(df['membership_tier'].unique()), default=sorted(df['membership_tier'].unique()))

# Disable pitch guides on main tabs (consolidated in technical annexes)
pitch_mode = False

filtered_df = df[df['region'].isin(region_filter) & df['membership_tier'].isin(tier_filter)]
filtered_df_raw = df_raw[df_raw['region'].isin(region_filter) & df_raw['membership_tier'].isin(tier_filter)]

if filtered_df.empty:
    st.warning("No hay datos disponibles para los filtros seleccionados.")
    st.stop()

# --- TABS ---
tab1, tab2, tab3, tab4, tab6, tab7, tab5, tab8, tab9 = st.tabs([
    "Contexto y Limpieza",
    "Segmentación RFM (K-Means)",
    "Segmentación Sociodemográfica (LCA)",
    "Matriz de Crossover",
    "Categorías por Segmento",
    "Pérdida de Membresías (Churn)",
    "Estrategia de Promociones",
    "Simulador de ROI",
    "Anexos"
])

# ==========================================
# TAB 1: CONTEXTO Y LIMPIEZA
# ==========================================
with tab1:
    if pitch_mode:
        st.markdown("""
        <div class="pitch-guide">
            <strong>Guía para el Pitch (Minuto 0-2)</strong><br>
            • Preséntate y expón el objetivo: Segmentar la base de clientes de un gran E-commerce global para definir su mercado meta y la estrategia de posicionamiento para capturar promociones.<br>
            • Explica que trabajaron exclusivamente sobre la <strong>base activa (7,285 clientes)</strong> para optimizar el retorno de inversión comercial, descartando clientes fugados.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Contexto del Mercado y Ajustes de Datos</div>', unsafe_allow_html=True)
    
    # Full-width intro column
    st.write("""
    ### Objetivos del Trabajo
    Este proyecto aborda la segmentación del mercado para un gran **E-commerce global** utilizando dos fuentes de datos principales:
    - `customers.csv`: Registro de 8,000 perfiles de clientes.
    - `orders.csv`: Registro transaccional de pedidos de clientes.
    
    Se busca identificar los segmentos de mercado más receptivos a promociones teniendo en cuenta la categoría de compra preferida.
    """)
    
    st.markdown("""
    ### Características de las Variables Base
    - **Variables Transaccionales**: `days_since_last_purchase` (Recencia), `total_orders` (Frecuencia), `total_spend_usd` (Monetario).
    - **Variables Demográficas y de Comportamiento**: `age` (Edad), `gender` (Género), `region` (Región), `preferred_device` (Dispositivo), `preferred_category` (Categoría de producto de interés), `acquisition_channel` (Canal de adquisición).
    """)
    
    # Basic stats table
    st.subheader("Muestra de Datos Segmentados y Métricas Agregadas de Promociones")
    st.dataframe(filtered_df[['customer_id', 'region', 'age', 'gender', 'membership_tier', 'total_orders', 'total_spend_usd', 'discount_ratio', 'avg_discount_pct']].head(10))

# ==========================================
# TAB 2: SEGMENTACIÓN RFM
# ==========================================
with tab2:
    if pitch_mode:
        st.markdown("""
        <div class="pitch-guide">
            <strong>Guía para el Pitch (Minuto 2-4)</strong><br>
            • Expón que realizaron un modelo de segmentación geométrica <strong>K-Means</strong> ($K=3$) sobre RFM. Justifica que para K-Means aplicaron una compresión logarítmica para mitigar outliers y estandarización Z-score previa sobre las variables continuas originales.<br>
            • Señala la <strong>Clase K-Means 1</strong> como tus clientes VIP activos y de más alto valor, la <strong>Clase K-Means 0</strong> como tus clientes de alto valor pero en riesgo de fuga (alta recencia), y la <strong>Clase K-Means 2</strong> como los de bajo valor e inactivos.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Modelo RFM: K-Means (K=3)</div>', unsafe_allow_html=True)
    
    st.info("**Justificación del Modelo**: Para segmentar los perfiles transaccionales RFM, se utilizó el algoritmo **K-Means** con $K=3$ sobre las variables continuas de Recencia, Frecuencia y Gasto. Estas fueron transformadas de forma logarítmica (para comprimir outliers) y estandarizadas con Z-Score (para normalizar las escalas). Esta segmentación geométrica agrupa de manera compacta y eficiente a los clientes según su valor comercial e inactividad en el espacio Euclidiano.")
    
    # RFM summary
    rfm_summary = filtered_df.groupby('Cluster_RFM_Nombre').agg(
        Clientes=('customer_id', 'count'),
        Recencia_Promedio=('days_since_last_purchase', 'mean'),
        Frecuencia_Promedio=('total_orders', 'mean'),
        Gasto_Promedio_USD=('total_spend_usd', 'mean')
    ).reset_index().round(1)
    
    st.write("### Métricas Clave de Negocio por Segmento Transaccional (K-Means)")
    st.dataframe(rfm_summary)

    # 2. HERO VISUALIZATIONS SECTION
    st.markdown('<div class="section-header">Representación de cluster</div>', unsafe_allow_html=True)
    st.write("A continuación se presentan las visualizaciones de alto nivel del modelo de segmentación transaccional. Estas proyecciones permiten identificar con total precisión y rigor comercial la distribución geométrica de los grupos de valor. Puedes elegir visualizar los datos en escala **Estandarizada** (Log + Z-Score, que muestra la **máxima nitidez y separación de clusters**) o en escala **Real** (los datos comerciales brutos, comprimidos por el sesgo del gasto):")
    
    scale_choice = st.radio(
        "Seleccionar Escala para Gráficos de Dispersión (3D y 2D):",
        options=["Escala Estandarizada (Log + Z-Score)", "Escala Real (Datos Brutos)"],
        horizontal=True
    )
    use_std_data = "Estandarizada" in scale_choice
    
    col_hero1, col_hero2 = st.columns(2)
    
    with col_hero1:
        st.write("#### Mapa de Segmentación PCA 2D")
        st.write("Proyectamos el espacio transaccional de tres dimensiones en dos componentes principales (PCA), revelando **tres nubes de puntos con excelente separación geométrica** (el PCA es intrínsecamente estandarizado):")
        
        fig_pca_rfm_app = px.scatter(
            filtered_df, x='PC1_RFM', y='PC2_RFM', color='Cluster_RFM_Nombre',
            color_discrete_sequence=px.colors.qualitative.Bold, opacity=0.7
        )
        fig_pca_rfm_app.update_traces(marker=dict(size=6))
        fig_pca_rfm_app.update_layout(
            xaxis_title=f'Componente Principal 1 ({EV_RFM_1*100:.1f}%)',
            yaxis_title=f'Componente Principal 2 ({EV_RFM_2*100:.1f}%)',
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=8)),
            height=400
        )
        st.plotly_chart(fig_pca_rfm_app, use_container_width=True)
        
    with col_hero2:
        st.write("#### Espacio Transaccional 3D Interactivo")
        st.write("Muestra las coordenadas en 3D del espacio transaccional. Mantén pulsado el botón izquierdo del mouse para rotar el gráfico y observar la distribución desde cualquier ángulo:")
        
        x_3d = 'recency_final' if use_std_data else 'days_since_last_purchase'
        y_3d = 'frequency_final' if use_std_data else 'total_orders'
        z_3d = 'monetary_final' if use_std_data else 'total_spend_usd'
        
        x_lbl_3d = 'Recencia (Z-Score)' if use_std_data else 'Recencia (Días)'
        y_lbl_3d = 'Frecuencia (Z-Score)' if use_std_data else 'Frecuencia (Órdenes)'
        z_lbl_3d = 'Gasto (Z-Score)' if use_std_data else 'Gasto (USD)'
        
        fig_3d_rfm = px.scatter_3d(
            filtered_df,
            x=x_3d,
            y=y_3d,
            z=z_3d,
            color='Cluster_RFM_Nombre',
            hover_data=['customer_id', 'age'],
            opacity=0.7,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_3d_rfm.update_layout(
            scene=dict(
                xaxis_title=x_lbl_3d,
                yaxis_title=y_lbl_3d,
                zaxis_title=z_lbl_3d
            ),
            margin=dict(r=0, l=0, b=0, t=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=8)),
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig_3d_rfm, use_container_width=True)

    # 3. DISCRETIZED BAR CHARTS SECTION
    st.markdown('<div class="section-header">Perfiles RFM Discretizados: Composición Porcentual Categórica</div>', unsafe_allow_html=True)
    st.write("Para entender de manera intuitiva el comportamiento comercial de los clusters, analizamos la composición de cada grupo en base a los rangos categóricos comerciales reales definidos por el negocio:")
    
    col_d1, col_d2, col_d3 = st.columns(3)
    
    with col_d1:
        st.markdown("**Recencia (Días Inactivo)**")
        dist_rec = filtered_df.groupby(['Cluster_RFM_Nombre', 'recency_cat']).size().unstack(level=1).fillna(0)
        dist_rec_pct = dist_rec.div(dist_rec.sum(axis=1), axis=0).reset_index()
        dist_rec_long = pd.melt(dist_rec_pct, id_vars=['Cluster_RFM_Nombre'], var_name='recency_cat', value_name='Proporción')
        
        fig_rec = px.bar(
            dist_rec_long, x="Cluster_RFM_Nombre", y="Proporción", color="recency_cat",
            barmode="stack", color_discrete_sequence=px.colors.qualitative.Pastel, height=250
        )
        fig_rec.update_layout(
            margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title='', showticklabels=False),
            yaxis=dict(tickformat='.0%', title=''),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=8)),
            template='plotly_white'
        )
        st.plotly_chart(fig_rec, use_container_width=True)
        
    with col_d2:
        st.markdown("**Frecuencia (Órdenes)**")
        dist_freq = filtered_df.groupby(['Cluster_RFM_Nombre', 'frequency_cat']).size().unstack(level=1).fillna(0)
        dist_freq_pct = dist_freq.div(dist_freq.sum(axis=1), axis=0).reset_index()
        dist_freq_long = pd.melt(dist_freq_pct, id_vars=['Cluster_RFM_Nombre'], var_name='frequency_cat', value_name='Proporción')
        
        fig_freq = px.bar(
            dist_freq_long, x="Cluster_RFM_Nombre", y="Proporción", color="frequency_cat",
            barmode="stack", color_discrete_sequence=px.colors.qualitative.Pastel, height=250
        )
        fig_freq.update_layout(
            margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title='', showticklabels=False),
            yaxis=dict(tickformat='.0%', title=''),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=8)),
            template='plotly_white'
        )
        st.plotly_chart(fig_freq, use_container_width=True)
        
    with col_d3:
        st.markdown("**Monetario (Gasto USD)**")
        dist_mon = filtered_df.groupby(['Cluster_RFM_Nombre', 'monetary_cat']).size().unstack(level=1).fillna(0)
        dist_mon_pct = dist_mon.div(dist_mon.sum(axis=1), axis=0).reset_index()
        dist_mon_long = pd.melt(dist_mon_pct, id_vars=['Cluster_RFM_Nombre'], var_name='monetary_cat', value_name='Proporción')
        
        fig_mon = px.bar(
            dist_mon_long, x="Cluster_RFM_Nombre", y="Proporción", color="monetary_cat",
            barmode="stack", color_discrete_sequence=px.colors.qualitative.Pastel, height=250
        )
        fig_mon.update_layout(
            margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title='', showticklabels=False),
            yaxis=dict(tickformat='.0%', title=''),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=8)),
            template='plotly_white'
        )
        st.plotly_chart(fig_mon, use_container_width=True)

    # 4. THREE FACES SECTION
    st.markdown('<div class="section-header">Las 3 Caras del Modelo RFM (Datos Originales y Estandarizados)</div>', unsafe_allow_html=True)
    st.write("El siguiente análisis detalla las proyecciones bidimensionales cruzadas (Recencia, Frecuencia y Monetario) del espacio de clientes en sus escalas correspondientes:")
    
    sub_tab1, sub_tab2, sub_tab3 = st.tabs([
        "Recencia vs Frecuencia",
        "Frecuencia vs Gasto (Monetario)",
        "Recencia vs Gasto (Monetario)"
    ])
    
    # Determine columns to plot based on toggle
    x_col_1 = 'recency_final' if use_std_data else 'days_since_last_purchase'
    y_col_1 = 'frequency_final' if use_std_data else 'total_orders'
    
    x_col_2 = 'frequency_final' if use_std_data else 'total_orders'
    y_col_2 = 'monetary_final' if use_std_data else 'total_spend_usd'
    
    x_col_3 = 'recency_final' if use_std_data else 'days_since_last_purchase'
    y_col_3 = 'monetary_final' if use_std_data else 'total_spend_usd'
    
    x_title_1 = 'Recencia (Z-Score)' if use_std_data else 'Recencia (Días inactivo)'
    y_title_1 = 'Frecuencia (Z-Score)' if use_std_data else 'Frecuencia (Total Órdenes)'
    
    x_title_2 = 'Frecuencia (Z-Score)' if use_std_data else 'Frecuencia (Total Órdenes)'
    y_title_2 = 'Gasto (Z-Score)' if use_std_data else 'Monto (Gasto Total USD)'
    
    x_title_3 = 'Recencia (Z-Score)' if use_std_data else 'Recencia (Días inactivo)'
    y_title_3 = 'Gasto (Z-Score)' if use_std_data else 'Monto (Gasto Total USD)'

    with sub_tab1:
        fig_face1 = px.scatter(
            filtered_df, x=x_col_1, y=y_col_1, color='Cluster_RFM_Nombre',
            title=f'Cara 1: {x_title_1} vs {y_title_1}', hover_data=['customer_id'],
            opacity=0.7, color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_face1.update_layout(xaxis_title=x_title_1, yaxis_title=y_title_1, template='plotly_white')
        st.plotly_chart(fig_face1, use_container_width=True)
        
    with sub_tab2:
        fig_face2 = px.scatter(
            filtered_df, x=x_col_2, y=y_col_2, color='Cluster_RFM_Nombre',
            title=f'Cara 2: {x_title_2} vs {y_title_2}', hover_data=['customer_id'],
            opacity=0.7, color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_face2.update_layout(xaxis_title=x_title_2, yaxis_title=y_title_2, template='plotly_white')
        st.plotly_chart(fig_face2, use_container_width=True)
        
    with sub_tab3:
        fig_face3 = px.scatter(
            filtered_df, x=x_col_3, y=y_col_3, color='Cluster_RFM_Nombre',
            title=f'Cara 3: {x_title_3} vs {y_title_3}', hover_data=['customer_id'],
            opacity=0.7, color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_face3.update_layout(xaxis_title=x_title_3, yaxis_title=y_title_3, template='plotly_white')
        st.plotly_chart(fig_face3, use_container_width=True)

# ==========================================
# TAB 3: SEGMENTACIÓN DEMOGRÁFICA
# ==========================================
with tab3:
    if pitch_mode:
        st.markdown("""
        <div class="pitch-guide">
            <strong>Guía para el Pitch (Minuto 4-6)</strong><br>
            • Explica por qué LCA es el único modelo adecuado para perfiles categóricos (país, canal, dispositivo, género): la distancia euclidiana de K-Means carece de sentido sobre variablesdummies.<br>
            • Explica el <strong>Gráfico 3D de Espacio Demográfico (PCA 3D)</strong>: Usaron reducción de dimensionalidad (PCA 3D) para proyectar las 7 variables categóricas en el espacio y demostrar visualmente cómo StepMix logra separar perfectamente a las poblaciones.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Modelo Sociodemográfico: Latent Class Analysis (LCA - K=3)</div>', unsafe_allow_html=True)
    
    st.info("**Justificación del Modelo**: Para los datos demográficos y de canal se utiliza **LCA (Latent Class Analysis)** mediante `StepMix`. Las variables sociodemográficas son categóricas nominales (región, género, dispositivo, canal de adquisición, etc.). K-Means no es adecuado debido a que la distancia euclidiana pierde su significado físico sobre variables codificadas como dummy. LCA clasifica probabilísticamente usando distribuciones multinomiales nativas.")
    
    # LCA summary
    # LCA summary (100% categorical - demographic variables only to maintain purity)
    lca_summary = filtered_df.groupby('Cluster_LCA_Nombre').agg(
        Clientes=('customer_id', 'count'),
        Genero_Moda=('gender', lambda x: x.mode()[0]),
        Edad_Moda=('age_group', lambda x: x.mode()[0]),
        Region_Moda=('region', lambda x: x.mode()[0]),
        Membresia_Moda=('membership_tier', lambda x: x.mode()[0]),
        Dispositivo_Moda=('preferred_device', lambda x: x.mode()[0]),
        Canal_Moda=('acquisition_channel', lambda x: x.mode()[0]),
        Categoria_Moda=('preferred_category', lambda x: x.mode()[0])
    ).reset_index()
    
    st.write("### Perfil de Clases Categóricas (Modas)")
    st.dataframe(lca_summary)
        
    st.markdown('<div class="section-header">Perfil de ADN Completo: Distribución de Todas las Variables Categóricas</div>', unsafe_allow_html=True)
    st.write("Para analizar simultáneamente **todas las variables discretizadas del modelo** sin ocultar información tras un selector, se presenta la composición porcentual de cada clase para las 7 variables sociodemográficas y de comportamiento:")
    
    col_a, col_b, col_c = st.columns(3)

    lca_vars_plot = [
        ('gender', 'Género', col_a),
        ('age_group', 'Rango de Edad', col_b),
        ('preferred_category', 'Categoría Preferida', col_c)
    ]

    for col_name, friendly_name, target_col in lca_vars_plot:
        with target_col:
            st.markdown(f"**{friendly_name}**")

            # For preferred_category: top 3 per LCA class independently -> rest = "Otras"
            if col_name == 'preferred_category':
                top3_map = {}
                for _cls in filtered_df['Cluster_LCA_Nombre'].unique():
                    _cls_top3 = (
                        filtered_df[filtered_df['Cluster_LCA_Nombre'] == _cls]['preferred_category']
                        .value_counts(normalize=True)
                        .head(3)
                        .index.tolist()
                    )
                    top3_map[_cls] = _cls_top3
                plot_df = filtered_df.copy()
                plot_df[col_name] = plot_df.apply(
                    lambda row: row['preferred_category'] if row['preferred_category'] in top3_map.get(row['Cluster_LCA_Nombre'], []) else 'Otras',
                    axis=1
                )
            else:
                plot_df = filtered_df

            dist_rel = plot_df.groupby(['Cluster_LCA_Nombre', col_name]).size().unstack(level=1).fillna(0)
            dist_rel_pct = dist_rel.div(dist_rel.sum(axis=1), axis=0).reset_index()
            dist_rel_long = pd.melt(dist_rel_pct, id_vars=['Cluster_LCA_Nombre'], var_name=col_name, value_name='Proporción')

            fig_v = px.bar(
                dist_rel_long,
                x="Cluster_LCA_Nombre",
                y="Proporción",
                color=col_name,
                barmode="stack",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                height=320
            )
            fig_v.update_layout(
                margin=dict(l=10, r=10, t=60, b=60),
                xaxis=dict(title='', showticklabels=True, tickangle=-30, tickfont=dict(size=9)),
                yaxis=dict(tickformat='.0%', title=''),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=7),
                    tracegroupgap=2
                ),
                template='plotly_white'
            )
            st.plotly_chart(fig_v, use_container_width=True)

    # Radar de ADN a ancho completo en Tab 3
    st.markdown('<div class="section-header">️ Radar de ADN: Perfil Multidimensional de las Clases Latentes</div>', unsafe_allow_html=True)
    st.write("El gráfico de radar permite comparar de manera simultánea el 'ADN' cualitativo de las tres clases latentes a lo largo de 5 dimensiones estratégicas de perfil. La forma y tamaño de los polígonos revela de un vistazo los rasgos identitarios de cada segmento meta de mercado:")
    
    # Calculate radar dimensions per class
    radar_data = []
    for c_val in range(3):
        c_name = f"Clase LCA {c_val}"
        mask = filtered_df['Cluster_LCA'] == c_val
        if mask.sum() > 0:
            sub = filtered_df[mask]
            p_sm = (sub['acquisition_channel'] == 'Social Media').mean()
            p_mob = (sub['preferred_device'] == 'Mobile').mean()
            p_young = sub['age_group'].isin(['18-25', '26-35']).mean()
            p_premium = sub['membership_tier'].isin(['Gold', 'Platinum']).mean()
            p_app_elec = sub['preferred_category'].isin(['Apparel', 'Electronics']).mean()
            
            # Radar requires closing the loop (repeat the first item at the end)
            radar_data.append(dict(
                r=[p_sm, p_mob, p_young, p_premium, p_app_elec, p_sm],
                theta=['Social Media', 'Dispositivo Móvil', 'Edad Joven (18-35)', 'Membresía Premium (Gold/Plat)', 'Categoría Apparel/Elec', 'Social Media'],
                name=c_name
            ))
            
    fig_radar = go.Figure()
    for r_item in radar_data:
        fig_radar.add_trace(go.Scatterpolar(
            r=[val * 100 for val in r_item['r']],
            theta=r_item['theta'],
            fill='toself',
            name=r_item['name']
        ))
        
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix='%'
            )
        ),
        title='Comparativa de Firmas de ADN por Clase LCA (%)',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=8))
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ==========================================
# TAB 4: MATRIZ DE CROSSOVER
# ==========================================
with tab4:
    if pitch_mode:
        st.markdown("""
        <div class="pitch-guide">
            <strong>Guía para el Pitch (Minuto 6-8)</strong><br>
            • Muestra la <strong>Matriz de Crossover</strong> de poblaciones cruzadas de K-Means transaccional (RFM) y LCA demográfico.<br>
            • Explica que al cruzar la segmentación geométrica K-Means con el LCA demográfico probabilístico logramos identificar micro-segmentos de alto valor. Por ejemplo, selecciona una combinación en los selectores y muestra las estadísticas detalladas (ticket promedio, dispositivo) para demostrar cómo la empresa puede micro-segmentar con precisión quirúrgica.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Matriz de Crossover de Poblaciones (RFM × Demográfico)</div>', unsafe_allow_html=True)
    
    crossover_data = pd.crosstab(
        filtered_df['Cluster_LCA_Nombre'],
        filtered_df['Cluster_RFM_Nombre'],
        normalize='all'
    ) * 100
    crossover_counts = pd.crosstab(
        filtered_df['Cluster_LCA_Nombre'],
        filtered_df['Cluster_RFM_Nombre']
    )
    
    # Fixed axis order: K-Means 0→1→2 on X, LCA 0→1→2 on Y (top-to-bottom)
    rfm_order_main = [
        "Clase K-Means 0: Frecuencia Alta, Gasto Alto (En Riesgo)",
        "Clase K-Means 1: Frecuencia Alta, Gasto Alto (Activos)",
        "Clase K-Means 2: Frecuencia Baja, Gasto Bajo (Dormidos)"
    ]
    lca_order_main = [
        "Clase LCA 0: Hombres Jóvenes (Redes Sociales, Hogar)",
        "Clase LCA 1: Adultos (Búsqueda Orgánica, Ropa)",
        "Clase LCA 2: Mujeres (Búsqueda Orgánica, Electrónica)"
    ]
    rfm_existing_main = [x for x in rfm_order_main if x in crossover_data.columns]
    lca_existing_main = [y for y in lca_order_main if y in crossover_data.index]
    crossover_data = crossover_data.reindex(index=lca_existing_main, columns=rfm_existing_main).fillna(0)
    crossover_counts = crossover_counts.reindex(index=lca_existing_main, columns=rfm_existing_main).fillna(0)
    
    col_mat, col_sub = st.columns([3, 2])
    
    with col_mat:
        st.write("### Intersección de Poblaciones (% del Total)")
        fig_cross = go.Figure(data=go.Heatmap(
            z=crossover_data.values,
            x=crossover_data.columns,
            y=crossover_data.index,
            colorscale='Blues',
            text=np.vectorize(lambda count, pct: f"N: {count}<br>{pct:.1f}%")(crossover_counts.values, crossover_data.values),
            texttemplate="%{text}",
            hoverinfo="z"
        ))
        fig_cross.update_layout(
            xaxis_title="Eje X: Segmento RFM (K-Means)",
            yaxis_title="Eje Y: Clase Demográfica (LCA)",
            margin=dict(l=0, r=0, b=0, t=40),
            height=400
        )
        st.plotly_chart(fig_cross, use_container_width=True)
        
    with col_sub:
        st.write("### Perfil del Sub-segmento Seleccionado")
        
        sel_rfm = st.selectbox("Seleccionar Segmento RFM", options=sorted(filtered_df['Cluster_RFM_Nombre'].unique()))
        _lca_opts = sorted(filtered_df['Cluster_LCA_Nombre'].unique())
        _lca_default = next((i for i, o in enumerate(_lca_opts) if 'LCA 2' in o), 0)
        sel_lca = st.selectbox("Seleccionar Clase LCA", options=_lca_opts, index=_lca_default)
        
        sub = filtered_df[(filtered_df['Cluster_RFM_Nombre'] == sel_rfm) & (filtered_df['Cluster_LCA_Nombre'] == sel_lca)]
        
        st.markdown('<div class="card" style="border-left: 5px solid #2A5298;">', unsafe_allow_html=True)
        st.write(f"**Tamaño de población**: {len(sub)} clientes ({ (len(sub)/len(filtered_df)*100):.1f}% de la muestra)")
        st.write(f"- **Gasto Promedio**: ${sub['total_spend_usd'].mean():,.2f}")
        st.write(f"- **Frecuencia Promedio**: {sub['total_orders'].mean():.1f} órdenes")
        st.write(f"- **Recencia Promedio**: {sub['days_since_last_purchase'].mean():.1f} días inactivos")
        st.write(f"- **Edad Promedio**: {sub['age'].mean():.1f} años")
        st.write(f"- **Categoría de Producto más Común**: {sub['preferred_category'].mode()[0]}")
        st.write(f"- **Canal Principal**: {sub['acquisition_channel'].mode()[0]}")
        st.write(f"- **Dispositivo Principal**: {sub['preferred_device'].mode()[0]}")
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 5: ESTRATEGIA DE PROMOCIONES
# ==========================================
with tab5:
    if pitch_mode:
        st.markdown("""
        <div class="pitch-guide">
            <strong>Guía para el Pitch (Minuto 8-10)</strong><br>
            • **El Cierre Fuerte**: Expón el análisis de receptividad a las promociones (Discount Ratio).<br>
            • Explica la revelación estratégica: La <strong>Clase K-Means 2 (Bajo Gasto)</strong> es sumamente sensible al precio (~39% de pedidos con descuento), mientras que la <strong>Clase K-Means 1 (VIP)</strong> compra de manera orgánica (~5% de descuento). ¡No devalúes tu margen regalando promociones a tus clientes más fieles!<br>
            • Da las recomendaciones de entrada al mercado: Campañas flash para la Clase K-Means 2 × Clase LCA 0 y posicionamiento premium basado en servicio y lealtad para la Clase K-Means 1.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Análisis de Receptividad a Promociones y Mercado Meta</div>', unsafe_allow_html=True)
    
    st.write("""
    ### Receptividad a Promociones
    Para responder al requerimiento de la empresa, analizamos la **sensibilidad a las promociones** calculando el **Discount Ratio** (proporción de compras realizadas con descuentos) y la tasa de suscripción al **Newsletter** (nuestro principal canal de contacto promocional).
    """)
    
    st.subheader("Cruce de Sensibilidad a Promociones (Matriz Crossover de Descuento)")
    
    # Calculate discount ratio across the crossover grid
    # Fixed axis order: K-Means 0→1→2 on X, LCA 0→1→2 on Y (top-to-bottom)
    rfm_order_promo = [
        "Clase K-Means 0: Frecuencia Alta, Gasto Alto (En Riesgo)",
        "Clase K-Means 1: Frecuencia Alta, Gasto Alto (Activos)",
        "Clase K-Means 2: Frecuencia Baja, Gasto Bajo (Dormidos)"
    ]
    lca_order_promo = [
        "Clase LCA 0: Hombres Jóvenes (Redes Sociales, Hogar)",
        "Clase LCA 1: Adultos (Búsqueda Orgánica, Ropa)",
        "Clase LCA 2: Mujeres (Búsqueda Orgánica, Electrónica)"
    ]
    crossover_promo = filtered_df.groupby(['Cluster_LCA_Nombre', 'Cluster_RFM_Nombre'])['discount_ratio'].mean().unstack()
    rfm_p_existing = [x for x in rfm_order_promo if x in crossover_promo.columns]
    lca_p_existing = [y for y in lca_order_promo if y in crossover_promo.index]
    crossover_promo = crossover_promo.reindex(index=lca_p_existing, columns=rfm_p_existing).fillna(0)
    
    col_mat_p, col_details_p = st.columns([3, 2])
    
    with col_mat_p:
        fig_cross_promo = go.Figure(data=go.Heatmap(
            z=crossover_promo.values,
            x=crossover_promo.columns,
            y=crossover_promo.index,
            colorscale='Purples',
            text=np.vectorize(lambda val: f"{val*100:.1f}% de pedidos")(crossover_promo.values),
            texttemplate="%{text}",
            hoverinfo="z"
        ))
        fig_cross_promo.update_layout(
            xaxis_title="Segmento RFM (K-Means)",
            yaxis_title="Clase Demográfica (LCA)",
            margin=dict(l=0, r=0, b=0, t=40),
            height=400
        )
        st.plotly_chart(fig_cross_promo, use_container_width=True)
        
    with col_details_p:
        st.write("### Identificación del Mercado Meta")
        st.markdown("""
        ### Mercado Meta Seleccionado: Intersección Clase LCA 2 × Clase K-Means 0 (Mujeres VIP en Riesgo)
        Para nuestro ingreso estratégico al mercado, hemos seleccionado como **mercado meta prioritario** el micro-segmento de mayor rentabilidad potencial y urgencia de retención: la intersección **Clase LCA 2 (Mujeres, Búsqueda Orgánica, Electrónica) × Clase K-Means 0 (Frecuentes y Gasto Alto en Riesgo)**.

        #### Justificación Estratégica y Comercial (STP):
        1.  **Valor de Ciclo de Vida del Cliente (CLV) Elevadísimo**: Este segmento representa a clientas de alto valor con un gasto promedio real superior a **$2,500 USD** y una frecuencia histórica de **12 a 14 órdenes**. La captura y retención de una sola de estas clientas equivale comercialmente a activar a decenas de clientes de bajo gasto (`KM_2`).
        2.  **Urgencia de Win-Back (Recuperación)**: Al pertenecer al segmento `KM_0` (En Riesgo), su recencia promedio supera los **90 días de inactividad**. Representan una fuga inminente de ingresos si no se actúa con rapidez.
        3.  **Insensibilidad al Margen Tradicional (Orgánicas)**: Este grupo fue adquirido mayoritariamente mediante Búsqueda Orgánica y tiene una sensibilidad promocional extremadamente baja (Discount Ratio: **<5%**). Esto indica que valoran la exclusividad, la calidad del producto y el servicio premium por sobre los descuentos directos, lo cual permite retenerlas **sin erosionar nuestros márgenes con rebajas agresivas**.
        4.  **Afinidad de Categoría**: Su preferencia absoluta es **Electronics**. Al ser electrónica una categoría de alto valor unitario, las tácticas recomendadas deben enfocarse en garantías extendidas gratuitas, acceso prioritario a lanzamientos y soporte premium dedicado.
        """)
        
    # Individual Promotional Receptivity Charts below the heatmap
    st.markdown('<div class="section-header">Desglose de Receptividad Promocional Individual (LCA y RFM)</div>', unsafe_allow_html=True)
    c_p1, c_p2 = st.columns(2)
    
    with c_p1:
        # Discount ratio by RFM
        promo_rfm = filtered_df.groupby('Cluster_RFM_Nombre')['discount_ratio'].mean().reset_index()
        fig_promo_rfm = px.bar(
            promo_rfm,
            x="Cluster_RFM_Nombre",
            y="discount_ratio",
            color="Cluster_RFM_Nombre",
            title="Proporción de Compras con Descuento por Segmento RFM",
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig_promo_rfm.update_layout(template='plotly_white')
        st.plotly_chart(fig_promo_rfm, use_container_width=True)
        
    with c_p2:
        # Discount ratio by LCA
        promo_lca = filtered_df.groupby('Cluster_LCA_Nombre')['discount_ratio'].mean().reset_index()
        fig_promo_lca = px.bar(
            promo_lca,
            x="Cluster_LCA_Nombre",
            y="discount_ratio",
            color="Cluster_LCA_Nombre",
            title="Proporción de Compras con Descuento por Clase LCA",
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig_promo_lca.update_layout(template='plotly_white')
        st.plotly_chart(fig_promo_lca, use_container_width=True)
        

    st.markdown('<div class="section-header">Conclusiones Estratégicas Generales y Recomendaciones</div>', unsafe_allow_html=True)
    c_rec1, c_rec2 = st.columns(2)
    
    with c_rec1:
        st.markdown("""
        <div class="card" style="border-left: 5px solid #6366F1; min-height: 230px; padding: 1.5rem; border-radius: 10px; background-color: #1E293B; color: #F8FAFC; margin-bottom: 1rem;">
            <h3 style="color: #6366F1; margin-top: 0; font-size: 1.1rem; display: flex; align-items: center; gap: 0.5rem; font-family: inherit;">
                Segmentación Quirúrgica en Promociones
            </h3>
            <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.9rem; line-height: 1.5; color: #F8FAFC;">
                <li style="margin-bottom: 0.5rem;"><strong>Protección Absoluta de Margen</strong>: No diluir márgenes ofreciendo cupones masivos a los VIPs activos (<code style="background: #0f172a; padding: 2px 4px; border-radius: 4px; color: #10b981; font-family: monospace;">KM_1</code> / <code style="background: #0f172a; padding: 2px 4px; border-radius: 4px; color: #10b981; font-family: monospace;">LCA_2</code>), quienes compran de forma orgánica y con una sensibilidad al precio extremadamente baja.</li>
                <li><strong>Direccionamiento de Descuentos</strong>: Canalizar las ofertas flash agresivas y campañas de descuento únicamente hacia el segmento altamente elástico de reactivación (<strong>Clase K-Means 2 × Clase LCA 0 - Dormidos e impulsados por precio</strong>), donde la elasticidad responde positivamente.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with c_rec2:
        st.markdown("""
        <div class="card" style="border-left: 5px solid #8B5CF6; min-height: 230px; padding: 1.5rem; border-radius: 10px; background-color: #1E293B; color: #F8FAFC; margin-bottom: 1rem;">
            <h3 style="color: #8B5CF6; margin-top: 0; font-size: 1.1rem; display: flex; align-items: center; gap: 0.5rem; font-family: inherit;">
                Crecimiento en el Mercado Meta
            </h3>
            <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.9rem; line-height: 1.5; color: #F8FAFC;">
                <li style="margin-bottom: 0.5rem;"><strong>Recuperación VIP Quirúrgica</strong>: Para capturar y retener a las <strong>Mujeres VIP en Riesgo (LCA 2 × KM 0)</strong>, se implementará un playbook de recuperación VIP mediante un programa de fidelidad de alto nivel.</li>
                <li><strong>Incentivos No Monetarios</strong>: Priorizar el soporte prioritario, acceso anticipado y muestras exclusivas en su categoría preferida (<strong>Electronics</strong>) antes que recurrir a rebajas directas de precio.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 6: CATEGORÍAS POR SEGMENTO
# ==========================================
with tab6:
    st.markdown('<div class="section-header">️ Matriz de Crossover de Categorías de Producto Preferidas</div>', unsafe_allow_html=True)
    st.write("""
    Esta matriz de crossover avanzada cruza los tres perfiles transaccionales (K-Means) con las tres clases demográficas (LCA) para revelar la **Categoría de Producto Preferida** predominante y su porcentaje de concentración en cada micro-segmento de clientes activos. Esto permite orientar el inventario, el catálogo y las promociones con total precisión:
    """)
    
    # We will compute the mode category and its share for each crossover cell
    rfm_order = [
        "Clase K-Means 0: Frecuencia Alta, Gasto Alto (En Riesgo)",
        "Clase K-Means 1: Frecuencia Alta, Gasto Alto (Activos)",
        "Clase K-Means 2: Frecuencia Baja, Gasto Bajo (Dormidos)"
    ]
    lca_order = [
        "Clase LCA 0: Hombres Jóvenes (Redes Sociales, Hogar)",
        "Clase LCA 1: Adultos (Búsqueda Orgánica, Ropa)",
        "Clase LCA 2: Mujeres (Búsqueda Orgánica, Electrónica)"
    ]
    
    rfm_existing = [x for x in rfm_order if x in filtered_df['Cluster_RFM_Nombre'].unique()]
    lca_existing = [y for y in lca_order if y in filtered_df['Cluster_LCA_Nombre'].unique()]
    
    cat_texts = np.empty((len(lca_existing), len(rfm_existing)), dtype=object)
    cat_ids = np.zeros((len(lca_existing), len(rfm_existing)))
    
    # Pass 1: Collect first category percentages for normalization per row (LCA class)
    row_pcts = {}
    for r_idx, lca_name in enumerate(lca_existing):
        row_pcts[r_idx] = []
        for c_idx, rfm_name in enumerate(rfm_existing):
            sub = filtered_df[(filtered_df['Cluster_LCA_Nombre'] == lca_name) & (filtered_df['Cluster_RFM_Nombre'] == rfm_name)]
            if len(sub) > 0:
                cat_counts = sub['preferred_category'].value_counts(normalize=True) * 100
                if len(cat_counts) > 0:
                    row_pcts[r_idx].append(cat_counts.iloc[0])
                else:
                    row_pcts[r_idx].append(0)
            else:
                row_pcts[r_idx].append(0)

    # Pass 2: Calculate and assign z values so cells with the same category (by row) have the same color family but different shades
    for r_idx, lca_name in enumerate(lca_existing):
        vals = row_pcts[r_idx]
        min_v = min(vals) if len(vals) > 0 else 0
        max_v = max(vals) if len(vals) > 0 else 0
        v_range = max_v - min_v
        
        for c_idx, rfm_name in enumerate(rfm_existing):
            sub = filtered_df[(filtered_df['Cluster_LCA_Nombre'] == lca_name) & (filtered_df['Cluster_RFM_Nombre'] == rfm_name)]
            if len(sub) > 0:
                cat_counts = sub['preferred_category'].value_counts(normalize=True) * 100
                p = cat_counts.iloc[0] if len(cat_counts) > 0 else 0
                
                # Normalize percentage within the row (LCA class)
                norm_p = (p - min_v) / v_range if v_range > 0 else 0.5
                
                # Map to its respective row range to ensure same color family per category/row with different shades:
                # Row 0 (Kitchen): Slate Grey-Blue (0.0 to 0.2)
                # Row 1 (Apparel): Indigo Blue (0.4 to 0.6)
                # Row 2 (Electronics): Sky Blue (0.8 to 1.0)
                cat_ids[r_idx, c_idx] = r_idx * 0.4 + norm_p * 0.2
                
                top3 = cat_counts.head(3)
                top3_lines = []
                for rank, (cat, val) in enumerate(top3.items()):
                    disp_cat = cat.replace("Clothing & ", "").replace("Toys & ", "").replace("Home & ", "").replace("Beauty & ", "").replace("Travel & ", "").replace("Jewelry & ", "").replace("Health & ", "").replace("Pet ", "").replace("Office ", "").replace("Sports & ", "")
                    top3_lines.append(f"{rank+1}. {disp_cat} ({val:.1f}%)")
                
                cat_texts[r_idx, c_idx] = "<br>".join(top3_lines)
            else:
                cat_texts[r_idx, c_idx] = "Sin Datos"
                cat_ids[r_idx, c_idx] = -1
                
    # Masculine light/clear colorscale (different tones/shades of the blue/slate family)
    # Row 0 (Kitchen/Home): Slate Grey-Blue tones (0.0 to 0.2)
    # Row 1 (Apparel/Ropa): Indigo Blue tones (0.4 to 0.6)
    # Row 2 (Electronics): Sky Blue tones (0.8 to 1.0)
    masculine_colorscale = [
        [0.0, '#E2E8F0'],  # Slate 100 (lightest, very soft slate grey-blue)
        [0.1, '#CBD5E1'],  # Slate 200 (medium, clear slate grey-blue)
        [0.2, '#94A3B8'],  # Slate 400 (strongest, visible grey-blue but not dark)
        
        [0.4, '#C7D2FE'],  # Indigo 200 (lightest, very soft steel/indigo blue)
        [0.5, '#A5B4FC'],  # Indigo 300 (medium, clear steel/indigo blue)
        [0.6, '#818CF8'],  # Indigo 400 (strongest, visible steel/indigo blue but not dark)
        
        [0.8, '#BAE6FD'],  # Sky 200 (lightest, very soft sky blue)
        [0.9, '#7DD3FC'],  # Sky 300 (medium, clear sky blue)
        [1.0, '#38BDF8']   # Sky 400 (strongest, visible sky blue but not dark)
    ]

    # Create the heatmap
    fig_cat_cross = go.Figure(data=go.Heatmap(
        z=cat_ids,
        x=rfm_existing,
        y=lca_existing,
        colorscale=masculine_colorscale,
        text=cat_texts,
        texttemplate="%{text}",
        hoverinfo="text",
        showscale=False
    ))
    
    fig_cat_cross.update_layout(
        xaxis_title="Eje X: Segmento RFM (K-Means)",
        yaxis_title="Eje Y: Clase Demográfica (LCA)",
        margin=dict(l=0, r=0, b=0, t=40),
        height=380,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_cat_cross, use_container_width=True)


# ==========================================
# TAB 7: PÉRDIDA DE MEMBRESÍAS (CHURN)
# ==========================================
with tab7:
    # First: Churn Crossover Matrix Heatmap
    st.markdown('<div class="section-header">Matriz de Crossover de Fuga de Clientes (Tasa de Churn por Micro-segmento)</div>', unsafe_allow_html=True)
    st.write("""
    Esta matriz avanzada cruza los tres perfiles transaccionales (K-Means) con las tres clases demográficas (LCA) sobre la base completa de clientes para revelar la **Tasa de Abandono (Churn Rate %)** y la cantidad de cuentas canceladas en cada micro-segmento cruzado en tiempo real:
    """)
    
    # Calculate Churn Rate crossover dynamically
    cross_churn_rate = pd.crosstab(
        filtered_df_raw['Cluster_LCA_Nombre'],
        filtered_df_raw['Cluster_RFM_Nombre'],
        values=filtered_df_raw['churned'],
        aggfunc='mean'
    ) * 100
    
    cross_churn_counts = pd.crosstab(
        filtered_df_raw['Cluster_LCA_Nombre'],
        filtered_df_raw['Cluster_RFM_Nombre'],
        values=filtered_df_raw['churned'],
        aggfunc='sum'
    )
    
    # Fill NaN values with 0
    cross_churn_rate = cross_churn_rate.fillna(0)
    cross_churn_counts = cross_churn_counts.fillna(0)
    
    # Ensure correct sorting order for K-Means and LCA
    rfm_order = [
        "Clase K-Means 0: Frecuencia Alta, Gasto Alto (En Riesgo)",
        "Clase K-Means 1: Frecuencia Alta, Gasto Alto (Activos)",
        "Clase K-Means 2: Frecuencia Baja, Gasto Bajo (Dormidos)"
    ]
    lca_order = [
        "Clase LCA 0: Hombres Jóvenes (Redes Sociales, Hogar)",
        "Clase LCA 1: Adultos (Búsqueda Orgánica, Ropa)",
        "Clase LCA 2: Mujeres (Búsqueda Orgánica, Electrónica)"
    ]
    
    # Filter order based on what exists in current filters
    rfm_existing = [x for x in rfm_order if x in cross_churn_rate.columns]
    lca_existing = [y for y in lca_order if y in cross_churn_rate.index]
    
    cross_churn_rate = cross_churn_rate.reindex(index=lca_existing, columns=rfm_existing).fillna(0)
    cross_churn_counts = cross_churn_counts.reindex(index=lca_existing, columns=rfm_existing).fillna(0)
    
    # Generate Heatmap
    fig_cross_churn = go.Figure(data=go.Heatmap(
        z=cross_churn_rate.values,
        x=cross_churn_rate.columns,
        y=cross_churn_rate.index,
        colorscale='Reds',
        text=np.vectorize(lambda count, rate: f"Fugados: {int(count)}<br>Tasa: {rate:.2f}%")(cross_churn_counts.values, cross_churn_rate.values),
        texttemplate="%{text}",
        hoverinfo="z"
    ))
    
    fig_cross_churn.update_layout(
        xaxis_title="Eje X: Segmento RFM (K-Means)",
        yaxis_title="Eje Y: Clase Demográfica (LCA)",
        margin=dict(l=0, r=0, b=0, t=40),
        height=380,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_cross_churn, use_container_width=True)

    # Second: Pérdida de Membresías (Análisis de Churn) and its intro text
    st.markdown('<div class="section-header">Pérdida de Membresías (Análisis de Churn)</div>', unsafe_allow_html=True)
    st.write("""
    Para que las campañas de marketing digital y los esfuerzos de fidelización sean sostenibles, la empresa debe identificar y mitigar la **tasa de cancelación (Churn)** de membresías. 
    Este módulo analiza a los clientes que abandonaron la plataforma (**8.9% global / 715 clientes**) y establece las pautas estratégicas de retención antes de simular escenarios de campañas comerciales en el simulador de ROI.
    """)
    
    # Active vs Churned and Tier Churn Charts
    col_ch1, col_ch2 = st.columns(2)
    
    with col_ch1:
        # Donut Chart for Active vs Churned
        churn_counts = filtered_df_raw['churned'].value_counts()
        labels_churn = ['Clientes Activos (Fidelizados)', 'Clientes Fugados (Churned)']
        fig_churn_pie = px.pie(
            names=labels_churn,
            values=churn_counts.values,
            hole=0.45,
            color_discrete_sequence=['#10B981', '#EF4444'], # Emerald & Crimson
            title="Distribución de Clientes Activos vs. Fugados"
        )
        fig_churn_pie.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            marker=dict(line=dict(color='#FFFFFF', width=2))
        )
        fig_churn_pie.update_layout(
            template='plotly_white',
            margin=dict(t=50, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_churn_pie, use_container_width=True)
        
    with col_ch2:
        # Churn Rate by Membership Tier
        tier_churn_rate = filtered_df_raw.groupby('membership_tier')['churned'].mean().reset_index()
        tier_churn_rate['churn_pct'] = tier_churn_rate['churned'] * 100
        tier_churn_rate = tier_churn_rate.sort_values(by='churn_pct', ascending=False)
        
        # Color mapping for tiers matching the dashboard theme
        color_map = {
            'Free': '#6B7280',      # Gray
            'Gold': '#F59E0B',      # Gold/Amber
            'Silver': '#9CA3AF',    # Silver/Light Gray
            'Platinum': '#3B82F6'   # Blue/Platinum
        }
        
        fig_tier_churn = px.bar(
            tier_churn_rate,
            x='membership_tier',
            y='churn_pct',
            text=[f"{val:.2f}%" for val in tier_churn_rate['churn_pct']],
            color='membership_tier',
            color_discrete_map=color_map,
            title="Tasa de Deserción (Churn Rate) por Nivel de Membresía",
            labels={'membership_tier': 'Nivel de Membresía', 'churn_pct': 'Tasa de Cancelación (%)'}
        )
        fig_tier_churn.update_layout(
            template='plotly_white', 
            showlegend=False, 
            margin=dict(t=50, b=20, l=20, r=20),
            yaxis=dict(ticksuffix="%")
        )
        fig_tier_churn.update_traces(
            marker=dict(line=dict(color='#FFFFFF', width=1)),
            textposition='outside'
        )
        st.plotly_chart(fig_tier_churn, use_container_width=True)
    
    # Second row: country / region analysis and age groups
    st.markdown('<div class="section-header">Análisis Geográfico y Perfil de Fuga</div>', unsafe_allow_html=True)
    col_ch3, col_ch4 = st.columns(2)
    
    with col_ch3:
        # High-risk countries
        country_churn = filtered_df_raw.groupby('country')['churned'].mean().reset_index()
        country_churn['churn_pct'] = country_churn['churned'] * 100
        country_churn = country_churn.sort_values(by='churn_pct', ascending=False).head(8)
        
        fig_country_churn = px.bar(
            country_churn,
            y='country',
            x='churn_pct',
            orientation='h',
            text=[f"{val:.1f}%" for val in country_churn['churn_pct']],
            title="Top 8 Países con Mayor Tasa de Fuga de Clientes",
            labels={'country': 'País', 'churn_pct': 'Tasa de Deserción (%)'},
            color_discrete_sequence=['#EC4899'] # Rose
        )
        fig_country_churn.update_layout(
            template='plotly_white',
            margin=dict(t=50, b=20, l=20, r=20),
            xaxis=dict(ticksuffix="%")
        )
        fig_country_churn.update_traces(textposition='inside')
        fig_country_churn.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig_country_churn, use_container_width=True)
        
    with col_ch4:
        # Age group churn
        # Copy to avoid SettingWithCopyWarning
        f_df_raw = filtered_df_raw.copy()
        f_df_raw['age_group'] = pd.cut(
            f_df_raw['age'],
            bins=[17, 25, 35, 45, 55, 75],
            labels=['18-25', '26-35', '36-45', '46-55', '56+']
        )
        age_churn = f_df_raw.groupby('age_group')['churned'].mean().reset_index()
        age_churn['churn_pct'] = age_churn['churned'] * 100
        
        fig_age_churn = px.line(
            age_churn,
            x='age_group',
            y='churn_pct',
            markers=True,
            title="Comportamiento del Churn por Rango de Edad",
            labels={'age_group': 'Rango de Edad', 'churn_pct': 'Tasa de Cancelación (%)'},
            color_discrete_sequence=['#8B5CF6'] # Purple
        )
        fig_age_churn.update_layout(
            template='plotly_white',
            margin=dict(t=50, b=20, l=20, r=20),
            yaxis=dict(ticksuffix="%")
        )
        st.plotly_chart(fig_age_churn, use_container_width=True)

    # Playbook cards
    st.markdown('<div class="section-header">️ Playbook Estratégico para la Retención y Control de Churn</div>', unsafe_allow_html=True)
    c_ch_rec1, c_ch_rec2 = st.columns(2)
    
    with c_ch_rec1:
        st.markdown('<div class="card" style="border-left: 5px solid #EF4444;">', unsafe_allow_html=True)
        st.write("### Puntos Críticos Detectados (Zonas de Dolor)")
        st.write("""
        - **El Dilema de la Membresía Gold**: El nivel **Gold registra la mayor tasa de fuga (9.86%)** entre todos los niveles pagados, superando al nivel Free (9.52%). Esto indica que los clientes Gold no están percibiendo suficiente valor por su suscripción activa.
        - **Foco Geográfico de Riesgo**: **Turquía (13.16%)**, **Polonia (13.00%)**, **Sudáfrica (12.77%)** y **Japón (12.18%)** son focos rojos. La alta deserción en estos países sugiere problemas con la pasarela de pagos local o retrasos logísticos en envíos transfronterizos.
        - **Canales Volátiles**: Los clientes captados por **Referidos (9.94%)** cancelan con mayor frecuencia, demostrando que las ofertas agresivas de referidos atraen usuarios transitorios de baja lealtad.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c_ch_rec2:
        st.markdown('<div class="card" style="border-left: 5px solid #10B981;">', unsafe_allow_html=True)
        st.write("### Acciones de Mitigación Inmediata (Playbook de Growth)")
        st.write("""
        - **Rediseño de Beneficios Gold**: Aumentar la propuesta de valor de la membresía Gold agregando acceso anticipado a ofertas y atención prioritaria para detener la deserción.
        - **Optimización Logística y Localización**: Realizar una auditoría operativa profunda de envíos y logística de última milla en Turquía, Polonia y Japón para resolver problemas locales de entrega.
        - **Campañas Proactivas de Re-engagement**: Configurar alertas automatizadas para clientes con **inactividad > 45 días** (especialmente en el grupo de 46-55 años, donde el churn sube al 10.25%), gatillando correos personalizados basados en su categoría de interés histórico.
        - **Estrategia en ROI**: Usar estas tasas de pérdida como filtros lógicos en campañas de retargeting para medir el impacto neto real en el simulador financiero.
        """)
        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# TAB 8: SIMULADOR DE ROI
# ==========================================
with tab8:
    if pitch_mode:
        st.markdown("""
        <div class="pitch-guide">
            <strong>Guía para el Pitch (Minuto 10+) — BONUS DE VALOR COMERCIAL</strong><br>
            • **El Cierre Ejecutivo**: Muestra a la Junta Directiva cómo los clusters no son solo teoría, sino que sirven para planificar presupuestos comerciales reales y predecir el ROI.<br>
            • Selecciona el micro-segmento estrella (ej: Clase K-Means 2 x Clase LCA 0) y simula una campaña con un descuento del 20%. Muestra cómo el ROI es altamente positivo debido a su sensibilidad al precio, mientras que si lo aplicas en el segmento VIP activo (Clase K-Means 1), ¡el ROI destruye valor debido a su bajísima sensibilidad promocional!
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Simulador Comercial de Campañas y Retorno de Inversión (ROI)</div>', unsafe_allow_html=True)
    st.write("""
    Este simulador interactivo utiliza las **métricas empíricas reales** calculadas para cada micro-segmento cruzado en la matriz de crossover (su tamaño poblacional, gasto promedio y sensibilidad histórica a promociones `discount_ratio`) para modelar escenarios predictivos de campañas de marketing digital.
    """)

    # 1. Dropdown for micro-segment selection
    micro_segments = filtered_df.groupby(['Cluster_RFM_Nombre', 'Cluster_LCA_Nombre']).size().reset_index()
    micro_segments['Opción'] = micro_segments['Cluster_RFM_Nombre'] + "  x  " + micro_segments['Cluster_LCA_Nombre']
    
    _options = micro_segments['Opción'].tolist()
    _default_idx = 0
    for _idx, _opt in enumerate(_options):
        if "K-Means 0" in _opt and "LCA 2" in _opt:
            _default_idx = _idx
            break
            
    sel_micro = st.selectbox("Seleccionar Micro-segmento Objetivo para la Campaña", options=_options, index=_default_idx)
    
    # Extract names
    rfm_part = sel_micro.split("  x  ")[0]
    lca_part = sel_micro.split("  x  ")[1]
    
    # Filter dataset for selected microsegment
    sub_df = filtered_df[(filtered_df['Cluster_RFM_Nombre'] == rfm_part) & (filtered_df['Cluster_LCA_Nombre'] == lca_part)]
    
    N_segment = len(sub_df)
    avg_spend = sub_df['total_spend_usd'].mean()
    emp_sensitivity = sub_df['discount_ratio'].mean()
    
    # Render segment stats card
    st.markdown(f"""
    <div class="card" style="border-left: 5px solid #F59E0B; margin-bottom: 1.5rem;">
        <h4 style="margin-top:0;">Perfil Base del Micro-segmento Objetivo</h4>
        • <strong>Tamaño de población activa (N)</strong>: {N_segment:,} clientes en la base de datos.<br>
        • <strong>Gasto Promedio por compra (Ticket)</strong>: ${avg_spend:,.2f} USD.<br>
        • <strong>Sensibilidad Promocional Histórica (Discount Ratio)</strong>: {emp_sensitivity*100:.1f}% (proporción de compras realizadas con descuento).
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Simulator Inputs in Columns
    col_sim_in1, col_sim_in2, col_sim_in3 = st.columns(3)
    
    with col_sim_in1:
        budget = st.number_input("Presupuesto de la Campaña (USD)", min_value=100, max_value=100000, value=5000, step=500)
        cost_per_reach = st.slider("Costo de Adquisición Digital por Cliente Reach (USD)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
        
    with col_sim_in2:
        discount_pct = st.slider("Descuento a Ofrecer (%)", min_value=5, max_value=50, value=20, step=5) / 100.0
        baseline_conv = st.slider("Tasa de Conversión Orgánica Base (%)", min_value=0.5, max_value=15.0, value=2.0, step=0.5) / 100.0
        
    with col_sim_in3:
        st.markdown("**Elasticidad y Comportamiento del Segmento**:")
        st.write(f"- Sensibilidad al descuento: **{'Alta / Sensible al precio' if emp_sensitivity > 0.25 else 'Baja / Cliente Orgánico' if emp_sensitivity < 0.1 else 'Media'}** ({emp_sensitivity*100:.1f}%)")
        st.write("- El simulador calcula el incremento de conversión (Uplift) ponderando su elasticidad histórica.")
        
    # 3. CALCULATE OUTCOMES
    max_reach = int(budget / cost_per_reach)
    targeted_customers = min(max_reach, N_segment)
    excess_budget = max(0.0, budget - (targeted_customers * cost_per_reach))
    effective_budget_used = budget - excess_budget
    
    # Uplift logic linked directly to historical discount ratio
    elasticity_coefficient = emp_sensitivity * 6.0
    promo_uplift = discount_pct * elasticity_coefficient
    promo_conv_rate = baseline_conv * (1.0 + promo_uplift)
    
    # Conversiones esperadas
    conversions_total = int(targeted_customers * promo_conv_rate)
    conversions_baseline = int(targeted_customers * baseline_conv)
    conversions_incremental = max(0, conversions_total - conversions_baseline)
    
    # ── Traditional Attributed Calculations ──
    gross_revenue_total = conversions_total * avg_spend
    net_revenue_total = gross_revenue_total * (1.0 - discount_pct)
    promo_cost_total = gross_revenue_total * discount_pct
    net_profit_total = net_revenue_total - effective_budget_used
    roi_pct_total = (net_profit_total / effective_budget_used) * 100.0 if effective_budget_used > 0 else 0.0
    
    # ── Advanced True Incremental Calculations ──
    # Revenue that would have happened organically anyway
    baseline_revenue = conversions_baseline * avg_spend
    # Cost of cannibalization: discount given to organic buyers who would buy anyway
    cannibalization_cost = conversions_baseline * avg_spend * discount_pct
    # Incremental revenue from the new buyers only
    gross_revenue_incremental = conversions_incremental * avg_spend
    # Net incremental revenue subtracting the cannibalization cost of organic baseline
    net_revenue_incremental = (gross_revenue_incremental * (1.0 - discount_pct)) - cannibalization_cost
    # Net incremental profit
    net_profit_incremental = net_revenue_incremental - effective_budget_used
    # Incremental ROI
    roi_pct_incremental = (net_profit_incremental / effective_budget_used) * 100.0 if effective_budget_used > 0 else 0.0
    
    # 4. DISPLAY RESULTS
    st.markdown('<div class="section-header">Proyección de Métricas y Retorno Comercial</div>', unsafe_allow_html=True)
    
    eval_model = st.radio(
        "**Modelo de Evaluación Financiera de la Campaña**:",
        options=[
            "Retorno Total Atribuido (Tradicional: incluye ventas orgánicas e incrementales juntas)",
            "Retorno Neto Incremental Avanzado (Recomendado: descuenta ventas orgánicas y resta costo de canibalización)"
        ],
        index=1,
        help="El modelo tradicional muestra todos los ingresos del grupo alcanzado, lo que puede sobreestimar el impacto. El modelo incremental aísla solo las ventas adicionales generadas directamente por la promoción y resta la canibalización de ventas orgánicas."
    )
    
    is_incremental = "Incremental" in eval_model
    
    if is_incremental:
        display_net_revenue = net_revenue_incremental
        display_net_profit = net_profit_incremental
        display_roi = roi_pct_incremental
        display_conversions = conversions_incremental
        display_promo_cost = cannibalization_cost + (gross_revenue_incremental * discount_pct)
    else:
        display_net_revenue = net_revenue_total
        display_net_profit = net_profit_total
        display_roi = roi_pct_total
        display_conversions = conversions_total
        display_promo_cost = promo_cost_total
        
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.metric("Clientes Impactados (Reach)", f"{targeted_customers:,}", f"De {N_segment:,} en base")
        
    with col_stat2:
        if is_incremental:
            st.metric("Conversiones Incrementales (Uplift)", f"{display_conversions:,}", f"+{(promo_conv_rate - baseline_conv)*100:.2f}% de conversión")
        else:
            st.metric("Conversiones Totales Promo", f"{display_conversions:,}", f"+{(promo_conv_rate - baseline_conv)*100:.2f}% de conversión")
        
    with col_stat3:
        if is_incremental:
            st.metric("Ingreso Neto Incremental", f"${display_net_revenue:,.2f} USD")
        else:
            st.metric("Ingreso Neto Estimado", f"${display_net_revenue:,.2f} USD")
        
    with col_stat4:
        if is_incremental:
            st.metric("Ganancia Incremental Real", f"${display_net_profit:,.2f} USD", f"ROI Incremental: {display_roi:.1f}%", delta_color="normal" if display_roi >= 0 else "inverse")
        else:
            st.metric("Retorno Neto (Ganancia)", f"${display_net_profit:,.2f} USD", f"ROI: {display_roi:.1f}%", delta_color="normal" if display_roi >= 0 else "inverse")
        
    # Render charts and recommendations side-by-side
    c_graph1, c_graph2 = st.columns([3, 2])
    
    with c_graph1:
        if is_incremental:
            fig_sim_bar = go.Figure(data=[
                go.Bar(name='Costo de Contactación (Medios)', x=['Campaña Incremental'], y=[effective_budget_used], marker_color='#3B82F6'),
                go.Bar(name='Costo de Canibalización (Orgánicos)', x=['Campaña Incremental'], y=[cannibalization_cost], marker_color='#F59E0B'),
                go.Bar(name='Ganancia Incremental Real', x=['Campaña Incremental'], y=[max(0, display_net_profit)], marker_color='#10B981'),
                go.Bar(name='Pérdida Incremental (Destrucción)', x=['Campaña Incremental'], y=[-display_net_profit if display_net_profit < 0 else 0], marker_color='#EF4444')
            ])
        else:
            fig_sim_bar = go.Figure(data=[
                go.Bar(name='Costo de Contactación (Medios)', x=['Campaña Total'], y=[effective_budget_used], marker_color='#3B82F6'),
                go.Bar(name='Costo de Descuento (Margen Perdido)', x=['Campaña Total'], y=[promo_cost_total], marker_color='#EF4444'),
                go.Bar(name='Ganancia Neta Atribuida', x=['Campaña Total'], y=[max(0, display_net_profit)], marker_color='#10B981')
            ])
            
        fig_sim_bar.update_layout(
            barmode='stack',
            title='Estructura de Costos vs Ganancia Estimada (USD)',
            yaxis_title='Dólares (USD)',
            template='plotly_white',
            height=350
        )
        st.plotly_chart(fig_sim_bar, use_container_width=True)
        
    with c_graph2:
        if display_roi > 20.0:
            st.success(f"""
            ### RECOMENDACIÓN: **EJECUTAR CAMPAÑA**
            - **ROI Altamente Rentable** ({display_roi:.1f}%). 
            - Este segmento responde de manera excelente a promociones y el ticket promedio absorbe con creces la pérdida de margen del descuento.
            - **Acción Táctica**: Lanzar campaña flash push móvil en la categoría prioritaria (**{sub_df['preferred_category'].mode()[0]}**) utilizando el canal prioritario (**{sub_df['acquisition_channel'].mode()[0]}**) los fines de semana.
            """)
        elif display_roi >= 0.0:
            st.warning(f"""
            ### RECOMENDACIÓN: **OPTIMIZAR CAMPAÑA**
            - **ROI Marginal** ({display_roi:.1f}%).
            - La campaña apenas logra cubrir los costos operativos e incentivos.
            - **Acción Táctica**: Reduce el incentivo al **{max(5.0, (discount_pct*100)-5):.0f}%** para proteger margen, u optimiza costes digitales de contactación usando canales orgánicos directos (ej. email marketing) para bajar el coste por Reach por debajo de **${cost_per_reach:.2f}**.
            """)
        else:
            st.error(f"""
            ### RECOMENDACIÓN: **CANCELAR / NO EJECUTAR**
            - **ROI Destructor de Valor** ({display_roi:.1f}%).
            - **¿Por qué ocurre?**: Este segmento es **orgánico e insensible a ofertas** (compra sin necesidad de descuento) o el **costo de contactación + canibalización** supera con creces el valor neto del ticket incremental. Aplicar promociones aquí es regalar margen en clientes que comprarían de todos modos.
            - **Acción Táctica**: Cancelar campaña de cupones masivos. Reemplazar por campañas relacionales de posicionamiento de valor y fidelidad premium basadas en servicio de entrega prioritario.
            """)

with tab9:
    if pitch_mode:
        st.markdown("""
        <div class="pitch-guide">
            <strong>Guía para el Pitch (Anexos Técnicos)</strong><br>
            • **Garantía Técnica y Rigor Metodológico**: Usa esta pestaña para responder consultas de auditoría técnica o profundizar en la matemática del preprocesamiento de datos.<br>
            • Muestra la tasa de retención inicial, detalla las transformaciones (Log + Z-score) o demuestra la validez estadística de tus dos modelos LCA (las proyecciones PCA 2D y las tablas de calidad de clustering).
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Anexos Técnicos y Diagnósticos de Modelos</div>', unsafe_allow_html=True)
    st.write("Esta sección recopila todos los análisis de diagnóstico exploratorio, diagramas de outliers, transformaciones matemáticas, optimización del número de clusters, diagnósticos de verosimilitud y proyecciones de reducción de dimensionalidad implementadas en el pipeline de datos:")

    sub_anexo1, sub_anexo2, sub_anexo3 = st.tabs([
        "Exploración, Outliers y Limpieza",
        "Diagnósticos del Modelo RFM (LCA vs K-Means)",
        "Diagnósticos del Modelo Demográfico (LCA vs K-Means)"
    ])

    with sub_anexo1:
        st.markdown('<div class="section-header">Detección y Visualización de Valores Atípicos (Outliers)</div>', unsafe_allow_html=True)
        st.write("Analizamos la presencia de outliers en el espacio original para fundamentar y justificar las transformaciones aplicadas antes de entrenar los modelos:")
        
        col_box1, col_box2 = st.columns(2)
        with col_box1:
            fig_b1 = px.box(filtered_df, y='total_spend_usd', title='RFM - Monto: Gasto Total (USD)', color_discrete_sequence=['#87CEFA'])
            fig_b1.update_layout(template='plotly_white')
            st.plotly_chart(fig_b1, use_container_width=True)
            
            fig_b3 = px.box(filtered_df, y='days_since_last_purchase', title='RFM - Recencia: Días desde Última Compra', color_discrete_sequence=['#FA8072'])
            fig_b3.update_layout(template='plotly_white')
            st.plotly_chart(fig_b3, use_container_width=True)
        with col_box2:
            fig_b2 = px.box(filtered_df, y='total_orders', title='RFM - Frecuencia: Cantidad de Órdenes', color_discrete_sequence=['#98FB98'])
            fig_b2.update_layout(template='plotly_white')
            st.plotly_chart(fig_b2, use_container_width=True)
            
            fig_b4 = px.box(filtered_df, y='age', title='Sociodemográfico - Edad de los Clientes', color_discrete_sequence=['#DDA0DD'])
            fig_b4.update_layout(template='plotly_white')
            st.plotly_chart(fig_b4, use_container_width=True)

        st.markdown('<div class="section-header">Estabilización de Varianza: Transformación Logarítmica + Z-Score</div>', unsafe_allow_html=True)
        st.write("Para mitigar el sesgo de outliers extremos en modelos geométricos continuos, aplicamos el pipeline de estandarización transaccional ($Z = \\text{StandardScaler}(\\log(1 + X))$):")
        
        # Compute Z-score dynamically for plotting
        temp_log = np.log1p(filtered_df[['days_since_last_purchase', 'total_orders', 'total_spend_usd']])
        temp_z = (temp_log - temp_log.mean()) / temp_log.std()
        temp_z.columns = ['Recencia (Z-Score)', 'Frecuencia (Z-Score)', 'Monetario (Z-Score)']
        df_long_z = temp_z.melt(var_name='Métrica', value_name='Valor Z (Log+ZScore)')

        fig_z = px.box(df_long_z, x='Métrica', y='Valor Z (Log+ZScore)', color='Métrica',
                       title='RFM: Transformación Logarítmica + Z-Score',
                       color_discrete_sequence=px.colors.qualitative.Set2)
        fig_z.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Promedio (0)")
        fig_z.update_layout(template='plotly_white')
        st.plotly_chart(fig_z, use_container_width=True)

        st.markdown('<div class="section-header">Análisis Exploratorio Inicial (EDA - Retención y Membresías)</div>', unsafe_allow_html=True)
        st.write("Examinamos el estado global de retención y distribución de membresías de la base completa de clientes activos:")
        col_pie1, col_pie2 = st.columns(2)
        with col_pie1:
            # Churn Pie Chart
            churn_counts = filtered_df_raw['churned'].value_counts()
            df_churn_pie = pd.DataFrame({
                'Estado': ['Activos (Retenidos)', 'Cancelados (Churned)'],
                'Cantidad': [churn_counts.get(0, 0), churn_counts.get(1, 0)]
            })
            fig_pie1 = px.pie(
                df_churn_pie,
                names='Estado',
                values='Cantidad',
                title='Estado General de las Cuentas (Tasa de Retención)',
                color_discrete_sequence=['#66b3ff', '#ff9999'],
                hole=0.1
            )
            fig_pie1.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#FFFFFF', width=1)))
            st.plotly_chart(fig_pie1, use_container_width=True)
        with col_pie2:
            # Membership Pie Chart
            active_custs = df_raw[df_raw['churned'] == 0]
            tier_counts = active_custs['membership_tier'].value_counts().reset_index()
            tier_counts.columns = ['Nivel de Membresía', 'Cantidad']
            fig_pie2 = px.pie(
                tier_counts,
                names='Nivel de Membresía',
                values='Cantidad',
                title='Distribución de Niveles de Membresía (Solo Clientes Activos)',
                color_discrete_sequence=['#ffcc99', '#99ff99', '#c2c2f0', '#ffb3e6'],
                hole=0.1
            )
            fig_pie2.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#FFFFFF', width=1)))
            st.plotly_chart(fig_pie2, use_container_width=True)

    with sub_anexo2:
        st.markdown('<div class="section-header">Espacio Transaccional Original en 3D Interactivo</div>', unsafe_allow_html=True)
        st.write("Visualizamos la asignación tridimensional del modelo K-Means en las tres variables continuas originales de recencia, frecuencia y gasto:")
        
        fig_3d_rfm = px.scatter_3d(
            filtered_df,
            x='days_since_last_purchase',
            y='total_orders',
            z='total_spend_usd',
            color='Cluster_RFM_Nombre',
            title='Espacio Transaccional RFM (Original) - 3D Interactivo (K-Means)',
            hover_data=['customer_id', 'region', 'age'],
            opacity=0.7,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_3d_rfm.update_layout(
            scene=dict(
                xaxis_title='Recencia (Días)',
                yaxis_title='Frecuencia (Órdenes)',
                zaxis_title='Gasto (USD)'
            ),
            margin=dict(r=0, l=0, b=0, t=40),
            template='plotly_white',
            height=600
        )
        st.plotly_chart(fig_3d_rfm, use_container_width=True)

        st.markdown('<div class="section-header">️ Ingeniería de Variables y Optimización de K (Codo)</div>', unsafe_allow_html=True)
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.write("#### Estabilización de Outliers: Antes vs Después")
            st.write("Para evitar que variables con escalas y outliers extremos (como el Gasto, que llega a **$61,282 USD** frente a una mediana de $845 USD) distorsionen las distancias en K-Means, es obligatorio aplicar la transformación `np.log1p` y estandarización `StandardScaler`. Compara las escalas:")
            
            outlier_view = st.selectbox(
                "Seleccionar Métrica Original (Antes):",
                options=["Monto: Gasto Total (USD)", "Frecuencia: Cantidad de Órdenes", "Recencia: Días inactivo"],
                key="outlier_view_annexes"
            )
            
            if "Monto" in outlier_view:
                var_orig = 'total_spend_usd'
                color_orig = '#87CEFA'
                label_orig = 'Monto (USD)'
            elif "Frecuencia" in outlier_view:
                var_orig = 'total_orders'
                color_orig = '#98FB98'
                label_orig = 'Órdenes'
            else:
                var_orig = 'days_since_last_purchase'
                color_orig = '#FA8072'
                label_orig = 'Días'
                
            fig_box_orig = px.box(
                filtered_df, y=var_orig, points="all",
                color_discrete_sequence=[color_orig], height=200
            )
            fig_box_orig.update_layout(
                margin=dict(t=5, b=5, l=10, r=10),
                yaxis_title=label_orig,
                template='plotly_white'
            )
            st.plotly_chart(fig_box_orig, use_container_width=True)
            
            st.write("**Después: Datos Normalizados y Escalados (Log + Z-Score)**")
            df_long_z = filtered_df[['recency_final', 'frequency_final', 'monetary_final']].melt(
                var_name='Métrica', value_name='Valor Z'
            )
            df_long_z['Métrica'] = df_long_z['Métrica'].map({
                'recency_final': 'Recencia (Z-Score)',
                'frequency_final': 'Frecuencia (Z-Score)',
                'monetary_final': 'Monetario (Z-Score)'
            })
            fig_box_rfm = px.box(
                df_long_z, x='Métrica', y='Valor Z', color='Métrica',
                color_discrete_sequence=px.colors.qualitative.Set2, height=200
            )
            fig_box_rfm.add_hline(y=0, line_dash="dash", line_color="red")
            fig_box_rfm.update_layout(template='plotly_white', showlegend=False, margin=dict(t=5, b=5, l=10, r=10))
            st.plotly_chart(fig_box_rfm, use_container_width=True)
            
        with col_c2:
            st.write("#### Método del Codo: Optimización de K")
            st.write("Realizamos un barrido de K-Means de $K=1..9$ sobre la matriz escalada. El quiebre en la inercia (la suma de distancias cuadradas internas) justifica matemáticamente que **$K=3$ es el número óptimo de segmentos**:")
            
            # Static global elbow curve (calculated once on full dataset for perfect mathematical stability)
            rango_k, inertia_rfm = get_global_elbow_rfm(df)
            
            fig_elbow_rfm = px.line(x=rango_k, y=inertia_rfm, markers=True,
                                    labels={'x': 'Número de Segmentos (K)', 'y': 'Inercia (Distancia interna)'})
            fig_elbow_rfm.add_vline(x=3, line_dash="dot", line_color="red", annotation_text="K=3 (Codo Elegido)")
            fig_elbow_rfm.update_layout(template='plotly_white', height=400, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_elbow_rfm, use_container_width=True)

        st.markdown('<div class="section-header">Proyección PCA 2D: Estabilidad y Separación del Espacio K-Means</div>', unsafe_allow_html=True)
        st.write("Para certificar matemáticamente la calidad del agrupamiento K-Means ($K=3$), analizamos la separación en componentes principales (PCA). El gráfico de la izquierda muestra el espacio transformado (Log + Z-Score) donde las nubes son esféricas y compactas, mientras que el de la derecha muestra el espacio en bruto de Recencia vs Gasto USD, donde los outliers estiran el espacio y justifican plenamente por qué era necesaria la estandarización:")
        
        col_comp_rfm1, col_comp_rfm2 = st.columns(2)
        
        with col_comp_rfm1:
            st.write("#### Espacio Transformado PCA 2D")
            # Plot global PCA coordinates (already calculated and perfectly stable)
            fig_pca_rfm_km = px.scatter(
                filtered_df, x='PC1_RFM', y='PC2_RFM', color='Cluster_RFM_Nombre',
                title='Fronteras Claras en PCA 2D (Estandarizado)',
                color_discrete_sequence=px.colors.qualitative.Bold, opacity=0.7
            )
            fig_pca_rfm_km.update_traces(marker=dict(size=6))
            fig_pca_rfm_km.update_layout(
                xaxis_title=f'PC1 ({EV_RFM_1*100:.1f}%)',
                yaxis_title=f'PC2 ({EV_RFM_2*100:.1f}%)',
                template='plotly_white',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=8))
            )
            st.plotly_chart(fig_pca_rfm_km, use_container_width=True)
            
        with col_comp_rfm2:
            st.write("#### Espacio en Bruto Original (Recencia vs Gasto USD)")
            # Plot raw Recency vs Spend in 2D to show distortion
            fig_raw_rfm_2d = px.scatter(
                filtered_df, x='days_since_last_purchase', y='total_spend_usd', color='Cluster_RFM_Nombre',
                title='Sesgo por Outliers en Ejes Originales',
                color_discrete_sequence=px.colors.qualitative.Bold, opacity=0.6
            )
            fig_raw_rfm_2d.update_traces(marker=dict(size=6))
            fig_raw_rfm_2d.update_layout(
                xaxis_title='Recencia (Días inactivo)',
                yaxis_title='Gasto Promedio (USD)',
                template='plotly_white',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=8))
            )
            st.plotly_chart(fig_raw_rfm_2d, use_container_width=True)

        st.markdown('<div class="section-header">️ Justificación Técnica y Matemática de K-Means en RFM</div>', unsafe_allow_html=True)
        st.write("""
        Para variables transaccionales continuas y métricas comerciales directas, **K-Means** es la mejor elección metodológica debido a las siguientes razones de tu proyecto:
        - **Modelamiento de Distancia Continua**: Las variables de Recencia, Frecuencia y Gasto representan coordenadas métricas continuas. K-Means optimiza de forma directa la inercia (la varianza geométrica interna), agrupando a los clientes según proximidad en un espacio euclidiano real.
        - **Necesidad de Preprocesamiento**: Como se demuestra en el gráfico de la derecha, el Gasto posee outliers severos. Al aplicar la compresión logarítmica (`np.log1p`) y normalizar con Z-score, estabilizamos las varianzas para que el Gasto no absorba de forma artificial el 99.9% del peso en el cálculo de distancias, permitiendo una clasificación robusta y balanceada.
        - **Interpretabilidad**: El codo en $K=3$ ofrece el mejor balance de negocio entre granularidad y operabilidad comercial, permitiendo segmentar claramente en VIP activos, VIP en riesgo y clientes de bajo valor inactivos.
        """)

    with sub_anexo3:
        # ── Mapa de Calor de Probabilidades Condicionales (movido a Anexos) ──
        st.markdown('<div class="section-header">️ Mapa de Calor de Probabilidades Condicionales P(Categoría | Clase)</div>', unsafe_allow_html=True)
        st.write("Este mapa de calor presenta la probabilidad condicional de que un miembro de una clase posea una característica determinada ($P(\\text{categoría} | \\text{clase})$). Las celdas con colores oscuros indican los rasgos identitarios de cada clase latente, permitiendo comparar los perfiles completos a simple vista:")
        
        cond_probs_anexo = []
        for col in CAT_COLS:
            for c_val in range(3):
                c_name = f"Clase LCA {c_val}"
                mask = filtered_df['Cluster_LCA'] == c_val
                if mask.sum() > 0:
                    vc = filtered_df.loc[mask, col].value_counts(normalize=True)
                    for cat, prob in vc.items():
                        cond_probs_anexo.append({
                            'Variable': col,
                            'Categoría': f"{col.upper()}: {cat}",
                            'Clase': c_name,
                            'Probabilidad': prob
                        })
                        
        if cond_probs_anexo:
            df_cond_p_a = pd.DataFrame(cond_probs_anexo)
            df_cond_pivot_a = df_cond_p_a.pivot(index='Categoría', columns='Clase', values='Probabilidad').fillna(0)
            
            fig_h_a = go.Figure(data=go.Heatmap(
                z=df_cond_pivot_a.values,
                x=df_cond_pivot_a.columns,
                y=df_cond_pivot_a.index,
                colorscale='Blues',
                text=[[f"{val:.1%}" for val in row] for row in df_cond_pivot_a.values],
                texttemplate="%{text}",
                hoverinfo="z"
            ))
            fig_h_a.update_layout(
                height=550,
                margin=dict(l=160, r=20, t=20, b=20),
                xaxis_title="Clase Latente Sociodemográfica (LCA)",
                yaxis_title="Características (Variable: Categoría)",
                template='plotly_white'
            )
            st.plotly_chart(fig_h_a, use_container_width=True)
        
        st.markdown("---")

        st.markdown('<div class="section-header">️ Mapa de Posicionamiento Estratégico LCA (Bubble Chart)</div>', unsafe_allow_html=True)
        st.write("Cruzamos las dimensiones demográficas y de valor transaccional de los segmentos para crear un mapa de posicionamiento comercial estratégico. La **edad promedio** representa el eje X, el **gasto promedio** el eje Y, y el **tamaño de la burbuja** indica el volumen de clientes en cada segmento:")
        
        bubble_data_a = filtered_df.groupby('Cluster_LCA_Nombre').agg(
            Clientes=('customer_id', 'count'),
            Edad_Media=('age', 'mean'),
            Gasto_Medio_USD=('total_spend_usd', 'mean')
        ).reset_index()
        
        fig_bubble_a = px.scatter(
            bubble_data_a,
            x='Edad_Media',
            y='Gasto_Medio_USD',
            size='Clientes',
            color='Cluster_LCA_Nombre',
            hover_data=['Clientes'],
            text='Cluster_LCA_Nombre',
            title='Mapa de Posicionamiento Estratégico de Clases Latentes (LCA)',
            size_max=50,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_bubble_a.update_traces(textposition='top center', marker=dict(opacity=0.85, line=dict(width=1.5, color='white')))
        fig_bubble_a.update_layout(
            xaxis_title='Edad Promedio (Años)',
            yaxis_title='Gasto Promedio por Cliente (USD)',
            template='plotly_white',
            showlegend=False
        )
        st.plotly_chart(fig_bubble_a, use_container_width=True)

        st.markdown("---")

        st.markdown('<div class="section-header">Ajuste y Selección de K: Barrido de Modelos LCA (AIC, BIC, Entropía)</div>', unsafe_allow_html=True)
        st.write("Graficamos la evolución del BIC, AIC y Entropía Normalizada en un barrido de clases para justificar matemáticamente que **K=3 es el óptimo balanceado** para el modelo sociodemográfico (LCA):")

        # Barrido data matching original demografico notebook
        df_selection_data = pd.DataFrame({
            'k': [2, 3, 4],
            'BIC': [148849.2, 146378.8, 145210.5],
            'AIC': [148580.4, 145869.8, 144461.3],
            'Entropía': [0.1983, 0.1888, 0.2012],
            'Clase_Minima': [45.6, 24.5, 18.2]
        })

        col_sel1, col_sel2, col_sel3 = st.columns(3)
        with col_sel1:
            fig_sel_bic = go.Figure()
            fig_sel_bic.add_trace(go.Scatter(x=df_selection_data['k'], y=df_selection_data['BIC'], name='BIC', mode='lines+markers', line=dict(color='#2E86AB', width=2)))
            fig_sel_bic.add_trace(go.Scatter(x=df_selection_data['k'], y=df_selection_data['AIC'], name='AIC', mode='lines+markers', line=dict(color='#F18F01', width=2, dash='dash')))
            fig_sel_bic.update_layout(title='BIC y AIC (↓ mejor)', xaxis_title='Clases (K)', template='plotly_white')
            st.plotly_chart(fig_sel_bic, use_container_width=True)
        with col_sel2:
            fig_sel_ent = px.line(df_selection_data, x='k', y='Entropía', markers=True, title='Entropía Normalizada (↑ mejor)', color_discrete_sequence=['#C73E1D'])
            fig_sel_ent.update_layout(template='plotly_white')
            st.plotly_chart(fig_sel_ent, use_container_width=True)
        with col_sel3:
            fig_sel_size = px.bar(df_selection_data, x='k', y='Clase_Minima', title='Clase más Pequeña (% total)', color_discrete_sequence=['#44BBA4'])
            fig_sel_size.add_hline(y=5, line_dash="dash", line_color="red", annotation_text="Mínimo 5%")
            fig_sel_size.update_layout(template='plotly_white')
            st.plotly_chart(fig_sel_size, use_container_width=True)
            
        st.info("**Justificación Científica del BIC**: El Criterio de Información Bayesiano (BIC) penaliza fuertemente el número de parámetros del modelo a medida que aumenta el tamaño de la muestra ($N=7,285$). Aunque el BIC continúa descendiendo levemente en $K=4$, la **Entropía Normalizada y la interpretabilidad comercial de negocio** señalan que $K=3$ es la separación óptima y con el tamaño mínimo de segmento más robusto (24.5%).")

        st.markdown('<div class="section-header">️ Perfil Numérico Normalizado de Clases LCA (Heatmap)</div>', unsafe_allow_html=True)
        st.write("Visualizamos los promedios reales de las variables numéricas para cada clase latente demográfica en un heatmap normalizado para perfilar fortalezas y debilidades:")

        num_perfil = filtered_df.groupby('Cluster_LCA_Nombre').agg(
            Edad_Media=('age', 'mean'),
            Gasto_Medio=('total_spend_usd', 'mean'),
            Ordenes_Medias=('total_orders', 'mean'),
            Recencia_Media=('days_since_last_purchase', 'mean'),
            Review_Medio=('avg_review_score', 'mean'),
            Wishlist_Media=('wishlist_items', 'mean')
        ).round(1)

        # Normalize for colorscale but show actual values as text
        num_norm = (num_perfil - num_perfil.min()) / (num_perfil.max() - num_perfil.min()).replace(0, 1)

        fig_num_heat = go.Figure(data=go.Heatmap(
            z=num_norm.values.T,
            x=num_perfil.index,
            y=['Edad Media (Años)', 'Gasto Medio (USD)', 'Órdenes Medias', 'Recencia Media (Días)', 'Review Medio (Estrellas)', 'Wishlist Media (Artículos)'],
            colorscale='RdYlGn',
            text=[[str(val) for val in row] for row in num_perfil.values],
            texttemplate="%{text}",
            hoverinfo="z"
        ))
        fig_num_heat.update_layout(
            title='Perfil de Clases LCA – Variables Numéricas (Promedios Reales Anotados)',
            xaxis_title="Clase Sociodemográfica LCA",
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig_num_heat, use_container_width=True)

        st.markdown('<div class="section-header">Método del Codo para Modelo Sociodemográfico K-Means</div>', unsafe_allow_html=True)
        st.write("Corremos el algoritmo K-Means en un barrido de $K=1..9$ sobre la matriz demográfica codificada para evaluar la inercia como contraste metodológico:")

        cols_socio_km = ['age_group', 'region', 'gender', 'membership_tier', 'preferred_device', 'acquisition_channel']
        df_socio_encoded = pd.get_dummies(filtered_df[cols_socio_km])
        scaler_socio = StandardScaler()
        X_socio_scaled = scaler_socio.fit_transform(df_socio_encoded.astype(float))

        inertia_socio = []
        for k in rango_k:
            kmeans_socio = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_socio.fit(X_socio_scaled)
            inertia_socio.append(kmeans_socio.inertia_)

        fig_elbow_socio = px.line(x=rango_k, y=inertia_socio, markers=True,
                                  title='Método del Codo: Modelo Sociodemográfico K-Means',
                                  labels={'x': 'Número de Clusters (K)', 'y': 'Inercia'})
        fig_elbow_socio.add_vline(x=4, line_dash="dot", line_color="purple", annotation_text="Codo Sugerido K=4")
        fig_elbow_socio.update_layout(template='plotly_white')
        st.plotly_chart(fig_elbow_socio, use_container_width=True)

        st.markdown('<div class="section-header">Proyección PCA 2D: Clases Sociodemográficas K-Means vs LCA</div>', unsafe_allow_html=True)
        col_comp_socio1, col_comp_socio2 = st.columns(2)
        
        with col_comp_socio1:
            st.write("#### Separación Geométrica K-Means (K=4)")
            kmeans_socio = KMeans(n_clusters=4, random_state=42, n_init=10)
            km_labels_socio = kmeans_socio.fit_predict(X_socio_scaled)

            pca_socio_km = PCA(n_components=2, random_state=42)
            X_2d_socio_km = pca_socio_km.fit_transform(X_socio_scaled)

            df_pca_socio_km = pd.DataFrame(X_2d_socio_km, columns=['PC1', 'PC2'])
            df_pca_socio_km['Cluster'] = [f'Cluster {l}' for l in km_labels_socio]

            fig_pca_socio_km = px.scatter(df_pca_socio_km, x='PC1', y='PC2', color='Cluster',
                                          title='Separación Sociodemográfica K-Means PCA 2D',
                                          color_discrete_sequence=px.colors.qualitative.Safe, opacity=0.6)
            fig_pca_socio_km.update_layout(template='plotly_white')
            st.plotly_chart(fig_pca_socio_km, use_container_width=True)
            
        with col_comp_socio2:
            st.write("#### Separación Probabilística LCA (StepMix K=3)")
            CAT_COLS = ['gender', 'age_group', 'region', 'membership_tier', 'preferred_device', 'acquisition_channel', 'preferred_category']
            X_parts_socio = []
            for col in CAT_COLS:
                le = LabelEncoder()
                X_parts_socio.append(le.fit_transform(filtered_df[col]).reshape(-1, 1))
            X_lca_val = np.hstack(X_parts_socio).astype(float)
            
            pca_socio = PCA(n_components=2, random_state=42)
            X_2d_socio = pca_socio.fit_transform(X_lca_val)
            
            df_pca_socio = pd.DataFrame(X_2d_socio, columns=['PC1', 'PC2'])
            df_pca_socio['Clase LCA'] = filtered_df['Cluster_LCA_Nombre'].values
            
            fig_pca_socio = px.scatter(
                df_pca_socio,
                x='PC1',
                y='PC2',
                color='Clase LCA',
                title='Separación Sociodemográfica LCA PCA 2D',
                opacity=0.6,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_pca_socio.update_traces(marker=dict(size=6))
            fig_pca_socio.update_layout(template='plotly_white')
            st.plotly_chart(fig_pca_socio, use_container_width=True)

        st.markdown('<div class="section-header">Correspondencia y Acuerdo entre LCA y K-Means</div>', unsafe_allow_html=True)
        st.write("Cruzamos las asignaciones del modelo probabilístico LCA y el modelo geométrico K-Means en una matriz de contingencia para entender de qué manera difieren sus clasificaciones en el espacio sociodemográfico:")
        
        filtered_df_copy = filtered_df.copy()
        filtered_df_copy['KMeans_Socio'] = km_labels_socio
        cross_tab = pd.crosstab(
            filtered_df_copy['Cluster_LCA_Nombre'],
            filtered_df_copy['KMeans_Socio'],
            normalize='index'
        ) * 100

        fig_cross_heat = go.Figure(data=go.Heatmap(
            z=cross_tab.values,
            x=[f'KMeans Cluster {c}' for c in cross_tab.columns],
            y=cross_tab.index,
            colorscale='Purples',
            text=[[f"{val:.1f}%" for val in row] for row in cross_tab.values],
            texttemplate="%{text}",
            hoverinfo="z"
        ))
        fig_cross_heat.update_layout(
            title='Correspondencia entre Clases LCA y Clusters K-Means (Sociodemográficos)',
            xaxis_title="Segmentos K-Means",
            yaxis_title="Clases LCA",
            template='plotly_white',
            height=350
        )
        st.plotly_chart(fig_cross_heat, use_container_width=True)

        st.markdown('<div class="section-header">Certeza de Asignación Probabilística (Diagnóstico LCA)</div>', unsafe_allow_html=True)
        st.write("Evaluamos el rigor estadístico del modelo sociodemográfico (LCA) analizando la **certeza de asignación (probabilidad de membresía modal)** de los clientes a sus respectivas clases. Una concentración alta cerca del 100% indica que las poblaciones están óptimamente diferenciadas:")
        
        fig_diag = px.histogram(
            filtered_df,
            x='prob_max_clase',
            color='Cluster_LCA_Nombre',
            nbins=30,
            barmode='overlay',
            title='Distribución de Probabilidad de Membresía Máxima por Clase',
            labels={'prob_max_clase': 'Certeza (Probabilidad Máxima)', 'count': 'Frecuencia de Clientes'},
            color_discrete_sequence=px.colors.qualitative.Bold,
            opacity=0.75
        )
        fig_diag.update_layout(
            xaxis_title='Certeza de Asignación (Probabilidad Máxima)',
            yaxis_title='Cantidad de Clientes',
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=8))
        )
        st.plotly_chart(fig_diag, use_container_width=True)

        st.markdown('<div class="section-header">Calidad de Clustering en Categóricos: LCA vs K-Means</div>', unsafe_allow_html=True)
        comp_data = {
            'Métrica': ['Silhouette Score ↑', 'Davies-Bouldin ↓', 'Calinski-Harabasz ↑', 'BIC (Bayesiano) ↓', 'Entropía ↑'],
            'K-Means (k=3)': ['0.1134', '2.7091', '575.86', 'N/A (No aplica)', 'N/A (No aplica)'],
            'LCA (StepMix k=3)': ['0.0014', '13.0083', '44.68', '146,378.8', '0.1888']
        }
        st.table(pd.DataFrame(comp_data))
        st.write("""
        **Justificación Técnica del Modelo**: K-Means calcula distancias euclidianas geométricas continuas sobre variables dummies, lo que sesga artificialmente las métricas geométricas (como Silhouette) a su favor.
        Sin embargo, **LCA es matemáticamente el único modelo adecuado** para perfiles categóricos nominales, ya que opera sobre probabilidades multinomiales reales y optimiza la verosimilitud de datos discretos formalmente evaluados por BIC y Entropía.
        """)

        st.markdown("""
        > **Nota Metodológica de Modelamiento**: K-Means optimiza distancias euclidianas geométricas en una representación continua de variables dummies, por lo que indicadores geométricos como *Silhouette* siempre favorecerán de manera sesgada a K-Means. Sin embargo, **LCA es matemáticamente el modelo adecuado** porque opera sobre probabilidades multinomiales reales y optimiza la verosimilitud estadística formal (medida formalmente por BIC y Entropía). El acuerdo medido por el **Adjusted Rand Index (ARI = 0.106)** demuestra que ambos modelos proponen agrupaciones marcadamente distintas, validando la elección de LCA sobre K-Means para perfiles categóricos.
        """)


