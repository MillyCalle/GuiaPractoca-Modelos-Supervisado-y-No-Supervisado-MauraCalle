import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    silhouette_score,
    davies_bouldin_score,
)
from sklearn.cluster import KMeans

# --------------------------------------------------------------------
# 1. CARGA Y PREPARACIÓN DE DATOS
# --------------------------------------------------------------------


@st.cache_data
def cargar_datos(ruta_csv: str = "data/academic_performance_master.csv") -> pd.DataFrame:
    # El header real está en la fila 3 (como en tu notebook)
    df = pd.read_csv(ruta_csv, header=3)

    # Eliminar columnas completamente vacías
    df = df.dropna(axis=1, how="all")
    df = df.dropna(how="all").reset_index(drop=True)

    # Asegurar nombres correctos (según lo que vimos en el notebook)
    # Columnas esperadas:
    # ['Periodo', 'Paralelo', 'Identificacion', 'Estudiante', 'Carrera',
    #  'Nivel', 'Asignatura', 'Num_matricula', 'Asistencia', 'Nota final',
    #  'Estado', 'Estado Matrícula', 'Tipo Ingreso', 'Cédula docente',
    #  'Nombre docente']

    # Convertir a numérico
    df["Asistencia"] = pd.to_numeric(df["Asistencia"], errors="coerce")
    df["Nota final"] = pd.to_numeric(df["Nota final"], errors="coerce")
    df["Num_matricula"] = pd.to_numeric(df["Num_matricula"], errors="coerce")

    # Rellenar nulos numéricos
    df[["Asistencia", "Nota final", "Num_matricula"]] = df[
        ["Asistencia", "Nota final", "Num_matricula"]
    ].fillna(0)

    # Variable objetivo: Aprobado (1) / Reprobado (0)
    df["Aprobado"] = df["Estado"].apply(
        lambda x: 1 if str(x).strip().upper() == "APROBADO" else 0
    )

    return df


# --------------------------------------------------------------------
# 2. CONFIGURACIÓN BÁSICA DE LA APP
# --------------------------------------------------------------------

st.set_page_config(
    page_title="Modelos Supervisado y No Supervisado – Notas",
    layout="wide",
)

# CSS mejorado pero simple
st.markdown("""
    <style>
    h1 { 
        color: #1e3a8a; 
        border-bottom: 3px solid #3b82f6; 
        padding-bottom: 1rem; 
    }
    h2 { 
        color: #1e40af; 
        background: linear-gradient(90deg, #dbeafe 0%, transparent 100%); 
        padding: 0.8rem; 
        border-radius: 8px; 
        margin-top: 1.5rem;
    }
    .stButton>button { 
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white; 
        font-weight: 600; 
        border-radius: 8px;
        padding: 0.6rem 2rem; 
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Modelos Supervisado y No Supervisado con Notas de Estudiantes")
st.write(
    """
Esta aplicación utiliza el **reporte maestro de notas** para:

1. Explorar el dataset de rendimiento académico.  
2. Entrenar un **modelo supervisado (clasificación)** para predecir si un estudiante aprobará.  
3. Aplicar un **modelo no supervisado (K-Means)** para agrupar estudiantes según **asistencia** y **nota final**.
"""
)

df = cargar_datos()

# Sidebar
st.sidebar.title("Navegación")
seccion = st.sidebar.radio(
    "Ir a:",
    ("Exploración de datos", "Modelo supervisado (clasificación)", "Modelo no supervisado (clustering)"),
)

# Estadísticas en sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Estadísticas Generales")
st.sidebar.metric("Total Registros", f"{df.shape[0]:,}")
st.sidebar.metric("Tasa Aprobación", f"{(df['Aprobado'].mean()*100):.1f}%")


# --------------------------------------------------------------------
# 3. EXPLORACIÓN DE DATOS
# --------------------------------------------------------------------
if seccion == "Exploración de datos":
    st.header("🔍 Exploración de datos")

    st.subheader("Vista general del dataset")
    st.dataframe(df.head(50))

    st.subheader("Información básica")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Dimensiones del dataset:**")
        st.write(f"Filas: {df.shape[0]:,}")
        st.write(f"Columnas: {df.shape[1]}")

        st.write("**Columnas disponibles:**")
        st.write(list(df.columns))

    with col2:
        st.write("**Descripción estadística (variables numéricas)**")
        st.write(df[["Asistencia", "Nota final", "Num_matricula"]].describe())

    st.subheader("Distribución de Nota Final")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["Nota final"], bins=20, kde=True, ax=ax, color="#3b82f6")
    ax.set_xlabel("Nota final", fontsize=11, fontweight='bold')
    ax.set_ylabel("Frecuencia", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

    st.subheader("Distribución de Asistencia")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.histplot(df["Asistencia"], bins=20, kde=True, ax=ax2, color="#10b981")
    ax2.set_xlabel("Asistencia", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Frecuencia", fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    plt.close()

    st.subheader("Aprobados vs Reprobados")
    conteo_estado = df["Aprobado"].value_counts().rename(index={0: "Reprobado", 1: "Aprobado"})
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    bars = ax3.bar(conteo_estado.index, conteo_estado.values, color=["#ef4444", "#10b981"])
    ax3.set_ylabel("Número de estudiantes", fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores en las barras
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
    
    st.pyplot(fig3)
    plt.close()


# --------------------------------------------------------------------
# 4. MODELO SUPERVISADO – CLASIFICACIÓN
# --------------------------------------------------------------------
elif seccion == "Modelo supervisado (clasificación)":
    st.header("🎓 Modelo Supervisado – Predicción de Aprobación")

    st.write(
        """
Usamos un modelo de **Regresión Logística** para predecir si un estudiante
**aprobará (1) o reprobará (0)** a partir de:
- Asistencia  
- Nota final  
- Número de matrículas
"""
    )

    # Parámetros ajustables
    st.sidebar.subheader("Parámetros del modelo supervisado")
    test_size = st.sidebar.slider("Proporción de test", 0.1, 0.4, 0.3, 0.05)
    c_value = st.sidebar.slider("Regularización (C)", 0.01, 5.0, 1.0, 0.05)

    if st.button("🚀 Entrenar modelo supervisado"):
        with st.spinner("🔄 Entrenando modelo..."):
            X = df[["Asistencia", "Nota final", "Num_matricula"]]
            y = df["Aprobado"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = LogisticRegression(C=c_value, max_iter=1000)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            acc = accuracy_score(y_test, y_pred)

        st.success("✅ Modelo entrenado exitosamente")
        
        st.subheader("Resultados del modelo")
        
        # Métricas en columnas
        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 Accuracy", f"{acc:.4f}")
        col2.metric("📊 Test Size", len(X_test))
        col3.metric("🎓 Train Size", len(X_train))

        st.subheader("Reporte de clasificación")
        reporte = classification_report(
            y_test, y_pred, target_names=["Reprobado", "Aprobado"], output_dict=False
        )
        st.text(reporte)

        st.subheader("Matriz de confusión")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Reprobado", "Aprobado"],
            yticklabels=["Reprobado", "Aprobado"],
            ax=ax_cm,
            square=True,
            linewidths=2
        )
        ax_cm.set_xlabel("Predicción", fontsize=11, fontweight='bold')
        ax_cm.set_ylabel("Real", fontsize=11, fontweight='bold')
        st.pyplot(fig_cm)
        plt.close()

        st.info(
            "💡 El modelo muestra un buen desempeño si el **accuracy** es alto y los valores de la matriz de confusión "
            "se concentran en la diagonal principal (predicciones correctas)."
        )


# --------------------------------------------------------------------
# 5. MODELO NO SUPERVISADO – K-MEANS CLUSTERING
# --------------------------------------------------------------------
elif seccion == "Modelo no supervisado (clustering)":
    st.header("🤖 Modelo No Supervisado – Clustering de Estudiantes (K-Means)")

    st.write(
        """
Aplicamos **K-Means** para agrupar estudiantes según:
- **Asistencia**
- **Nota final**

El objetivo es descubrir **perfiles** de rendimiento académico sin usar la etiqueta de aprobado/reprobado.
"""
    )

    st.sidebar.subheader("Parámetros del modelo no supervisado")
    n_clusters = st.sidebar.slider("Número de clusters (K)", 2, 6, 3, 1)

    # Datos numéricos para clustering
    X_cluster = df[["Asistencia", "Nota final"]].dropna()

    if st.button("🚀 Ejecutar clustering K-Means"):
        with st.spinner("🔄 Ejecutando algoritmo K-Means..."):
            # Escalamiento opcional
            scaler_c = StandardScaler()
            X_scaled = scaler_c.fit_transform(X_cluster)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(X_scaled)

            df_cluster = X_cluster.copy()
            df_cluster["Cluster"] = labels

        st.success("✅ Clustering completado")

        st.subheader("Visualización de clusters")
        fig_cl, ax_cl = plt.subplots(figsize=(10, 7))
        sns.scatterplot(
            data=df_cluster,
            x="Asistencia",
            y="Nota final",
            hue="Cluster",
            palette="viridis",
            ax=ax_cl,
            s=80,
            alpha=0.7
        )

        # Centroides (en espacio escalado -> los llevamos de vuelta)
        centroids_scaled = kmeans.cluster_centers_
        centroids = scaler_c.inverse_transform(centroids_scaled)
        ax_cl.scatter(
            centroids[:, 0],
            centroids[:, 1],
            c="red",
            s=300,
            marker="X",
            label="Centroides",
            edgecolor='darkred',
            linewidth=2
        )
        ax_cl.set_xlabel("Asistencia", fontsize=11, fontweight='bold')
        ax_cl.set_ylabel("Nota final", fontsize=11, fontweight='bold')
        ax_cl.set_title("Clusters de estudiantes por Asistencia y Nota Final", 
                       fontsize=12, fontweight='bold', pad=15)
        ax_cl.legend()
        ax_cl.grid(True, alpha=0.3)
        st.pyplot(fig_cl)
        plt.close()

        st.subheader("Métricas de calidad del clustering")
        silhouette = silhouette_score(X_scaled, labels)
        davies = davies_bouldin_score(X_scaled, labels)
        inertia = kmeans.inertia_

        colm1, colm2, colm3 = st.columns(3)
        colm1.metric("Silhouette Score", f"{silhouette:.4f}")
        colm2.metric("Davies-Bouldin Index", f"{davies:.4f}")
        colm3.metric("Inercia (SSE)", f"{inertia:.2f}")

        st.write(
            """
**Interpretación rápida:**
- Un **Silhouette** más cercano a 1 indica mejores clusters.
- Un **Davies-Bouldin** más cercano a 0 indica mejor separación.
- Una **Inercia** menor indica clusters más compactos.
"""
        )

        st.subheader("Tamaño de cada cluster")
        cluster_counts = df_cluster["Cluster"].value_counts().sort_index()
        
        # Mostrar en columnas
        cols = st.columns(n_clusters)
        for idx, (cluster_id, count) in enumerate(cluster_counts.items()):
            cols[idx].metric(f"Cluster {cluster_id}", f"{count:,} estudiantes")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 1rem;">
    <p><strong>Sistema de Análisis de Rendimiento Académico</strong> | Desarrollado con Streamlit | Realizado por: Maura Calle 🎈</p>
</div>
""", unsafe_allow_html=True)