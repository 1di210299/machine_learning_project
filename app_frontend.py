import streamlit as st
import requests
import pandas as pd
from PIL import Image
import io
import os
import plotly.express as px
import plotly.graph_objects as go

# Configuración inicial
st.set_page_config(
    page_title="Sistema de Predicción de Sensores",
    layout="wide"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Configuración de la API
API_URL = "http://localhost:8000"

def main():
    st.title("🤖 Sistema de Predicción de Sensores")
    
    # Creación de columnas en el sidebar
    with st.sidebar:
        st.header("Configuración")
        sensor_type = st.selectbox(
            "Tipo de Sensor",
            ['ELF', 'MAG', 'GEO']
        )
        
        size = st.selectbox(
            "Tamaño",
            [160, 320, 480]
        )
    
    # Crear tabs para las diferentes funcionalidades
    tab1, tab2, tab3 = st.tabs(["Entrenamiento", "Evaluación", "Predicción"])
    
    # Tab de Entrenamiento
    with tab1:
        st.header("📊 Entrenamiento del Modelo")
        col1, col2 = st.columns([2,1])
        
        with col1:
            st.markdown("""
                ### Instrucciones
                1. Los datos deben estar en formato CSV
                2. Debe contener las columnas necesarias para el entrenamiento
                3. Los datos serán procesados automáticamente
            """)
            
            if st.button("Iniciar Entrenamiento", type="primary"):
                with st.spinner("Entrenando modelo..."):
                    try:
                        response = requests.post(f"{API_URL}/train")
                        if response.status_code == 200:
                            st.success("✅ Modelo entrenado exitosamente")
                            st.json(response.json())
                        else:
                            st.error("❌ Error en el entrenamiento")
                    except Exception as e:
                        st.error(f"Error de conexión: {str(e)}")
        
        with col2:
            st.metric(label="Sensor Seleccionado", value=sensor_type)
            st.metric(label="Tamaño", value=size)
    
    # Tab de Evaluación
    with tab2:
        st.header("🔍 Evaluación del Modelo")
        
        if st.button("Ejecutar Pruebas", type="primary"):
            with st.spinner("Ejecutando evaluación..."):
                try:
                    response = requests.post(f"{API_URL}/test")
                    if response.status_code == 200:
                        results = response.json()
                        
                        # Mostrar resumen
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Resumen de Pruebas")
                            st.info(f"Pruebas Básicas: {results['basic_tests']}")
                            st.info(f"Pruebas Extremas: {results['extreme_tests']}")
                        
                        # Mostrar matriz de confusión
                        with col2:
                            st.subheader("Matriz de Confusión")
                            cm_response = requests.get(
                                f"{API_URL}/test-results/confusion-matrix/{sensor_type.lower()}/{size}"
                            )
                            if cm_response.status_code == 200:
                                image = Image.open(io.BytesIO(cm_response.content))
                                st.image(image)
                        
                        # Mostrar resultados detallados
                        if results.get('detailed_results'):
                            st.subheader("Resultados Detallados")
                            st.json(results['detailed_results'])
                    else:
                        st.error("Error en la evaluación")
                except Exception as e:
                    st.error(f"Error de conexión: {str(e)}")
    
    # Tab de Predicción
    with tab3:
        st.header("🎯 Predicción")
        
        uploaded_file = st.file_uploader(
            "Subir archivo CSV para predicción",
            type=['csv']
        )
        
        if uploaded_file is not None:
            df_preview = pd.read_csv(uploaded_file)
            st.write("Vista previa de los datos:")
            st.dataframe(df_preview.head())
            
            if st.button("Realizar Predicción", type="primary"):
                with st.spinner("Generando predicciones..."):
                    try:
                        files = {"file": uploaded_file}
                        response = requests.post(
                            f"{API_URL}/predict/{sensor_type}/{size}",
                            files=files
                        )
                        
                        if response.status_code == 200:
                            results = response.json()
                            
                            # Crear DataFrame de resultados
                            df_results = pd.DataFrame(results['predictions'])
                            
                            # Mostrar resumen
                            st.subheader("Resumen de Predicciones")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Total de Muestras", 
                                    results['summary']['total_samples']
                                )
                            
                            with col2:
                                st.metric(
                                    "Predicciones Positivas", 
                                    f"{results['summary']['positive_predictions']} ({results['summary']['positive_percentage']:.2f}%)"
                                )
                            
                            with col3:
                                st.metric(
                                    "Confianza Media", 
                                    f"{results['summary']['mean_confidence']:.2f}"
                                )
                            
                            # Gráficos
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Distribución de predicciones
                                fig_pred = px.pie(
                                    df_results, 
                                    names='Prediction',
                                    title='Distribución de Predicciones'
                                )
                                st.plotly_chart(fig_pred)
                            
                            with col2:
                                # Histograma de confianza
                                fig_conf = px.histogram(
                                    df_results,
                                    x='Confidence',
                                    title='Distribución de Confianza'
                                )
                                st.plotly_chart(fig_conf)
                            
                            # Mostrar resultados detallados
                            st.subheader("Resultados Detallados")
                            st.dataframe(df_results)
                            
                            # Botón de descarga
                            csv = df_results.to_csv(index=False)
                            st.download_button(
                                label="📥 Descargar Predicciones",
                                data=csv,
                                file_name=f"predicciones_{sensor_type}_{size}.csv",
                                mime="text/csv"
                            )
                            
                            st.success(f"✅ Predicciones guardadas en: {results['output_file']}")
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()