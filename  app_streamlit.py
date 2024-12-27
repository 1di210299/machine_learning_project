import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import numpy as np

st.set_page_config(page_title="Model Dashboard", layout="wide")

# URL de tu API FastAPI
API_URL = "http://localhost:8000"

def main():
    st.title("Dashboard de Predicciones")
    
    # Sidebar para selección
    st.sidebar.title("Configuración")
    sensor_type = st.sidebar.selectbox(
        "Seleccionar tipo de sensor",
        ["ELF", "MAG", "GEO"]
    )
    
    # Tabs principales
    tab1, tab2, tab3 = st.tabs(["Predicción Individual", "Métricas del Modelo", "Datos Históricos"])
    
    # Tab de Predicción Individual
    with tab1:
        st.header("Predicción Individual")
        
        # Entrada de datos
        input_data = st.text_input(
            "Ingrese los datos del sensor (separados por coma)",
            placeholder="1.2, 2.3, 3.4, ..."
        )
        
        if st.button("Realizar Predicción"):
            try:
                # Convertir entrada a lista de números
                data = [float(x.strip()) for x in input_data.split(",")]
                
                # Realizar predicción a través de la API
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"data": data, "sensor_type": sensor_type}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Mostrar resultados
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Predicción",
                            "Positivo" if result["prediction"] == 1 else "Negativo"
                        )
                    with col2:
                        st.metric(
                            "Probabilidad",
                            f"{result['probability']:.2%}"
                        )
                        
                    # Gráfico de confianza
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = result["probability"] * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 100], 'color': "gray"}
                            ]
                        }
                    ))
                    st.plotly_chart(fig)
                    
            except Exception as e:
                st.error(f"Error al procesar la predicción: {str(e)}")
    
    # Tab de Métricas del Modelo
    with tab2:
        st.header("Métricas del Modelo")
        
        try:
            # Leer métricas desde el archivo
            metrics_path = f"plots/{sensor_type}/metrics_{sensor_type}.txt"
            with open(metrics_path, 'r') as f:
                metrics_text = f.read()
            
            st.text(metrics_text)
            
            # Mostrar matriz de confusión
            st.image(
                f"plots/{sensor_type}/confusion_matrix_{sensor_type}.png",
                caption="Matriz de Confusión"
            )
            
        except Exception as e:
            st.error(f"Error al cargar métricas: {str(e)}")
    
    # Tab de Datos Históricos
    with tab3:
        st.header("Datos Históricos")
        try:
            # Cargar datos históricos si existen
            data_path = f"data/{sensor_type}/passages_export_{sensor_type}.csv"
            df = pd.read_csv(data_path)
            
            # Mostrar estadísticas básicas
            st.subheader("Estadísticas Básicas")
            st.dataframe(df.describe())
            
            # Visualizaciones
            st.subheader("Distribución de Datos")
            fig = px.histogram(df)
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"Error al cargar datos históricos: {str(e)}")

if __name__ == "__main__":
    main()