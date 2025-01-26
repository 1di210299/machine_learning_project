import streamlit as st
import requests
import pandas as pd
import os
import plotly.express as px

# Initial configuration
st.set_page_config(
    page_title="Sensor Prediction System",
    layout="wide"
)

# Custom CSS styles
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 20px; }
    </style>
""", unsafe_allow_html=True)

# API configuration
API_URL = "http://localhost:8000"
BASE_PATH = '/Users/juandiegogutierrezcortez/machine_learning_project'
DATA_PATH = os.path.join(BASE_PATH, 'data')
DATA_PREDICTIONS_PATH = os.path.join(BASE_PATH, 'data_predictions')

def main():
    st.title("🤖 Sensor Prediction System")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        sensor_type = st.selectbox("Sensor Type", ['ELF', 'MAG', 'GEO'])
        size = st.selectbox("Size", [160, 320, 480])
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Training", "Evaluation", "Prediction"])
    
    # Training tab
    with tab1:
        st.header("📊 Model Training")
        
        # Display current training data info
        st.subheader("Current Training Data")
        data_files = {}
        for current_sensor in ['ELF', 'MAG', 'GEO']:
            data_files[current_sensor] = {}
            for current_size in [160, 320, 480]:
                path = os.path.join(DATA_PATH, current_sensor, str(current_size))
                if os.path.exists(path):
                    files = [f for f in os.listdir(path) if f.endswith('.csv')]
                    data_files[current_sensor][current_size] = len(files)
        
        for sensor_name, sizes in data_files.items():
            with st.expander(f"{sensor_name} Data"):
                for size_value, count in sizes.items():
                    st.write(f"Size {size_value}: {count} files")
        
        # Upload new training data
        st.subheader("Upload New Training Data")
        uploaded_files = st.file_uploader(
            "Upload CSV files",
            type=['csv'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            # En app.py, modificamos la parte del botón "Process and Train"
            if st.button("Process and Train", type="primary"):
                with st.spinner(f"Training {sensor_type}/{size} model with all available data..."):
                    try:
                        # Si hay nuevos archivos, primero los guardamos
                        if uploaded_files:
                            for file in uploaded_files:
                                try:
                                    file.seek(0)
                                    files_dict = {"file": (file.name, file.getvalue(), "text/csv")}
                                    response = requests.post(
                                        f"{API_URL}/upload-training/{sensor_type}/{size}",
                                        files=files_dict
                                    )
                                    if response.status_code == 200:
                                        st.success(f"✅ {file.name} uploaded successfully")
                                    else:
                                        st.error(f"❌ Error uploading {file.name}")
                                except Exception as e:
                                    st.error(f"Upload error: {str(e)}")

                        # Entrenar con todos los archivos de la carpeta seleccionada
                        training_path = os.path.join(DATA_PATH, sensor_type, str(size))
                        available_files = [f for f in os.listdir(training_path) if f.endswith('.csv')]
                        
                        if available_files:
                            st.info(f"Training {sensor_type}/{size} with {len(available_files)} files from {training_path}")
                            
                            train_response = requests.post(
                                f"{API_URL}/train/{sensor_type}",
                                json={"size": size}
                            )
                            
                            if train_response.status_code == 200:
                                result = train_response.json()
                                st.success(f"✅ Model trained successfully with {result['files_used']} files")
                                st.success(f"Total samples used: {result['total_samples']}")
                                st.json(result)
                            else:
                                st.error("❌ Training failed")
                        else:
                            st.warning(f"No CSV files found in {training_path}")
                            
                    except Exception as e:
                        st.error(f"Training error: {str(e)}")
    
    # Evaluation tab
    with tab2:
        st.header("🔍 Model Evaluation")
        
        if st.button("Evaluate Model", type="primary"):
            with st.spinner("Running evaluation..."):
                try:
                    response = requests.post(
                        f"{API_URL}/evaluate/{sensor_type}/{size}"
                    )
                    if response.status_code == 200:
                        results = response.json()
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Accuracy", f"{results['accuracy']:.2f}")
                        with col2:
                            st.metric("Precision", f"{results['precision']:.2f}")
                        with col3:
                            st.metric("Recall", f"{results['recall']:.2f}")
                        
                        # Display detailed results
                        st.subheader("Detailed Results")
                        st.json(results['detailed_results'])
                except Exception as e:
                    st.error(f"Evaluation error: {str(e)}")
    
    # Prediction tab
    with tab3:
        st.header("🎯 Prediction")
        
        uploaded_file = st.file_uploader(
            "Upload file for prediction",
            type=['csv']
        )
        
        if uploaded_file is not None:
            try:
                # Mostrar información del archivo
                st.write("File Info:")
                st.write(f"- Name: {uploaded_file.name}")
                st.write(f"- Size: {len(uploaded_file.getvalue())} bytes")
                
                # Preview del contenido
                df_preview = pd.read_csv(uploaded_file)
                st.write("Data preview:")
                st.dataframe(df_preview.head())
                
                if st.button("Generate Predictions", type="primary"):
                    with st.spinner("Processing predictions..."):
                        try:
                            # Resetear el puntero del archivo
                            uploaded_file.seek(0)
                            
                            # Preparar el archivo para envío
                            files = {
                                "file": (
                                    uploaded_file.name,
                                    uploaded_file.getvalue(),
                                    "text/csv"
                                )
                            }
                            
                            # Realizar la petición
                            response = requests.post(
                                f"{API_URL}/predict/{sensor_type}/{size}",
                                files=files
                            )
                            
                            if response.status_code == 200:
                                results = response.json()
                                df_results = pd.DataFrame(results['predictions'])
                                
                                # Display summary
                                st.subheader("Results Summary")
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric(
                                        "Total Samples",
                                        results['summary']['total_samples']
                                    )
                                
                                with col2:
                                    st.metric(
                                        "Positive Predictions",
                                        f"{results['summary']['positive_predictions']} ({results['summary']['positive_percentage']:.2f}%)"
                                    )
                                
                                with col3:
                                    st.metric(
                                        "Mean Confidence",
                                        f"{results['summary']['mean_confidence']:.2f}"
                                    )
                                
                                # Display charts
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    fig_pred = px.pie(
                                        df_results,
                                        names='Prediction',
                                        title='Predictions Distribution'
                                    )
                                    st.plotly_chart(fig_pred)
                                
                                with col2:
                                    fig_conf = px.histogram(
                                        df_results,
                                        x='Confidence',
                                        title='Confidence Distribution'
                                    )
                                    st.plotly_chart(fig_conf)
                                
                                # Display detailed results
                                st.subheader("Detailed Results")
                                st.dataframe(df_results)
                                
                                # Download results
                                csv = df_results.to_csv(index=False)
                                st.download_button(
                                    "📥 Download Results",
                                    csv,
                                    f"predictions_{sensor_type}_{size}.csv",
                                    "text/csv"
                                )
                                
                            else:
                                st.error(f"Error: {response.text}")
                        except Exception as e:
                            st.error(f"Prediction error: {str(e)}")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

if __name__ == "__main__":
    main()