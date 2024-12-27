# Model Prediction and Testing Pipeline

This document provides an overview of the system developed for predicting and testing machine learning models trained for binary classification tasks. The project focuses on automating processes such as training, testing, and generating predictions for new datasets.

## Project Structure

### Directories
- **model/**: Contains pre-trained models organized by sensor type (**ELF** and **MAG**) and data sizes (**160**, **320**, **480**).
- **data_predictions/**: Input directory for new datasets to be analyzed.
  - `data_predictions/ELF/160`
  - `data_predictions/ELF/320`
  - `data_predictions/ELF/480`
  - `data_predictions/MAG/160`
  - `data_predictions/MAG/320`
  - `data_predictions/MAG/480`
- **predictions/**: Output directory where the results of the predictions are saved. Organized similarly to `data_predictions` for ease of reference.
- **test/**: Input directory for test datasets used to validate the models during testing.

### Files
- **model_pipeline.py**: Handles data preprocessing, model training, and loading.
- **model_testing.py**: Tests the performance of the trained models using specified test datasets.
- **model_prediction.py**: Generates predictions for new datasets using the trained models.
- **requirements.txt**: Contains the list of Python dependencies required to run the project.

## Features

### 1. Automated Prediction Pipeline
The `model_prediction.py` script automates the process of generating predictions for new datasets:
- It identifies and processes new data from the `data_predictions` directory.
- Prepares the data for the models by cleaning unnecessary columns.
- Generates predictions using the appropriate model based on the sensor type and size of the data.
- Saves the predictions in the `predictions` directory in a structured manner.

### 2. Flexible Model Training and Testing
The `model_pipeline.py` and `model_testing.py` scripts ensure that:
- Models are trained and tested for different data sizes (**160**, **320**, **480**) and sensor types (**ELF** and **MAG**).
- Test datasets can be loaded and evaluated with detailed performance metrics such as F1-scores and classification reports.
- Cases with extreme values (e.g., all zeros or random noise) are tested to ensure model robustness.

## Steps to Use

### 1. Setup
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure the `model/` directory contains pre-trained models organized by sensor and size.
3. Prepare input datasets in the `data_predictions/` directory, maintaining the folder structure.

### 2. Running Predictions
Run the `model_prediction.py` script to generate predictions:
```bash
python model_prediction.py
```
- Predictions will be saved in the `predictions/` directory.
- Ensure the datasets in `data_predictions` match the required structure (index column, `SampleID`, and feature columns).

### 3. Testing Models
Run the `model_testing.py` script to validate the performance of the models:
```bash
python model_testing.py
```
- The script will evaluate the models on datasets from the `test/` directory.
- Results, including F1-scores and classification reports, are displayed in the terminal.

### 4. Retraining Models
Modify `model_pipeline.py` to add new training datasets and retrain models if necessary. Preprocessing and retraining scripts are already built into the pipeline.

## Key Metrics
- **F1-Score**: Measures the balance between precision and recall.
  - ELF 320: **0.9910**
  - MAG 320: **0.9968**
- **Robustness**: Models handle edge cases like zeros, ones, and random noise effectively.

## Summary
This project provides an end-to-end solution for training, testing, and predicting with machine learning models. The structured directories, automated scripts, and performance tracking ensure efficient handling of large datasets across different use cases.

