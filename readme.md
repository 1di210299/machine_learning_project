# README: How to Run the Machine Learning Training Code

## Overview
This README file explains how to run the code to train and evaluate the machine learning models for ELF and MAG sensor datasets. Follow these instructions to prepare the environment, execute the training process, and evaluate the model's performance.

---

## Prerequisites
### Software Requirements
- Python 3.x installed on your system.
- Required libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `imbalanced-learn`
  - `matplotlib`
  - `seaborn`

### Hardware Recommendations
- At least 8 GB of RAM for handling large datasets.
- A multi-core processor for faster training.
- GPU support (optional) for faster processing when using large datasets.

### Input Data Requirements
- CSV file format.
- Columns:
  - **Features**: 320 columns representing sensor data.
  - **Label**: Binary column indicating event detection (1 for Yes, 0 for No).

---

## Installation
1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Set Up the Environment**
   - Create a virtual environment (optional but recommended):
     ```bash
     python -m venv env
     source env/bin/activate   # On Windows: env\Scripts\activate
     ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

---

## Running the Code
### 1. Prepare Your Dataset
- Place your CSV dataset in the `/data` folder (or update the file path in the script).
- Ensure your data matches the required structure.

### 2. Execute the Script
Run the main training script:
```bash
python train_model.py
```

### 3. Monitor the Output
- The script will:
  1. Load and preprocess the dataset.
  2. Balance the classes using SMOTEENN.
  3. Train and optimize the model.
  4. Evaluate the model and display metrics such as accuracy, precision, and recall.
  5. Save the trained model for future use.

---

## Outputs
1. **Model File**:
   - Saved in the `/models` directory as `trained_model.joblib`.

2. **Performance Reports**:
   - Classification report and confusion matrix displayed in the terminal.
   - Graphical outputs saved in the `/outputs` folder.

---

## Troubleshooting
1. **Missing Libraries**:
   - Ensure all required libraries are installed. Use:
     ```bash
     pip install -r requirements.txt
     ```

2. **Data Format Issues**:
   - Ensure the input CSV matches the specified format.
   - Use a data validation script if needed (can be provided on request).

3. **Memory Errors**:
   - Use a system with higher RAM or work with smaller batches of data.

---



