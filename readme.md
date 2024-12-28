# Model Training and Prediction System for Sensor Data

## Overview
This system processes and analyzes sensor data (ELF and MAG) across different sizes (160, 320, 480) using machine learning models. It consists of three main components:
- Model training pipeline
- Model testing framework
- Prediction system for new data

## File Structure
```
project/
├── model/                  # Trained models storage
│   ├── ELF/
│   │   ├── 160/
│   │   ├── 320/
│   │   └── 480/
│   └── MAG/
│       ├── 160/
│       ├── 320/
│       └── 480/
├── data_predictions/       # Place new data here for predictions
│   ├── ELF/
│   │   ├── 160/
│   │   ├── 320/
│   │   └── 480/
│   └── MAG/
│       ├── 160/
│       ├── 320/
│       └── 480/
├── predictions/           # Output folder for predictions
├── model_pipeline.py
├── model_prediction.py
└── model_testing.py
```

## Main Components

### model_pipeline.py
- Core class for training models
- Handles data loading and preprocessing
- Trains Random Forest models for each sensor/size combination
- Manages model saving and loading

### model_prediction.py
- Used for making predictions on new data
- Takes input from data_predictions folder
- Generates predictions (0 or 1) for each case
- Saves results in predictions folder

### model_testing.py
- Used during development phase
- Validates model performance
- Not needed for regular usage

## How to Use

### For Predictions on New Data:
1. Place your new data files in the appropriate folders under `data_predictions/`
2. Run:
```python
python model_prediction.py
```
3. Find results in the `predictions/` folder

## Notes
- The system uses RobustScaler for data normalization
- Models are trained separately for each sensor type (ELF/MAG) and size (160/320/480)
- Predictions are binary (0 or 1)

## Requirements
- Python 3.x
- scikit-learn
- pandas
- numpy
- torch
- joblib

## Support
For any issues or questions, please contact the development team.