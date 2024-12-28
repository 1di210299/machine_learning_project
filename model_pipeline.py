import os
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix

import torch

###############################################################################
# Main Class: ModelPipeline
###############################################################################
class ModelPipeline:
    def __init__(self):
        """
        Here we store models and scalers in dictionaries for
        each (sensor, size) pair, for example:
          self.models['elf'][160]
          self.models['mag'][320]
          etc.
        """
        self.models = {
            'elf': {},
            'mag': {}
        }
        self.scalers = {
            'elf': {},
            'mag': {}
        }
        # For PyTorch models (LSTM, etc.), we define a device.
        # If you're only using sklearn, this isn't essential.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ############################################################################
    # ------------------------- Load data from subfolders ----------------------
    ############################################################################
    def load_data_subfolders(self, base_dir='data'):
        """
        Expected structure:
          data/
           ├─ ELF/
           │   ├─ 160/ (multiple .csv files)
           │   ├─ 320/
           │   └─ 480/
           ├─ MAG/
           │   ├─ 160/
           │   ├─ 320/
           │   └─ 480/

        Returns a dict:
          {
            160: (df_elf_160, df_mag_160),
            320: (df_elf_320, df_mag_320),
            480: (df_elf_480, df_mag_480)
          }

        Each resulting DataFrame is the concatenation of all .csv files 
        found in those subfolders.
        """
        sizes = [160, 320, 480]
        data_dict = {}

        print("\n=== LOADING DATA FROM SUBFOLDERS ===")
        for size in sizes:
            elf_path = os.path.join(base_dir, 'ELF', str(size))
            mag_path = os.path.join(base_dir, 'MAG', str(size))

            # ELF
            elf_dfs = []
            if os.path.exists(elf_path):
                csv_elf = [f for f in os.listdir(elf_path) if f.endswith('.csv')]
                print(f"[ELF][{size}] CSV files found: {len(csv_elf)}")
                for csv_file in csv_elf:
                    full_path = os.path.join(elf_path, csv_file)
                    print(f"   Reading {full_path}")
                    df_tmp = pd.read_csv(full_path)
                    elf_dfs.append(df_tmp)
                df_elf = pd.concat(elf_dfs, ignore_index=True) if elf_dfs else None
            else:
                print(f"[ELF][{size}] Folder doesn't exist: {elf_path}")
                df_elf = None

            # MAG
            mag_dfs = []
            if os.path.exists(mag_path):
                csv_mag = [f for f in os.listdir(mag_path) if f.endswith('.csv')]
                print(f"[MAG][{size}] CSV files found: {len(csv_mag)}")
                for csv_file in csv_mag:
                    full_path = os.path.join(mag_path, csv_file)
                    print(f"   Reading {full_path}")
                    df_tmp = pd.read_csv(full_path)
                    mag_dfs.append(df_tmp)
                df_mag = pd.concat(mag_dfs, ignore_index=True) if mag_dfs else None
            else:
                print(f"[MAG][{size}] Folder doesn't exist: {mag_path}")
                df_mag = None

            data_dict[size] = (df_elf, df_mag)

        print("=== DATA LOADING COMPLETE ===\n")
        return data_dict
    
    ############################################################################
    # --------------------------- Train and save -------------------------------
    ############################################################################
    def train_and_save_models(self, data_dir='data', model_out='model'):
        """
        1) Loads data from subfolders (ELF/160, MAG/160, etc.).
        2) Trains a RandomForest for each (sensor, size).
        3) Saves models and scalers in model_out/ELF/160, etc.
        """
        data_dict = self.load_data_subfolders(data_dir)

        for size, (df_elf, df_mag) in data_dict.items():
            print(f"\n--- PROCESSING SIZE: {size} ---")
            # Train for ELF
            self.train_model(df_elf, 'elf', size)
            # Train for MAG
            self.train_model(df_mag, 'mag', size)
        
        # Save
        self.save_models(model_out)
    
    def train_model(self, df, sensor_type, size):
        """
        Trains a model (RandomForest) for (sensor_type, size).
        """
        if df is None or df.empty:
            print(f"[SKIP] No data for {sensor_type.upper()} size {size}")
            return

        print(f"\n=== TRAINING {sensor_type.upper()} (size {size}) ===")
        print(f"DataFrame shape: {df.shape}")

        # Verify Label column exists
        if 'Label' not in df.columns:
            print(f"[ERROR] Missing 'Label' column in {sensor_type.upper()} size {size}")
            return

        # Features and Label
        X = df.drop(columns=['Label','SampleID'], errors='ignore')
        y = df['Label']

        if y.nunique() < 2:
            print(f"[SKIP] Only {y.nunique()} class(es) in {sensor_type.upper()} size {size}")
            return

        # Scale with RobustScaler
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[sensor_type][size] = scaler

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        print(" - Training RandomForestClassifier ...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Quick evaluation
        y_pred = model.predict(X_test)
        print(">>> EVALUATION <<<")
        print(classification_report(y_test, y_pred))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))

        # Save in dictionary
        self.models[sensor_type][size] = model
    
    ############################################################################
    # ------------------- Save and Load Models/Scalers -------------------------
    ############################################################################
    def save_models(self, base_dir='model'):
        """
        Saves each model and scaler in subfolders:
          model/ELF/160, model/ELF/320, ...
          model/MAG/160, model/MAG/320, ...
        """
        print("\n=== SAVING MODELS AND SCALERS ===")
        for sensor_type in ['elf', 'mag']:
            for size, model in self.models[sensor_type].items():
                if model is None:
                    continue
                # Create folder
                folder = os.path.join(base_dir, sensor_type.upper(), str(size))
                os.makedirs(folder, exist_ok=True)

                model_path = os.path.join(folder, f"model_{sensor_type}_{size}.pkl")
                scaler_path = os.path.join(folder, f"scaler_{sensor_type}_{size}.pkl")

                # Save model
                joblib.dump(model, model_path)

                # Save scaler
                scaler_obj = self.scalers[sensor_type][size]
                joblib.dump(scaler_obj, scaler_path)

                print(f"[OK] {sensor_type.upper()} size {size}:")
                print(f"   Model  :  {model_path}")
                print(f"   Scaler :  {scaler_path}")

        print("=== SAVING COMPLETE ===\n")

    def load_models(self, base_dir='model'):
        """
        Loads each model and scaler from subfolders
        model/ELF/160, model/ELF/320, 480, etc.
        """
        if not os.path.exists(base_dir):
            print(f"Directory {base_dir} doesn't exist. No models loaded.")
            return

        for sensor_type in ['elf','mag']:
            sensor_folder = os.path.join(base_dir, sensor_type.upper())
            if not os.path.exists(sensor_folder):
                continue

            for size_str in os.listdir(sensor_folder):
                sub_path = os.path.join(sensor_folder, size_str)
                if os.path.isdir(sub_path):
                    try:
                        size = int(size_str)
                    except ValueError:
                        continue  # skip if not an integer

                    model_path = os.path.join(sub_path, f"model_{sensor_type}_{size}.pkl")
                    scaler_path = os.path.join(sub_path, f"scaler_{sensor_type}_{size}.pkl")

                    if os.path.exists(model_path) and os.path.exists(scaler_path):
                        print(f"Loading {sensor_type.upper()} size {size} from {sub_path}")
                        loaded_model = joblib.load(model_path)
                        loaded_scaler = joblib.load(scaler_path)
                        self.models[sensor_type][size] = loaded_model
                        self.scalers[sensor_type][size] = loaded_scaler
    
    ############################################################################
    # ------------------ Preprocess data in test phase ------------------------
    ############################################################################
    def preprocess_data(self, data: pd.DataFrame, is_training: bool = True):
        """
        If 'Label' is in columns, we separate it from X.
        We do a "temporary" scaling with RobustScaler for demo purposes
        (in a real case, you'd use self.scalers[sensor][size] if you know the size).
        
        Returns (X_torch, lengths, y_torch) if Label exists, else (X_torch, lengths).
        """
        import torch
        from sklearn.preprocessing import RobustScaler

        if 'Label' in data.columns:
            X = data.drop(columns=['Label','SampleID'], errors='ignore')
            y = data['Label']
        else:
            X = data
            y = None

        # For demonstration purposes, we scale with a new scaler.
        # Ideally, if we knew sensor & size, we would do:
        # scaler = self.scalers[sensor][size]
        # X_scaled = scaler.transform(X)
        X_scaled = RobustScaler().fit_transform(X)

        X_torch = torch.FloatTensor(X_scaled)  # [N, features]
        lengths = torch.LongTensor([X_torch.size(1)] * X_torch.size(0))

        if y is not None:
            y_torch = torch.LongTensor(y.values)
            return X_torch, lengths, y_torch
        else:
            return X_torch, lengths


###############################################################################
# Example main if needed:
###############################################################################
def main():
    mp = ModelPipeline()
    # Train and save
    mp.train_and_save_models(data_dir='data', model_out='model')
    # Load (to verify)
    mp.load_models('model')

if __name__ == "__main__":
    main()