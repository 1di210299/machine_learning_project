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
# Clase principal: ModelPipeline
###############################################################################
class ModelPipeline:
    def __init__(self):
        """
        Aquí almacenamos en diccionarios los modelos y scalers para
        cada (sensor, size), por ejemplo:
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
        # Para el caso de que uses un modelo con PyTorch (LSTM, etc.), 
        # definimos un device. Si solo usas sklearn, esto no es esencial.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ############################################################################
    # ------------------------- Cargar datos de subcarpetas --------------------
    ############################################################################
    def load_data_subfolders(self, base_dir='data'):
        """
        Estructura esperada:
          data/
           ├─ ELF/
           │   ├─ 160/ (varios .csv)
           │   ├─ 320/
           │   └─ 480/
           ├─ MAG/
           │   ├─ 160/
           │   ├─ 320/
           │   └─ 480/

        Retorna un dict:
          {
            160: (df_elf_160, df_mag_160),
            320: (df_elf_320, df_mag_320),
            480: (df_elf_480, df_mag_480)
          }

        Cada DataFrame resultante es la concatenación de todos los .csv 
        que encuentre en esas subcarpetas.
        """
        sizes = [160, 320, 480]
        data_dict = {}

        print("\n=== CARGANDO DATOS DESDE SUBCARPETAS ===")
        for size in sizes:
            elf_path = os.path.join(base_dir, 'ELF', str(size))
            mag_path = os.path.join(base_dir, 'MAG', str(size))

            # ELF
            elf_dfs = []
            if os.path.exists(elf_path):
                csv_elf = [f for f in os.listdir(elf_path) if f.endswith('.csv')]
                print(f"[ELF][{size}] CSV encontrados: {len(csv_elf)}")
                for csv_file in csv_elf:
                    full_path = os.path.join(elf_path, csv_file)
                    print(f"   Leyendo {full_path}")
                    df_tmp = pd.read_csv(full_path)
                    elf_dfs.append(df_tmp)
                df_elf = pd.concat(elf_dfs, ignore_index=True) if elf_dfs else None
            else:
                print(f"[ELF][{size}] Carpeta no existe: {elf_path}")
                df_elf = None

            # MAG
            mag_dfs = []
            if os.path.exists(mag_path):
                csv_mag = [f for f in os.listdir(mag_path) if f.endswith('.csv')]
                print(f"[MAG][{size}] CSV encontrados: {len(csv_mag)}")
                for csv_file in csv_mag:
                    full_path = os.path.join(mag_path, csv_file)
                    print(f"   Leyendo {full_path}")
                    df_tmp = pd.read_csv(full_path)
                    mag_dfs.append(df_tmp)
                df_mag = pd.concat(mag_dfs, ignore_index=True) if mag_dfs else None
            else:
                print(f"[MAG][{size}] Carpeta no existe: {mag_path}")
                df_mag = None

            data_dict[size] = (df_elf, df_mag)

        print("=== FIN CARGA DATOS ===\n")
        return data_dict
    
    ############################################################################
    # --------------------------- Entrenar y guardar ---------------------------
    ############################################################################
    def train_and_save_models(self, data_dir='data', model_out='model'):
        """
        1) Carga datos desde subcarpetas (ELF/160, MAG/160, etc.).
        2) Entrena un RandomForest por cada (sensor, size).
        3) Guarda los modelos y scalers en model_out/ELF/160, etc.
        """
        data_dict = self.load_data_subfolders(data_dir)

        for size, (df_elf, df_mag) in data_dict.items():
            print(f"\n--- PROCESANDO TAMAÑO: {size} ---")
            # Entrenar para ELF
            self.train_model(df_elf, 'elf', size)
            # Entrenar para MAG
            self.train_model(df_mag, 'mag', size)
        
        # Guardar
        self.save_models(model_out)
    
    def train_model(self, df, sensor_type, size):
        """
        Entrena un modelo (RandomForest) para (sensor_type, size).
        """
        if df is None or df.empty:
            print(f"[SKIP] Sin datos para {sensor_type.upper()} size {size}")
            return

        print(f"\n=== ENTRENANDO {sensor_type.upper()} (size {size}) ===")
        print(f"Forma del DataFrame: {df.shape}")

        # Verificar que exista la columna Label
        if 'Label' not in df.columns:
            print(f"[ERROR] Falta la columna 'Label' en {sensor_type.upper()} size {size}")
            return

        # Features y Label
        X = df.drop(columns=['Label','SampleID'], errors='ignore')
        y = df['Label']

        if y.nunique() < 2:
            print(f"[SKIP] Solo {y.nunique()} clase(s) en {sensor_type.upper()} size {size}")
            return

        # Escalar con RobustScaler
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[sensor_type][size] = scaler

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        print(" - Entrenando RandomForestClassifier ...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluación rápida
        y_pred = model.predict(X_test)
        print(">>> EVALUACIÓN <<<")
        print(classification_report(y_test, y_pred))
        print("Matriz de confusión:")
        print(confusion_matrix(y_test, y_pred))

        # Guardar en diccionario
        self.models[sensor_type][size] = model
    
    ############################################################################
    # ------------------- Guardado y Carga de Modelos/Scalers ------------------
    ############################################################################
    def save_models(self, base_dir='model'):
        """
        Guarda cada modelo y scaler en subcarpetas:
          model/ELF/160, model/ELF/320, ...
          model/MAG/160, model/MAG/320, ...
        """
        print("\n=== GUARDANDO MODELOS Y SCALERS ===")
        for sensor_type in ['elf', 'mag']:
            for size, model in self.models[sensor_type].items():
                if model is None:
                    continue
                # Crear carpeta
                folder = os.path.join(base_dir, sensor_type.upper(), str(size))
                os.makedirs(folder, exist_ok=True)

                model_path = os.path.join(folder, f"model_{sensor_type}_{size}.pkl")
                scaler_path = os.path.join(folder, f"scaler_{sensor_type}_{size}.pkl")

                # Guardar modelo
                joblib.dump(model, model_path)

                # Guardar scaler
                scaler_obj = self.scalers[sensor_type][size]
                joblib.dump(scaler_obj, scaler_path)

                print(f"[OK] {sensor_type.upper()} size {size}:")
                print(f"   Modelo :  {model_path}")
                print(f"   Scaler :  {scaler_path}")

        print("=== FIN GUARDADO ===\n")

    def load_models(self, base_dir='model'):
        """
        Carga cada modelo y scaler desde las subcarpetas
        model/ELF/160, model/ELF/320, 480, etc.
        """
        if not os.path.exists(base_dir):
            print(f"No existe la carpeta {base_dir}. No se cargaron modelos.")
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
                        continue  # si no es un entero, lo ignoramos

                    model_path = os.path.join(sub_path, f"model_{sensor_type}_{size}.pkl")
                    scaler_path = os.path.join(sub_path, f"scaler_{sensor_type}_{size}.pkl")

                    if os.path.exists(model_path) and os.path.exists(scaler_path):
                        print(f"Cargando {sensor_type.upper()} size {size} desde {sub_path}")
                        loaded_model = joblib.load(model_path)
                        loaded_scaler = joblib.load(scaler_path)
                        self.models[sensor_type][size] = loaded_model
                        self.scalers[sensor_type][size] = loaded_scaler
    
    ############################################################################
    # ------------------ Preprocesar datos en fase de test ---------------------
    ############################################################################
    def preprocess_data(self, data: pd.DataFrame, is_training: bool = True):
        """
        Si 'Label' está en columns, la separamos de X. 
        Hacemos un escalado "temporal" con RobustScaler para la demo 
        (en un caso real, usarías self.scalers[sensor][size] si sabes el size).
        
        Retorna (X_torch, lengths, y_torch) si hay Label, sino (X_torch, lengths).
        """
        import torch
        from sklearn.preprocessing import RobustScaler

        if 'Label' in data.columns:
            X = data.drop(columns=['Label','SampleID'], errors='ignore')
            y = data['Label']
        else:
            X = data
            y = None

        # Para fines de demostración, escalamos con un nuevo scaler.
        # Idealmente, si supiéramos sensor & size, haríamos:
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
# Si quieres un main de ejemplo:
###############################################################################
def main():
    mp = ModelPipeline()
    # Entrenamos y guardamos
    mp.train_and_save_models(data_dir='data', model_out='model')
    # Cargar (para comprobar)
    mp.load_models('model')

if __name__ == "__main__":
    main()
