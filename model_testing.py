import os
import torch
import numpy as np
import pandas as pd
import traceback

from typing import Dict
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Ajusta si tu archivo real se llama distinto.
# Debe existir una clase "ModelPipeline" definida en model_pipeline.py
from model_pipeline import ModelPipeline


###############################################################################
# Función auxiliar: cargar data de test en subcarpetas test/ELF/<size>, ...
###############################################################################
def load_test_data_subfolders(base_dir='test'):
    """
    Estructura de carpetas:
      test/
       ├─ ELF/
       │   ├─ 160/ (varios .csv)
       │   ├─ 320/
       │   └─ 480/
       ├─ MAG/
       │   ├─ 160/ (varios .csv)
       │   ├─ 320/
       │   └─ 480/
    
    Retorna un dict:
      {
        160: (df_elf_160, df_mag_160),
        320: (df_elf_320, df_mag_320),
        480: (df_elf_480, df_mag_480)
      }
    """
    sizes = [160, 320, 480]
    data_dict = {}
    print("\n=== CARGANDO DATOS DE PRUEBA DESDE SUBCARPETAS ===")

    for size in sizes:
        elf_path = os.path.join(base_dir, 'ELF', str(size))
        mag_path = os.path.join(base_dir, 'MAG', str(size))

        # ELF
        df_elf_list = []
        if os.path.exists(elf_path):
            csv_elf = [f for f in os.listdir(elf_path) if f.endswith('.csv')]
            print(f"[ELF][{size}] {len(csv_elf)} CSV encontrados")
            for csv_file in csv_elf:
                full_fp = os.path.join(elf_path, csv_file)
                print(f"   Leyendo {full_fp}")
                tmp = pd.read_csv(full_fp)
                df_elf_list.append(tmp)
            df_elf = pd.concat(df_elf_list, ignore_index=True) if df_elf_list else None
        else:
            print(f"[ELF][{size}] Carpeta no existe: {elf_path}")
            df_elf = None

        # MAG
        df_mag_list = []
        if os.path.exists(mag_path):
            csv_mag = [f for f in os.listdir(mag_path) if f.endswith('.csv')]
            print(f"[MAG][{size}] {len(csv_mag)} CSV encontrados")
            for csv_file in csv_mag:
                full_fp = os.path.join(mag_path, csv_file)
                print(f"   Leyendo {full_fp}")
                tmp = pd.read_csv(full_fp)
                df_mag_list.append(tmp)
            df_mag = pd.concat(df_mag_list, ignore_index=True) if df_mag_list else None
        else:
            print(f"[MAG][{size}] Carpeta no existe: {mag_path}")
            df_mag = None

        data_dict[size] = (df_elf, df_mag)

    print("=== FIN CARGA DE DATOS DE PRUEBA ===\n")
    return data_dict


###############################################################################
# Clase ModelTester
###############################################################################
class ModelTester:
    def __init__(self, model_path='model'):
        # Instanciamos el pipeline y cargamos los modelos guardados
        self.pipeline = ModelPipeline()
        self.pipeline.load_models(model_path)
        # Diccionario para guardar resultados (f1, reporte, etc.)
        self.test_results: Dict[str, Dict] = {}

    def run_basic_tests(self, test_data_dir='test') -> bool:
        """
        Carga data de prueba desde subcarpetas test/ELF/160, test/MAG/160, etc.
        y evalúa cada modelo (sensor, tamaño).
        """
        print("\n=== Iniciando pruebas básicas ===")
        data_dict = load_test_data_subfolders(test_data_dir)

        for size, (df_elf, df_mag) in data_dict.items():
            print(f"\n--- TAMAÑO: {size} ---")

            # ELF
            if df_elf is not None and not df_elf.empty:
                print(f"[ELF][{size}] shape: {df_elf.shape}")
                X_elf, lengths_elf, y_elf = self.pipeline.preprocess_data(df_elf, is_training=False)
                results_elf = self.test_model(X_elf, lengths_elf, y_elf, 'elf', size)
                self.test_results[f'elf_{size}'] = results_elf
            else:
                print(f"[SKIP] ELF {size} no tiene datos")

            # MAG
            if df_mag is not None and not df_mag.empty:
                print(f"[MAG][{size}] shape: {df_mag.shape}")
                X_mag, lengths_mag, y_mag = self.pipeline.preprocess_data(df_mag, is_training=False)
                results_mag = self.test_model(X_mag, lengths_mag, y_mag, 'mag', size)
                self.test_results[f'mag_{size}'] = results_mag
            else:
                print(f"[SKIP] MAG {size} no tiene datos")

        # Al final, evaluamos resultados
        return self.evaluate_results()

    def test_model(self, X: torch.Tensor, lengths: torch.Tensor, 
                   y: torch.Tensor, sensor_type: str, size: int) -> Dict:
        """
        Aplica el modelo (sensor_type, size) y devuelve métricas.
        """
        print(f"-> Testeando modelo {sensor_type.upper()} size {size}")
        # Obtenemos el modelo que corresponde a (sensor_type, size)
        model = self.pipeline.models[sensor_type].get(size, None)
        if model is None:
            print(f"[ERROR] No existe modelo para {sensor_type.upper()} size {size}")
            return {}

        # Si es un modelo PyTorch (por ejemplo, un nn.Module), tendrá 'eval'.
        if hasattr(model, 'eval'):
            model.eval()

        with torch.no_grad():
            # Para un LSTM, etc., podríamos necesitar `X = X.unsqueeze(-1)`
            # Pero si es scikit-learn, no importa, no se usa.
            if X.dim() == 2:
                X = X.unsqueeze(-1)
            
            # RandomForest no usa GPU, pero si fuera PyTorch, lo haríamos:
            X_test = X.to(self.pipeline.device)
            lengths_dev = lengths.to(self.pipeline.device)

            # Para scikit-learn, no existe forward(...) con tensores, 
            # en cambio se usa model.predict(X_np). 
            # Pero si tu pipeline mezcla LSTM con scikit, hay que diferenciarlos.
            
            # Chequeamos si el modelo es scikit:
            if hasattr(model, 'predict'):
                # scikit-learn typical usage: 
                #   conviertele a numpy (sacando de GPU) 
                X_test_np = X_test.cpu().numpy()  # [N, features, 1]? 
                # Si tus datos en scikit son [N, features], quita la tercera dim.
                X_test_np = X_test_np.squeeze(-1)  # -> [N, features]
                predicted = model.predict(X_test_np)
            else:
                # Asumimos PyTorch model
                outputs = model(X_test, lengths_dev)
                _, predicted_torch = torch.max(outputs.data, 1)
                predicted = predicted_torch.cpu().numpy()

        y_true = y.cpu().numpy()

        return {
            'f1_score': f1_score(y_true, predicted, average='weighted'),
            'predictions': predicted,
            'true_values': y_true,
            'classification_report': classification_report(y_true, predicted)
        }

    def evaluate_results(self) -> bool:
        """
        Muestra F1-scores, matrices de confusión, etc., y devuelve True/False
        si todas las pruebas cumplen un umbral.
        """
        all_ok = True
        threshold = 0.85

        for key, results in self.test_results.items():
            if not results:
                # Si el diccionario está vacío, no hay resultados
                continue

            print(f"\n=== Resultados para {key} ===")
            f1 = results['f1_score']
            print(f"F1 Score: {f1:.4f}")
            if f1 < threshold:
                print(f"[FAIL] F1 < {threshold}")
                all_ok = False
            else:
                print(f"[OK] F1 >= {threshold}")
            
            print("\nReporte de Clasificación:")
            print(results['classification_report'])

            self.plot_confusion_matrix(results['true_values'], 
                                       results['predictions'], 
                                       key)

        return all_ok

    def test_extreme_cases(self, num_samples=50) -> bool:
        """
        Genera datos 'extremos' (zeros, ones, random) y los pasa por cada modelo.
        """
        print("\n=== Probando casos extremos ===")

        found_size = None
        for stype in ['elf','mag']:
            if self.pipeline.models[stype]:
                found_size = next(iter(self.pipeline.models[stype].keys()))
                break

        if found_size is None:
            print("[SKIP] No hay modelos cargados en pipeline.")
            return False

        scaler = self.pipeline.scalers['elf'].get(found_size, None)
        if scaler is None:
            print(f"[SKIP] No se encontró scaler en elf {found_size}")
            return False

        n_features = scaler.scale_.shape[0]
        print(f"[INFO] n_features = {n_features}")

        extremes = {
            'zeros': np.zeros((num_samples, n_features)),
            'ones': np.ones((num_samples, n_features)),
            'random': np.random.randn(num_samples, n_features)
        }

        all_good = True
        for case_name, data in extremes.items():
            print(f"\n[CASE] {case_name}")
            try:
                X = torch.FloatTensor(data)
                lengths = torch.LongTensor([X.size(1)] * X.size(0))

                for s_type in ['elf','mag']:
                    for sz, mdl in self.pipeline.models[s_type].items():
                        if hasattr(mdl, 'eval'):
                            mdl.eval()
                        with torch.no_grad():
                            X_dev = X.to(self.pipeline.device)
                            len_dev = lengths.to(self.pipeline.device)

                            # scikit or PyTorch check
                            if hasattr(mdl, 'predict'):  # scikit
                                X_np = X_dev.cpu().numpy()
                                # Solo aplícalo si de verdad la última dim es 1:
                                if X_np.ndim == 3 and X_np.shape[-1] == 1:
                                    X_np = X_np.squeeze(-1)
                                preds = mdl.predict(X_np)
                            else:
                                # PyTorch
                                out = mdl(X_dev, len_dev)
                                preds_t = torch.argmax(out, dim=1)
                                preds = preds_t.cpu().numpy()


                            print("   Distribución:", np.bincount(preds))

            except Exception as e:
                print(f"[FAIL] Error en {case_name}: {str(e)}")
                traceback.print_exc()
                all_good = False

        return all_good

    def plot_confusion_matrix(self, y_true, y_pred, key):
        """
        Genera y guarda la matriz de confusión en test_results/cm_<key>.png
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusión - {key}')
        plt.ylabel('Verdadero')
        plt.xlabel('Predicho')
        os.makedirs('test_results', exist_ok=True)
        plt.savefig(f'test_results/cm_{key}.png')
        plt.close()


def main():
    tester = ModelTester(model_path='model')  # Ajusta si tus modelos están en otra ruta
    print("=== Iniciando pruebas de modelo ===")

    # 1) Pruebas básicas con data real en subcarpetas test/
    basic_ok = tester.run_basic_tests('test')

    # 2) Pruebas de casos extremos
    extremes_ok = tester.test_extreme_cases()

    print("\n=== Resumen final de pruebas ===")
    print(f"Pruebas básicas:  {'OK' if basic_ok else 'FAIL'}")
    print(f"Casos extremos:   {'OK' if extremes_ok else 'FAIL'}")

    if basic_ok and extremes_ok:
        print("✅ Todas las pruebas han pasado exitosamente")
    else:
        print("❌ Hubo problemas en alguna(s) prueba(s)")


if __name__ == "__main__":
    main()
