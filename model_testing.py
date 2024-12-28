import os
import torch
import numpy as np
import pandas as pd
import traceback

from typing import Dict
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Adjust if your actual file has a different name.
# A "ModelPipeline" class must be defined in model_pipeline.py
from model_pipeline import ModelPipeline


###############################################################################
# Helper function: load test data from subfolders test/ELF/<size>, ...
###############################################################################
def load_test_data_subfolders(base_dir='test'):
    """
    Folder structure:
      test/
       ├─ ELF/
       │   ├─ 160/ (multiple .csv)
       │   ├─ 320/
       │   └─ 480/
       ├─ MAG/
       │   ├─ 160/ (multiple .csv)
       │   ├─ 320/
       │   └─ 480/
    
    Returns a dict:
      {
        160: (df_elf_160, df_mag_160),
        320: (df_elf_320, df_mag_320),
        480: (df_elf_480, df_mag_480)
      }
    """
    sizes = [160, 320, 480]
    data_dict = {}
    print("\n=== LOADING TEST DATA FROM SUBFOLDERS ===")

    for size in sizes:
        elf_path = os.path.join(base_dir, 'ELF', str(size))
        mag_path = os.path.join(base_dir, 'MAG', str(size))

        # ELF
        df_elf_list = []
        if os.path.exists(elf_path):
            csv_elf = [f for f in os.listdir(elf_path) if f.endswith('.csv')]
            print(f"[ELF][{size}] {len(csv_elf)} CSV files found")
            for csv_file in csv_elf:
                full_fp = os.path.join(elf_path, csv_file)
                print(f"   Reading {full_fp}")
                tmp = pd.read_csv(full_fp)
                df_elf_list.append(tmp)
            df_elf = pd.concat(df_elf_list, ignore_index=True) if df_elf_list else None
        else:
            print(f"[ELF][{size}] Folder doesn't exist: {elf_path}")
            df_elf = None

        # MAG
        df_mag_list = []
        if os.path.exists(mag_path):
            csv_mag = [f for f in os.listdir(mag_path) if f.endswith('.csv')]
            print(f"[MAG][{size}] {len(csv_mag)} CSV files found")
            for csv_file in csv_mag:
                full_fp = os.path.join(mag_path, csv_file)
                print(f"   Reading {full_fp}")
                tmp = pd.read_csv(full_fp)
                df_mag_list.append(tmp)
            df_mag = pd.concat(df_mag_list, ignore_index=True) if df_mag_list else None
        else:
            print(f"[MAG][{size}] Folder doesn't exist: {mag_path}")
            df_mag = None

        data_dict[size] = (df_elf, df_mag)

    print("=== TEST DATA LOADING COMPLETE ===\n")
    return data_dict


###############################################################################
# ModelTester Class
###############################################################################
class ModelTester:
    def __init__(self, model_path='model'):
        # Instantiate pipeline and load saved models
        self.pipeline = ModelPipeline()
        self.pipeline.load_models(model_path)
        # Dictionary to store results (f1, report, etc.)
        self.test_results: Dict[str, Dict] = {}

    def run_basic_tests(self, test_data_dir='test') -> bool:
        """
        Load test data from subfolders test/ELF/160, test/MAG/160, etc.
        and evaluate each model (sensor, size).
        """
        print("\n=== Starting basic tests ===")
        data_dict = load_test_data_subfolders(test_data_dir)

        for size, (df_elf, df_mag) in data_dict.items():
            print(f"\n--- SIZE: {size} ---")

            # ELF
            if df_elf is not None and not df_elf.empty:
                print(f"[ELF][{size}] shape: {df_elf.shape}")
                X_elf, lengths_elf, y_elf = self.pipeline.preprocess_data(df_elf, is_training=False)
                results_elf = self.test_model(X_elf, lengths_elf, y_elf, 'elf', size)
                self.test_results[f'elf_{size}'] = results_elf
            else:
                print(f"[SKIP] ELF {size} has no data")

            # MAG
            if df_mag is not None and not df_mag.empty:
                print(f"[MAG][{size}] shape: {df_mag.shape}")
                X_mag, lengths_mag, y_mag = self.pipeline.preprocess_data(df_mag, is_training=False)
                results_mag = self.test_model(X_mag, lengths_mag, y_mag, 'mag', size)
                self.test_results[f'mag_{size}'] = results_mag
            else:
                print(f"[SKIP] MAG {size} has no data")

        # Finally, evaluate results
        return self.evaluate_results()

    def test_model(self, X: torch.Tensor, lengths: torch.Tensor, 
                   y: torch.Tensor, sensor_type: str, size: int) -> Dict:
        """
        Apply model (sensor_type, size) and return metrics.
        """
        print(f"-> Testing model {sensor_type.upper()} size {size}")
        # Get the model corresponding to (sensor_type, size)
        model = self.pipeline.models[sensor_type].get(size, None)
        if model is None:
            print(f"[ERROR] No model exists for {sensor_type.upper()} size {size}")
            return {}

        # If it's a PyTorch model (e.g., nn.Module), it will have 'eval'.
        if hasattr(model, 'eval'):
            model.eval()

        with torch.no_grad():
            # For LSTM, etc., we might need `X = X.unsqueeze(-1)`
            # But for scikit-learn, it doesn't matter, it's not used.
            if X.dim() == 2:
                X = X.unsqueeze(-1)
            
            # RandomForest doesn't use GPU, but if it were PyTorch, we would do:
            X_test = X.to(self.pipeline.device)
            lengths_dev = lengths.to(self.pipeline.device)

            # For scikit-learn, forward(...) with tensors doesn't exist,
            # instead model.predict(X_np) is used.
            # But if your pipeline mixes LSTM with scikit, we need to differentiate.
            
            # Check if the model is scikit:
            if hasattr(model, 'predict'):
                # scikit-learn typical usage:
                #   convert to numpy (removing from GPU)
                X_test_np = X_test.cpu().numpy()  # [N, features, 1]?
                # If your scikit data is [N, features], remove third dim.
                X_test_np = X_test_np.squeeze(-1)  # -> [N, features]
                predicted = model.predict(X_test_np)
            else:
                # Assume PyTorch model
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
        Display F1-scores, confusion matrices, etc., and return True/False
        if all tests meet a threshold.
        """
        all_ok = True
        threshold = 0.85

        for key, results in self.test_results.items():
            if not results:
                # If dictionary is empty, no results
                continue

            print(f"\n=== Results for {key} ===")
            f1 = results['f1_score']
            print(f"F1 Score: {f1:.4f}")
            if f1 < threshold:
                print(f"[FAIL] F1 < {threshold}")
                all_ok = False
            else:
                print(f"[OK] F1 >= {threshold}")
            
            print("\nClassification Report:")
            print(results['classification_report'])

            self.plot_confusion_matrix(results['true_values'], 
                                       results['predictions'], 
                                       key)

        return all_ok

    def test_extreme_cases(self, num_samples=50) -> bool:
        """
        Generate 'extreme' data (zeros, ones, random) and pass through each model.
        """
        print("\n=== Testing extreme cases ===")

        found_size = None
        for stype in ['elf','mag']:
            if self.pipeline.models[stype]:
                found_size = next(iter(self.pipeline.models[stype].keys()))
                break

        if found_size is None:
            print("[SKIP] No models loaded in pipeline.")
            return False

        scaler = self.pipeline.scalers['elf'].get(found_size, None)
        if scaler is None:
            print(f"[SKIP] No scaler found in elf {found_size}")
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
                                # Only apply if last dim is actually 1:
                                if X_np.ndim == 3 and X_np.shape[-1] == 1:
                                    X_np = X_np.squeeze(-1)
                                preds = mdl.predict(X_np)
                            else:
                                # PyTorch
                                out = mdl(X_dev, len_dev)
                                preds_t = torch.argmax(out, dim=1)
                                preds = preds_t.cpu().numpy()

                            print("   Distribution:", np.bincount(preds))

            except Exception as e:
                print(f"[FAIL] Error in {case_name}: {str(e)}")
                traceback.print_exc()
                all_good = False

        return all_good

    def plot_confusion_matrix(self, y_true, y_pred, key):
        """
        Generate and save confusion matrix in test_results/cm_<key>.png
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {key}')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        os.makedirs('test_results', exist_ok=True)
        plt.savefig(f'test_results/cm_{key}.png')
        plt.close()


def main():
    tester = ModelTester(model_path='model')  # Adjust if your models are in another path
    print("=== Starting model tests ===")

    # 1) Basic tests with real data in test/ subfolders
    basic_ok = tester.run_basic_tests('test')

    # 2) Extreme case tests
    extremes_ok = tester.test_extreme_cases()

    print("\n=== Final Test Summary ===")
    print(f"Basic tests:     {'OK' if basic_ok else 'FAIL'}")
    print(f"Extreme cases:   {'OK' if extremes_ok else 'FAIL'}")

    if basic_ok and extremes_ok:
        print("✅ All tests passed successfully")
    else:
        print("❌ There were problems in some test(s)")


if __name__ == "__main__":
    main()