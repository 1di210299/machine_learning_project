import torch
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from model_pipeline import ModelPipeline
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os

class ModelTester:
    def __init__(self, model_path: str = 'model'):
        self.pipeline = ModelPipeline()
        self.pipeline.load_models(model_path)
        self.test_results: Dict[str, Dict] = {}
        
    def run_basic_tests(self, test_data_dir: str = 'test') -> bool:
        """Ejecuta pruebas básicas en el modelo"""
        print("\nEjecutando pruebas básicas...")
        
        # Cargar datos de prueba
        data_elf, data_mag = self.pipeline.load_data(test_data_dir)
        
        if data_elf is None or data_mag is None:
            print("❌ Error: No se pudieron cargar los datos de prueba")
            return False
            
        print(f"Datos cargados - ELF shape: {data_elf.shape}, MAG shape: {data_mag.shape}")
            
        # Procesar datos
        X_elf, lengths_elf, y_elf = self.pipeline.preprocess_data(data_elf, is_training=False)
        X_mag, lengths_mag, y_mag = self.pipeline.preprocess_data(data_mag, is_training=False)
        
        # Probar modelos
        results_elf = self.test_model(X_elf, lengths_elf, y_elf, 'elf')
        results_mag = self.test_model(X_mag, lengths_mag, y_mag, 'mag')
        
        self.test_results['elf'] = results_elf
        self.test_results['mag'] = results_mag
        
        return self.evaluate_results()
    
    def test_model(self, X: torch.Tensor, lengths: torch.Tensor, 
                  y: torch.Tensor, sensor_type: str) -> Dict:
        """Prueba un modelo específico y retorna métricas"""
        model = self.pipeline.models[sensor_type]
        model.eval()
        
        with torch.no_grad():
            if X.dim() == 2:
                X = X.unsqueeze(-1)
            X_test = X.to(self.pipeline.device)
            lengths = lengths.to(self.pipeline.device)
            outputs = model(X_test, lengths)
            _, predicted = torch.max(outputs.data, 1)
            
        y_pred = predicted.cpu().numpy()
        y_true = y.cpu().numpy()
        
        return {
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'predictions': y_pred,
            'true_values': y_true,
            'classification_report': classification_report(y_true, y_pred)
        }
        
    def evaluate_results(self) -> bool:
        """Evalúa los resultados de las pruebas"""
        all_tests_passed = True
        
        for sensor_type, results in self.test_results.items():
            print(f"\nResultados para {sensor_type.upper()}:")
            
            # Verificar F1-score
            f1 = results['f1_score']
            threshold = 0.85
            print(f"F1-Score: {f1:.4f}")
            if f1 < threshold:
                print(f"❌ F1-Score por debajo del umbral ({threshold})")
                all_tests_passed = False
            else:
                print(f"✅ F1-Score por encima del umbral")
            
            # Mostrar reporte de clasificación
            print("\nReporte de clasificación:")
            print(results['classification_report'])
            
            # Generar matriz de confusión
            self.plot_confusion_matrix(results['true_values'], 
                                    results['predictions'], 
                                    sensor_type)
        
        return all_tests_passed
    
    def test_extreme_cases(self, num_samples: int = 100) -> bool:
        """Prueba el modelo con casos extremos"""
        print("\nProbando casos extremos...")
        
        # Generar datos de prueba extremos
        extreme_cases = {
            'zeros': np.zeros((num_samples, 320)),
            'ones': np.ones((num_samples, 320)),
            'random': np.random.randn(num_samples, 320),
            'alternating': np.tile([0, 1], (num_samples, 160))
        }
        
        for case_name, data in extreme_cases.items():
            print(f"\nProbando caso: {case_name}")
            try:
                # Convertir a tensor
                X = torch.FloatTensor(data)
                lengths = torch.LongTensor([320] * num_samples)
                
                # Probar ambos modelos
                for sensor_type in ['elf', 'mag']:
                    with torch.no_grad():
                        X = X.to(self.pipeline.device)
                        lengths = lengths.to(self.pipeline.device)
                        outputs = self.pipeline.models[sensor_type](
                            X.unsqueeze(-1), 
                            lengths
                        )
                    print(f"✅ {sensor_type.upper()}: Predicción exitosa")
            except Exception as e:
                print(f"❌ Error en {case_name}: {str(e)}")
                return False
                
        return True
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            sensor_type: str):
        """Genera y guarda la matriz de confusión"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title(f'Matriz de Confusión - {sensor_type.upper()} (Test)')
        plt.ylabel('Verdadero')
        plt.xlabel('Predicho')
        
        # Crear directorio si no existe
        os.makedirs('test_results', exist_ok=True)
        plt.savefig(f'test_results/confusion_matrix_test_{sensor_type}.png')
        plt.close()

def main():
    # Crear instancia del tester
    tester = ModelTester()
    
    # Ejecutar pruebas básicas
    print("=== Iniciando pruebas de modelo ===")
    basic_tests_passed = tester.run_basic_tests()
    
    # Ejecutar pruebas de casos extremos
    extreme_tests_passed = tester.test_extreme_cases()
    
    # Reporte final
    print("\n=== Resumen de pruebas ===")
    print(f"Pruebas básicas: {'✅ Pasaron' if basic_tests_passed else '❌ Fallaron'}")
    print(f"Pruebas de casos extremos: {'✅ Pasaron' if extreme_tests_passed else '❌ Fallaron'}")
    
    if basic_tests_passed and extreme_tests_passed:
        print("\n✅ Todas las pruebas pasaron exitosamente")
    else:
        print("\n❌ Algunas pruebas fallaron")

if __name__ == "__main__":
    main()