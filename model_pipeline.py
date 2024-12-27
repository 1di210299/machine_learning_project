import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import joblib
import os
from typing import Tuple, Optional

class VariableLengthLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(VariableLengthLSTM, self).__init__()
        
        # Detectar dimensiones automáticamente
        self.num_features = input_size
        
        # Normalización y reducción de dimensionalidad
        self.batch_norm = nn.BatchNorm1d(self.num_features)
        self.feature_reducer = nn.Linear(self.num_features, self.num_features//2)
        
        # LSTM optimizado
        self.lstm = nn.LSTM(
            self.num_features//2,  # Entrada reducida
            hidden_size, 
            num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )
        
        # Capas fully connected mejoradas
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x, lengths):
        # Asegurarnos que x tiene las dimensiones correctas
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        # Reshape para batch norm
        x = x.view(batch_size, -1)  # Aplanar para BatchNorm1d
        x = self.batch_norm(x)
        x = x.view(batch_size, seq_length, -1)  # Restaurar forma
        
        # Reducción de dimensionalidad
        x = self.feature_reducer(x)
        
        # LSTM
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm(packed_input)
        
        # Obtener último estado oculto
        last_hidden = torch.cat((hn[-2], hn[-1]), dim=1)
        out = self.fc_layers(last_hidden)
        return out

class ModelPipeline:
    def __init__(self):
        self.scaler = RobustScaler()
        self.models = {
            'elf': None,
            'mag': None
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_weights = None
        
    def calculate_class_weights(self, y):
        class_counts = torch.bincount(y)
        total = len(y)
        weights = total / (2.0 * class_counts)
        return weights

    def load_data(self, data_dir: str = 'data') -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        try:
            elf_files = os.listdir(os.path.join(data_dir, 'ELF'))
            elf_data = []
            for file in elf_files:
                if file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(data_dir, 'ELF', file))
                    elf_data.append(df)
            data_elf = pd.concat(elf_data, ignore_index=True) if elf_data else None

            mag_files = os.listdir(os.path.join(data_dir, 'MAG'))
            mag_data = []
            for file in mag_files:
                if file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(data_dir, 'MAG', file))
                    mag_data.append(df)
            data_mag = pd.concat(mag_data, ignore_index=True) if mag_data else None

            if data_elf is not None and data_mag is not None:
                print(f"Data loaded - ELF shape: {data_elf.shape}, MAG shape: {data_mag.shape}")
            return data_elf, data_mag
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None
        
    def preprocess_data(self, data: pd.DataFrame, is_training: bool = True):
        if 'Label' in data.columns:
            X = data.drop(['Label', 'SampleID'] if 'SampleID' in data.columns else ['Label'], axis=1)
            y = data['Label']
            
            if is_training:
                smote = SMOTE(
                    random_state=42,
                    k_neighbors=3,
                    sampling_strategy=0.5
                )
                X_resampled, y_resampled = smote.fit_resample(X, y)
                X = pd.DataFrame(X_resampled, columns=X.columns)
                y = pd.Series(y_resampled)
        else:
            X = data
            y = None

        X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
        
        # Convertir a tensor con dimensiones correctas
        tensor_data = torch.FloatTensor(X.values)
        tensor_data = tensor_data.unsqueeze(1)  # Agregar dimensión de secuencia
        
        lengths = torch.LongTensor([tensor_data.size(1)] * tensor_data.size(0))
        
        if y is not None:
            y = torch.LongTensor(y.values)
            return tensor_data, lengths, y
        return tensor_data, lengths

    def train_model(self, X: torch.Tensor, lengths: torch.Tensor, y: torch.Tensor, 
                sensor_type: str, epochs=150, batch_size=32):
        if sensor_type == 'elf':
            hidden_size = 256
            num_layers = 4
            class_weights = torch.FloatTensor([1.0, 8.0]).to(self.device)
        else:
            hidden_size = 128
            num_layers = 3
            class_weights = self.calculate_class_weights(y).to(self.device)
        
        num_classes = len(torch.unique(y))
        num_features = X.size(-1)
        
        print(f"Input features: {num_features}")
        print(f"Batch size: {X.size(0)}")
        print(f"Sequence length: {X.size(1)}")
        
        model = VariableLengthLSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.0003 if sensor_type == 'elf' else 0.001,
            weight_decay=0.05 if sensor_type == 'elf' else 0.01
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2 if sensor_type == 'elf' else 0.5,
            patience=5 if sensor_type == 'elf' else 10,
            verbose=True
        )
        
        # Crear directorio para guardar modelos
        save_dir = os.path.join('model', sensor_type.upper())
        os.makedirs(save_dir, exist_ok=True)
        best_model_path = os.path.join(save_dir, f'model_{sensor_type}.pth')
        
        best_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        print(f"\nTraining {sensor_type} model...")
        for epoch in tqdm(range(epochs)):
            model.train()
            optimizer.zero_grad()
            
            X_reshaped = X.to(self.device)
            y = y.to(self.device)
            
            outputs = model(X_reshaped, lengths)
            loss = criterion(outputs, y)
            
            loss.backward()
            optimizer.step()
            
            scheduler.step(loss)
            
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        model.load_state_dict(torch.load(best_model_path))
        self.models[sensor_type] = model
        return model

    def evaluate_model(self, X_test: torch.Tensor, lengths_test: torch.Tensor, 
                      y_test: torch.Tensor, sensor_type: str):
        model = self.models[sensor_type]
        model.eval()
        
        with torch.no_grad():
            X_test = X_test.unsqueeze(-1).to(self.device)
            outputs = model(X_test, lengths_test)
            _, predicted = torch.max(outputs.data, 1)
            
        y_pred = predicted.cpu().numpy()
        y_true = y_test.cpu().numpy()
        
        print(f"\nEvaluation results for {sensor_type}:")
        print(classification_report(y_true, y_pred))
        
        plots_dir = os.path.join('train_plots', sensor_type.upper())
        os.makedirs(plots_dir, exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title(f'Confusion Matrix - {sensor_type}')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(plots_dir, f'confusion_matrix_{sensor_type}.png'))
        plt.close()

    def save_models(self, base_dir: str = 'model'):
        for sensor_type in ['elf', 'mag']:
            if self.models[sensor_type] is not None:
                save_dir = os.path.join(base_dir, sensor_type.upper())
                os.makedirs(save_dir, exist_ok=True)
                torch.save(self.models[sensor_type].state_dict(), 
                        os.path.join(save_dir, f'model_{sensor_type}.pth'))

    def load_models(self, base_dir: str = 'model'):
        for sensor_type in ['elf', 'mag']:
            model_path = os.path.join(base_dir, sensor_type.upper(), f'model_{sensor_type}.pth')
            if os.path.exists(model_path):
                input_size = 1
                hidden_size = 256 if sensor_type == 'elf' else 128
                num_layers = 4 if sensor_type == 'elf' else 3
                num_classes = 2
                
                self.models[sensor_type] = VariableLengthLSTM(
                    input_size, hidden_size, num_layers, num_classes).to(self.device)
                self.models[sensor_type].load_state_dict(
                    torch.load(model_path, map_location=self.device))

def main():
    pipeline = ModelPipeline()
    
    with tqdm(total=6, desc="Pipeline Progress") as pbar:
        data_elf, data_mag = pipeline.load_data('data')
        pbar.update(1)

        if data_elf is None or data_mag is None:
            print("Error loading data. Exiting...")
            return

        X_elf, lengths_elf, y_elf = pipeline.preprocess_data(data_elf)
        X_mag, lengths_mag, y_mag = pipeline.preprocess_data(data_mag)
        pbar.update(1)

        X_train_elf, X_test_elf, y_train_elf, y_test_elf = train_test_split(
            X_elf, y_elf, test_size=0.2, random_state=42, stratify=y_elf)
        lengths_train_elf, lengths_test_elf, _, _ = train_test_split(
            lengths_elf, lengths_elf, test_size=0.2, random_state=42)

        X_train_mag, X_test_mag, y_train_mag, y_test_mag = train_test_split(
            X_mag, y_mag, test_size=0.2, random_state=42, stratify=y_mag)
        lengths_train_mag, lengths_test_mag, _, _ = train_test_split(
            lengths_mag, lengths_mag, test_size=0.2, random_state=42)
        pbar.update(1)

        pipeline.train_model(X_train_elf, lengths_train_elf, y_train_elf, 'elf')
        pbar.update(1)
        
        pipeline.train_model(X_train_mag, lengths_train_mag, y_train_mag, 'mag')
        pbar.update(1)

        pipeline.evaluate_model(X_test_elf, lengths_test_elf, y_test_elf, 'elf')
        pipeline.evaluate_model(X_test_mag, lengths_test_mag, y_test_mag, 'mag')
        pipeline.save_models('model')
        pbar.update(1)

if __name__ == "__main__":
    main()