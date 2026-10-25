import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from itertools import product
import os
import json
import warnings
from datetime import datetime
import math
from typing import Tuple, List, Optional, Dict

warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

# Base paths
base_results_path = "../../../results/"
models_base_path = base_results_path + "models/SA_Enhanced_BiLSTM/"
plots_base_path = base_results_path + "plots/SA_Enhanced_BiLSTM/"
metrics_base_path = base_results_path + "metrics/SA_Enhanced_BiLSTM/"


class ResidualBiLSTMLayer(nn.Module):
    """BiLSTM Layer with Residual Connection"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 dropout: float = 0.2):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_size * 2)  # *2 for bidirectional

        # Projection for residual connection if input_size != output_size
        self.residual_projection = None
        if input_size != hidden_size * 2:
            self.residual_projection = nn.Linear(input_size, hidden_size * 2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            output: (batch_size, seq_len, hidden_size * 2)
            (hidden, cell): Tuple of final states
        """
        residual = x

        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Apply layer normalization
        lstm_out = self.layer_norm(lstm_out)

        # Apply dropout
        lstm_out = self.dropout(lstm_out)

        # Residual connection
        if self.residual_projection is not None:
            residual = self.residual_projection(residual)

        # Add residual (make sure dimensions match)
        if residual.size(-1) == lstm_out.size(-1):
            output = lstm_out + residual
        else:
            # If dimensions don't match, just use LSTM output
            output = lstm_out

        return output, (hidden, cell)


class EnhancedBiLSTM(nn.Module):
    """
    Enhanced BiLSTM with Residual Connections and Time Feature Processing
    """

    def __init__(self,
                 sequence_length: int = 12,
                 target_features: int = 1,
                 time_features: int = 5,  # hour, day, month, dayofweek, is_weekend
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout_rate: float = 0.2,
                 fc_units: List[int] = [64, 32, 16],
                 use_residual: bool = True):
        super().__init__()

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_residual = use_residual

        # Input sizes
        self.target_features = target_features
        self.time_features = time_features
        self.total_features = target_features + time_features

        # Time feature embedding layers
        self.hour_embedding = nn.Embedding(24, 8)  # 24 hours
        self.day_embedding = nn.Embedding(31, 8)  # 31 days
        self.month_embedding = nn.Embedding(12, 8)  # 12 months
        self.weekday_embedding = nn.Embedding(7, 8)  # 7 days of week

        # Time feature MLP for additional processing
        self.time_mlp = nn.Sequential(
            nn.Linear(time_features * 8, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Project time features to match target feature dimension
        self.time_projection = nn.Linear(16, target_features)

        # Stacked BiLSTM layers with residual connections
        self.lstm_layers = nn.ModuleList()
        current_input_size = self.total_features

        for layer_idx in range(num_layers):
            lstm_layer = ResidualBiLSTMLayer(
                input_size=current_input_size if layer_idx == 0 else hidden_size * 2,
                hidden_size=hidden_size,
                num_layers=1,  # Each layer is single-layer for residual connections
                dropout=dropout_rate
            )
            self.lstm_layers.append(lstm_layer)

        # Attention layer to weight important timesteps
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.softmax = nn.Softmax(dim=1)

        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        input_fc_size = hidden_size * 2  # Bidirectional

        for fc_size in fc_units:
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(input_fc_size, fc_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.LayerNorm(fc_size)
                )
            )
            input_fc_size = fc_size

        # Final output layer
        self.output_layer = nn.Linear(input_fc_size, 1)

        # Layer normalization for input
        self.input_norm = nn.LayerNorm(self.total_features)

    def extract_time_features(self, time_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract and embed time features

        Args:
            time_tensor: (batch_size, seq_length, time_features)
                        Expected features: [hour, day, month, dayofweek, is_weekend]
        Returns:
            embedded_time: (batch_size, seq_length, target_features)
        """
        batch_size, seq_length, _ = time_tensor.shape

        # Extract individual time features
        hour = time_tensor[..., 0].long() % 24
        day = time_tensor[..., 1].long() % 31
        month = (time_tensor[..., 2].long() % 12) + 1
        weekday = time_tensor[..., 3].long() % 7
        is_weekend = time_tensor[..., 4].float().unsqueeze(-1)

        # Get embeddings
        hour_emb = self.hour_embedding(hour)  # (batch_size, seq_length, 8)
        day_emb = self.day_embedding(day)  # (batch_size, seq_length, 8)
        month_emb = self.month_embedding(month)  # (batch_size, seq_length, 8)
        weekday_emb = self.weekday_embedding(weekday)  # (batch_size, seq_length, 8)

        # Concatenate embeddings
        time_embeddings = torch.cat([
            hour_emb, day_emb, month_emb, weekday_emb
        ], dim=-1)  # (batch_size, seq_length, 32)

        # Process through MLP
        time_features = self.time_mlp(time_embeddings)  # (batch_size, seq_length, 16)

        # Project to target feature dimension
        projected_time = self.time_projection(time_features)  # (batch_size, seq_length, target_features)

        # Combine with weekend feature
        time_features = projected_time + is_weekend

        return time_features

    def forward(self, x: torch.Tensor, time_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with time features

        Args:
            x: (batch_size, seq_length, target_features) - Target time series
            time_features: (batch_size, seq_length, time_features) - Time features
        Returns:
            output: (batch_size, 1) - Prediction
        """
        batch_size, seq_length, _ = x.shape

        # Process time features if provided
        if time_features is not None:
            processed_time = self.extract_time_features(time_features)
            # Combine with input features
            x = torch.cat([x, processed_time], dim=-1)

        # Apply layer normalization to input
        x = self.input_norm(x)

        # Pass through stacked BiLSTM layers with residual connections
        lstm_output = x
        for lstm_layer in self.lstm_layers:
            lstm_output, (hidden, cell) = lstm_layer(lstm_output)

        # Attention mechanism
        attention_scores = self.attention(lstm_output)  # (batch_size, seq_length, 1)
        attention_weights = self.softmax(attention_scores)  # (batch_size, seq_length, 1)

        # Weighted sum of LSTM outputs
        weighted_output = torch.sum(attention_weights * lstm_output, dim=1)  # (batch_size, hidden_size * 2)

        # Pass through fully connected layers
        fc_output = weighted_output
        for fc_layer in self.fc_layers:
            fc_output = fc_layer(fc_output)

        # Final output
        output = self.output_layer(fc_output)

        return output


class EnhancedBiLSTMPredictor:
    def __init__(self, sequence_length=12, target_col='Residential_Usage'):
        self.sequence_length = sequence_length
        self.target_col = target_col
        self.target_scaler = StandardScaler()
        self.time_scaler = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.create_directories()
        print(f"Using device: {self.device}, Sequence length: {sequence_length}")

    def create_directories(self):
        """Create directories for saving results"""
        self.models_path = os.path.join(models_base_path, f"seq_{self.sequence_length}")
        self.plots_path = os.path.join(plots_base_path, f"seq_{self.sequence_length}")
        self.metrics_path = os.path.join(metrics_base_path, f"seq_{self.sequence_length}")

        for path in [self.models_path, self.plots_path, self.metrics_path]:
            os.makedirs(path, exist_ok=True)

    def extract_time_features(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """
        Extract time-based features from dates

        Returns array with columns: [hour, day, month, dayofweek, is_weekend]
        """
        features = []

        for date in dates:
            # Extract various time features
            hour = date.hour
            day = date.day
            month = date.month
            dayofweek = date.dayofweek  # Monday=0, Sunday=6
            is_weekend = 1 if dayofweek >= 5 else 0

            features.append([hour, day, month, dayofweek, is_weekend])

        return np.array(features)

    def prepare_sequences(self, data: np.ndarray, time_features: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """Create sequences for time series prediction with time features"""
        sequences = []
        time_sequences = []
        targets = []

        for i in range(len(data) - self.sequence_length):
            # Target sequence
            seq = data[i:i + self.sequence_length]
            target = data[i + self.sequence_length]

            # Time features for the sequence
            time_seq = time_features[i:i + self.sequence_length]

            sequences.append(seq)
            time_sequences.append(time_seq)
            targets.append(target)

        return np.array(sequences), np.array(time_sequences), np.array(targets)

    def prepare_data(self, data_frame: pd.DataFrame) -> Tuple[Tuple, Tuple, Tuple]:
        """Prepare data for training with time features"""
        # Extract target data
        target_data = data_frame[self.target_col].values.reshape(-1, 1)

        # Scale target data
        scaled_target = self.target_scaler.fit_transform(target_data).flatten()

        # Extract and scale time features
        time_features = self.extract_time_features(data_frame.index)
        scaled_time = self.time_scaler.fit_transform(time_features)

        # Create sequences
        X, X_time, y = self.prepare_sequences(scaled_target, scaled_time)

        # Split data: 80% train, 10% validation, 10% test
        train_size = int(len(X) * 0.8)
        val_size = int(len(X) * 0.9)

        X_train, y_train = X[:train_size], y[:train_size]
        X_time_train = X_time[:train_size]

        X_val, y_val = X[train_size:val_size], y[train_size:val_size]
        X_time_val = X_time[train_size:val_size]

        X_test, y_test = X[val_size:], y[val_size:]
        X_time_test = X_time[val_size:]

        # Convert to tensors
        X_train = torch.FloatTensor(X_train).unsqueeze(-1)  # (samples, seq_length, 1)
        X_time_train = torch.FloatTensor(X_time_train)
        y_train = torch.FloatTensor(y_train).unsqueeze(-1)

        X_val = torch.FloatTensor(X_val).unsqueeze(-1)
        X_time_val = torch.FloatTensor(X_time_val)
        y_val = torch.FloatTensor(y_val).unsqueeze(-1)

        X_test = torch.FloatTensor(X_test).unsqueeze(-1)
        X_time_test = torch.FloatTensor(X_time_test)
        y_test = torch.FloatTensor(y_test).unsqueeze(-1)

        return (X_train, X_time_train, y_train), (X_val, X_time_val, y_val), (X_test, X_time_test, y_test)

    def train_model(self, X_train, X_time_train, y_train,
                    X_val, X_time_val, y_val, params, epochs=100, patience=15):
        """Train the enhanced BiLSTM model"""
        print(f"\nTraining Enhanced BiLSTM with parameters: {params}")

        # Initialize model
        self.model = EnhancedBiLSTM(
            sequence_length=self.sequence_length,
            target_features=1,
            time_features=5,  # hour, day, month, dayofweek, is_weekend
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout_rate=params['dropout_rate'],
            fc_units=params['fc_units'],
            use_residual=True
        ).to(self.device)

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=params['learning_rate'])

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # Move data to device
        X_train = X_train.to(self.device)
        X_time_train = X_time_train.to(self.device)
        y_train = y_train.to(self.device)

        X_val = X_val.to(self.device)
        X_time_val = X_time_val.to(self.device)
        y_val = y_val.to(self.device)

        # Create DataLoader for training
        train_dataset = TensorDataset(X_train, X_time_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=0
        )

        train_losses = []
        val_losses = []
        learning_rates = []

        print("\nStarting training...")

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0

            for batch_X, batch_time, batch_y in train_loader:
                optimizer.zero_grad()

                # Forward pass with time features
                predictions = self.model(batch_X, batch_time)
                loss = criterion(predictions, batch_y)

                # Backward pass
                loss.backward()

                # Gradient clipping for LSTM stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val, X_time_val)
                val_loss = criterion(val_predictions, y_val).item()
                val_losses.append(val_loss)

            # Update learning rate
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()

                # Save best model
                model_filename = f'best_EnhancedBiLSTM_model_{self.target_col}_seq{self.sequence_length}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'params': params
                }, os.path.join(self.models_path, model_filename))

                print(f"Epoch {epoch + 1}/{epochs} - New best model saved! "
                      f"Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return best_val_loss, train_losses, val_losses, learning_rates

    def predict(self, X, X_time=None):
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            if X_time is not None:
                X_time = X_time.to(self.device)
            predictions = self.model(X, X_time)
        return predictions.cpu().numpy()

    def inverse_transform_target(self, data):
        """Inverse transform scaled target data"""
        return self.target_scaler.inverse_transform(data.reshape(-1, 1)).flatten()

    def grid_search(self, data_frame, params_grid, epochs=50):
        """Perform grid search for best parameters"""
        (X_train, X_time_train, y_train), (X_val, X_time_val, y_val), _ = self.prepare_data(data_frame)

        param_names = list(params_grid.keys())
        param_values = list(params_grid.values())
        param_combinations = list(product(*param_values))

        best_score = float('inf')
        best_params = None
        results = []

        print(f"\nStarting grid search with {len(param_combinations)} combinations...")

        for i, combination in enumerate(param_combinations):
            params = dict(zip(param_names, combination))
            print(f"\nTesting combination {i + 1}/{len(param_combinations)}: {params}")

            # Train model with current parameters
            val_loss, train_losses, val_losses, learning_rates = self.train_model(
                X_train, X_time_train, y_train,
                X_val, X_time_val, y_val,
                params, epochs=epochs
            )

            results.append({
                'params': params,
                'val_loss': val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'learning_rates': learning_rates
            })

            if val_loss < best_score:
                best_score = val_loss
                best_params = params
                print(f"New best validation loss: {val_loss:.6f}")

        print(f"\nGrid search complete!")
        print(f"Best validation loss: {best_score:.6f}")
        print(f"Best parameters: {best_params}")

        self.save_grid_search_results(results, best_params)

        return best_params, results

    def save_grid_search_results(self, results, best_params):
        """Save grid search results to file"""
        serializable_results = []
        for result in results:
            serializable_result = {
                'params': result['params'],
                'val_loss': float(result['val_loss']),
                'train_losses': [float(loss) for loss in result['train_losses']],
                'val_losses': [float(loss) for loss in result['val_losses']],
                'learning_rates': [float(lr) for lr in result['learning_rates']]
            }
            serializable_results.append(serializable_result)

        results_data = {
            'sequence_length': self.sequence_length,
            'target_column': self.target_col,
            'best_params': best_params,
            'all_results': serializable_results,
            'timestamp': datetime.now().isoformat()
        }

        filename = f'grid_search_results_{self.target_col}_seq{self.sequence_length}.json'
        with open(os.path.join(self.metrics_path, filename), 'w') as f:
            json.dump(results_data, f, indent=4)

        print(f"Grid search results saved to {filename}")

    def plot_results(self, df, best_params, epochs=100):
        """Train with the best parameters and plot results"""
        # Prepare data
        (X_train, X_time_train, y_train), (X_val, X_time_val, y_val), (X_test, X_time_test, y_test) = self.prepare_data(
            df)

        # Combine train and validation for final training
        X_train_full = torch.cat([X_train, X_val], dim=0)
        X_time_train_full = torch.cat([X_time_train, X_time_val], dim=0)
        y_train_full = torch.cat([y_train, y_val], dim=0)

        print(f"\nTraining final Enhanced BiLSTM model for {self.target_col}...")

        # Train final model
        final_val_loss, train_losses, val_losses, learning_rates = self.train_model(
            X_train_full, X_time_train_full, y_train_full,
            X_val, X_time_val, y_val,
            best_params, epochs=epochs
        )

        # Make predictions
        test_predictions_scaled = self.predict(X_test, X_time_test)
        test_actual_scaled = y_test.numpy()

        # Inverse transform
        test_predictions = self.inverse_transform_target(test_predictions_scaled)
        test_actual = self.inverse_transform_target(test_actual_scaled)

        # Calculate metrics
        mse = np.mean((test_predictions - test_actual) ** 2)
        mae = np.mean(np.abs(test_predictions - test_actual))
        rmse = np.sqrt(mse)

        ss_res = np.sum((test_actual - test_predictions) ** 2)
        ss_tot = np.sum((test_actual - np.mean(test_actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        mask = test_actual != 0
        if np.any(mask):
            mape = np.mean(np.abs((test_actual[mask] - test_predictions[mask]) / test_actual[mask])) * 100
        else:
            mape = np.nan
            print("Warning: All actual values are zero, MAPE cannot be calculated")

        print(f"\nTest Set Metrics for {self.target_col} (Seq Len: {self.sequence_length}):")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.6f}")
        print(f"MAPE: {mape:.2f}%")

        # Create comprehensive plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Time series prediction plot
        time_points = range(len(test_predictions))
        axes[0, 0].plot(time_points, test_actual, 'b-', label='Actual', alpha=0.8, linewidth=2)
        axes[0, 0].plot(time_points, test_predictions, 'r--', label='Predicted', alpha=0.8, linewidth=2)
        axes[0, 0].fill_between(time_points, test_actual, test_predictions,
                                alpha=0.2, color='gray', label='Error')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel(f'{self.target_col} Value')
        axes[0, 0].set_title(f'Test Set Predictions\nSequence Length: {self.sequence_length}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Scatter plot
        axes[0, 1].scatter(test_actual, test_predictions, alpha=0.6, edgecolors='k', s=50)
        min_val = min(test_actual.min(), test_predictions.min())
        max_val = max(test_actual.max(), test_predictions.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        axes[0, 1].set_xlabel('Actual Values')
        axes[0, 1].set_ylabel('Predicted Values')
        axes[0, 1].set_title(f'Actual vs Predicted\nR² = {r2:.6f}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Error distribution
        errors = test_predictions - test_actual
        axes[1, 0].hist(errors, bins=30, edgecolor='black', alpha=0.7, density=True)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        axes[1, 0].axvline(x=np.mean(errors), color='g', linestyle='-',
                           linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
        axes[1, 0].set_xlabel('Prediction Error')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title(f'Error Distribution\nStd: {np.std(errors):.2f}, MAE: {mae:.2f}')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Training history
        epochs_range = range(1, len(train_losses) + 1)
        axes[1, 1].plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[1, 1].plot(epochs_range, val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss (MSE)')
        axes[1, 1].set_title('Training History')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')

        # Secondary axis for learning rate
        ax2 = axes[1, 1].twinx()
        ax2.plot(epochs_range, learning_rates, 'g--', label='Learning Rate', linewidth=1)
        ax2.set_ylabel('Learning Rate')
        ax2.set_yscale('log')
        ax2.legend(loc='upper right')

        plt.suptitle(
            f'Enhanced BiLSTM with Residual Connections\n{self.target_col} Prediction (Seq Len: {self.sequence_length})',
            fontsize=16, y=1.02)
        plt.tight_layout()

        plot_filename = f'EnhancedBiLSTM_results_{self.target_col}_seq{self.sequence_length}.png'
        plt.savefig(os.path.join(self.plots_path, plot_filename), dpi=150, bbox_inches='tight')
        plt.show()

        return test_predictions, test_actual, mse, mae, rmse, r2, mape

    def save_final_metrics(self, metrics_dict):
        """Save final metrics to CSV"""
        metrics_df = pd.DataFrame([metrics_dict])
        filename = f'final_metrics_{self.target_col}_seq{self.sequence_length}.csv'
        metrics_df.to_csv(os.path.join(self.metrics_path, filename), index=False)
        print(f"Final metrics saved to {filename}")


def run_enhanced_bilstm_experiment(df, sequence_length, param_grid, epochs=50):
    """Run complete Enhanced BiLSTM experiment"""
    print(f"\n{'=' * 80}")
    print(f"ENHANCED BI-LSTM EXPERIMENT - SEQUENCE LENGTH: {sequence_length}")
    print(f"{'=' * 80}")

    all_results = {}

    # Predict Residential Usage
    print(f"\n{'=' * 60}")
    print(f"PREDICTING RESIDENTIAL USAGE (Seq Len: {sequence_length})")
    print(f"{'=' * 60}")

    residential_predictor = EnhancedBiLSTMPredictor(
        sequence_length=sequence_length,
        target_col='Residential_Usage'
    )

    best_params_res, results_res = residential_predictor.grid_search(
        df, param_grid, epochs=epochs
    )

    res_predictions, res_actual, res_mse, res_mae, res_rmse, res_r2, res_mape = residential_predictor.plot_results(
        df, best_params_res, epochs=epochs
    )

    res_metrics = {
        'model': 'EnhancedBiLSTM',
        'target': 'Residential_Usage',
        'sequence_length': sequence_length,
        'mse': float(res_mse),
        'mae': float(res_mae),
        'rmse': float(res_rmse),
        'r2': float(res_r2),
        'mape': float(res_mape) if not np.isnan(res_mape) else None,
        'best_params': str(best_params_res),
        'timestamp': datetime.now().isoformat()
    }
    residential_predictor.save_final_metrics(res_metrics)

    # Predict Commercial Usage
    print(f"\n{'=' * 60}")
    print(f"PREDICTING COMMERCIAL USAGE (Seq Len: {sequence_length})")
    print(f"{'=' * 60}")

    commercial_predictor = EnhancedBiLSTMPredictor(
        sequence_length=sequence_length,
        target_col='Commercial_Usage'
    )

    best_params_com, results_com = commercial_predictor.grid_search(
        df, param_grid, epochs=epochs
    )

    com_predictions, com_actual, com_mse, com_mae, com_rmse, com_r2, com_mape = commercial_predictor.plot_results(
        df, best_params_com, epochs=epochs
    )

    com_metrics = {
        'model': 'EnhancedBiLSTM',
        'target': 'Commercial_Usage',
        'sequence_length': sequence_length,
        'mse': float(com_mse),
        'mae': float(com_mae),
        'rmse': float(com_rmse),
        'r2': float(com_r2),
        'mape': float(com_mape) if not np.isnan(com_mape) else None,
        'best_params': str(best_params_com),
        'timestamp': datetime.now().isoformat()
    }
    commercial_predictor.save_final_metrics(com_metrics)

    print(f"\n{'=' * 60}")
    print(f"ENHANCED BI-LSTM SUMMARY (Seq Len: {sequence_length})")
    print(f"{'=' * 60}")
    print(f"{'Metric':<15} {'Residential':<15} {'Commercial':<15}")
    print(f"{'-' * 45}")
    print(f"{'MSE':<15} {res_mse:<15.4f} {com_mse:<15.4f}")
    print(f"{'MAE':<15} {res_mae:<15.4f} {com_mae:<15.4f}")
    print(f"{'RMSE':<15} {res_rmse:<15.4f} {com_rmse:<15.4f}")
    print(f"{'R^2':<15} {res_r2:<15.6f} {com_r2:<15.6f}")
    print(f"{'MAPE':<15} {res_mape:<15.2f} {com_mape:<15.2f}")

    all_results[sequence_length] = {
        'residential': res_metrics,
        'commercial': com_metrics
    }

    return all_results


def create_enhanced_bilstm_summary(all_results):
    """Create summary comparison across sequence lengths"""
    print(f"\n{'=' * 80}")
    print("ENHANCED BI-LSTM COMPARISON ACROSS SEQUENCE LENGTHS")
    print(f"{'=' * 80}")

    summary_data = []

    for seq_len, results in all_results.items():
        summary_data.append({
            'Model': 'EnhancedBiLSTM',
            'Sequence_Length': seq_len,
            'Target': 'Residential',
            'MSE': results['residential']['mse'],
            'MAE': results['residential']['mae'],
            'RMSE': results['residential']['rmse'],
            'R2': results['residential']['r2'],
            'MAPE': results['residential']['mape']
        })

        summary_data.append({
            'Model': 'EnhancedBiLSTM',
            'Sequence_Length': seq_len,
            'Target': 'Commercial',
            'MSE': results['commercial']['mse'],
            'MAE': results['commercial']['mae'],
            'RMSE': results['commercial']['rmse'],
            'R2': results['commercial']['r2'],
            'MAPE': results['commercial']['mape']
        })

    summary_df = pd.DataFrame(summary_data)

    summary_path = os.path.join(metrics_base_path, 'EnhancedBiLSTM_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    return summary_df


if __name__ == "__main__":
    # Create directories
    for base_path in [models_base_path, plots_base_path, metrics_base_path]:
        os.makedirs(base_path, exist_ok=True)

    print("Loading processed data...")
    df = pd.read_csv("../../data/processed/dataset.csv", index_col='Date', parse_dates=True)
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    # Enhanced BiLSTM parameter grid
    enhanced_bilstm_param_grid = {
        'hidden_size': [32, 64, 128],
        'num_layers': [1, 2, 3],
        'dropout_rate': [0.1, 0.2, 0.3],
        'fc_units': [[64, 32], [128, 64], [128, 64, 32]],
        'learning_rate': [0.001, 0.005, 0.01],
        'batch_size': [32, 64, 128]
    }

    # Test with fewer combinations first
    test_param_grid = {
        'hidden_size': [64],
        'num_layers': [2],
        'dropout_rate': [0.2],
        'fc_units': [[64, 32]],
        'learning_rate': [0.001],
        'batch_size': [64]
    }

    sequence_lengths = [12, 24, 36]  # Test different sequence lengths
    TOTAL_EPOCHS = 50

    all_results = {}

    for seq_len in sequence_lengths:
        seq_results = run_enhanced_bilstm_experiment(
            df, seq_len, test_param_grid, epochs=TOTAL_EPOCHS
        )
        all_results.update(seq_results)

    summary_df = create_enhanced_bilstm_summary(all_results)