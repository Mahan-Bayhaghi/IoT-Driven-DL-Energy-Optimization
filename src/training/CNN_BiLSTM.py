import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from itertools import product
import os
import json
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

# Base paths
base_results_path = "../../results/"
models_base_path = base_results_path + "models/CNN_BiLSTM/"
plots_base_path = base_results_path + "plots/CNN_BiLSTM/"
metrics_base_path = base_results_path + "metrics/CNN_BiLSTM/"


class CNNBiLSTM(nn.Module):
    """
    Hybrid CNN-BiLSTM model for time series prediction
    CNN extracts spatial features, BiLSTM captures temporal dependencies
    """

    def __init__(self, input_size=1, seq_length=12,
                 conv_filters=32, kernel_size=3,
                 lstm_hidden_size=64, lstm_num_layers=2,
                 dropout_rate=0.2, fc1_units=32, fc2_units=16):
        super(CNNBiLSTM, self).__init__()

        self.seq_length = seq_length

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size,
                      out_channels=conv_filters,
                      kernel_size=kernel_size,
                      padding='same'),
            nn.BatchNorm1d(conv_filters),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Conv1d(in_channels=conv_filters,
                      out_channels=conv_filters * 2,
                      kernel_size=kernel_size,
                      padding='same'),
            nn.BatchNorm1d(conv_filters * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Conv1d(in_channels=conv_filters * 2,
                      out_channels=conv_filters * 4,
                      kernel_size=kernel_size,
                      padding='same'),
            nn.BatchNorm1d(conv_filters * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        cnn_output_size = conv_filters * 4  # After CNN, we have (batch_size, features, seq_length)

        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if lstm_num_layers > 1 else 0
        )

        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.Tanh(),
            nn.Linear(lstm_hidden_size, 1)
        )

        self.fc1 = nn.Linear(lstm_hidden_size * 2, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)  # output: a single prediction

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_features=False):
        # x shape: (batch_size, seq_length, input_size)

        # CNN expects (batch_size, input_size, seq_length)
        x_cnn = x.transpose(1, 2)

        cnn_features = self.cnn(x_cnn)  # (batch_size, cnn_output_size, seq_length)

        # Prepare for LSTM: (batch_size, seq_length, features)
        cnn_features = cnn_features.transpose(1, 2)

        lstm_out, (hidden, cell) = self.lstm(cnn_features)

        # Option 1: Use attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        weighted_sum = torch.sum(attention_weights * lstm_out, dim=1)

        # Option 2: Use last time step (simpler, often works well)
        # weighted_sum = lstm_out[:, -1, :]

        x = self.relu(self.fc1(weighted_sum))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        if return_features:
            return x, cnn_features.detach(), lstm_out.detach()
        return x


class CNNBiLSTMPredictor:
    def __init__(self, sequence_length=12, target_col='Residential_Usage'):
        self.sequence_length = sequence_length
        self.target_col = target_col
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.create_directories()
        print(f"Using device: {self.device}, Sequence length: {sequence_length}")

        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': []
        }

    def create_directories(self):
        """Create directories for saving results"""
        self.models_path = os.path.join(models_base_path, f"seq_{self.sequence_length}")
        self.plots_path = os.path.join(plots_base_path, f"seq_{self.sequence_length}")
        self.metrics_path = os.path.join(metrics_base_path, f"seq_{self.sequence_length}")

        for path in [self.models_path, self.plots_path, self.metrics_path]:
            os.makedirs(path, exist_ok=True)

    def prepare_sequences(self, data):
        """Create sequences for time series prediction"""
        sequences = []
        targets = []

        for i in range(len(data) - self.sequence_length):
            seq = data[i:i + self.sequence_length]
            target = data[i + self.sequence_length]
            sequences.append(seq)
            targets.append(target)

        return np.array(sequences), np.array(targets)

    def prepare_data(self, data_frame):
        """Prepare data for training"""
        target_data = data_frame[self.target_col].values.reshape(-1, 1)

        scaled_data = self.scaler.fit_transform(target_data).flatten()

        X, y = self.prepare_sequences(scaled_data)

        # 80% train, 10% validation, 10% test
        train_size = int(len(X) * 0.8)
        val_size = int(len(X) * 0.9)

        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:val_size], y[train_size:val_size]
        X_test, y_test = X[val_size:], y[val_size:]

        # Reshape for CNN-BiLSTM: (samples, seq_length, features)
        X_train = torch.FloatTensor(X_train).unsqueeze(-1)
        y_train = torch.FloatTensor(y_train).unsqueeze(-1)
        X_val = torch.FloatTensor(X_val).unsqueeze(-1)
        y_val = torch.FloatTensor(y_val).unsqueeze(-1)
        X_test = torch.FloatTensor(X_test).unsqueeze(-1)
        y_test = torch.FloatTensor(y_test).unsqueeze(-1)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def train_model(self, X_train, y_train, X_val, y_val, params, epochs=100, patience=15):
        """Train the model with given parameters"""
        # Initialize model
        self.model = CNNBiLSTM(
            input_size=1,
            seq_length=self.sequence_length,
            conv_filters=params['conv_filters'],
            kernel_size=params['kernel_size'],
            lstm_hidden_size=params['lstm_hidden_size'],
            lstm_num_layers=params['lstm_num_layers'],
            dropout_rate=params['dropout_rate'],
            fc1_units=params['fc1_units'],
            fc2_units=params['fc2_units']
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print(f"Total parameters: {total_params:,}")
        # print(f"Trainable parameters: {trainable_params:,}")

        # loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(),
                                lr=params['learning_rate'],
                                weight_decay=params.get('weight_decay', 1e-5))

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6
        )

        # early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': []
        }

        print(f"\nTraining CNN-BiLSTM model with parameters: {params}")

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            batch_count = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()
                batch_count += 1

            avg_train_loss = train_loss / len(train_loader)
            self.training_history['train_losses'].append(avg_train_loss)

            # validation
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val)
                val_loss = criterion(val_predictions, y_val).item()
                self.training_history['val_losses'].append(val_loss)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            self.training_history['learning_rates'].append(current_lr)

            # early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()

                model_filename = f'best_CNN_BiLSTM_model_{self.target_col}_seq{self.sequence_length}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'params': params,
                    'training_history': self.training_history
                }, os.path.join(self.models_path, model_filename))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        self.save_training_history(params)

        return best_val_loss, self.training_history['train_losses'], \
            self.training_history['val_losses'], self.training_history['learning_rates']

    def save_training_history(self, params):
        """Save training history to file"""
        history = {
            'params': params,
            'train_losses': [float(loss) for loss in self.training_history['train_losses']],
            'val_losses': [float(loss) for loss in self.training_history['val_losses']],
            'learning_rates': [float(lr) for lr in self.training_history['learning_rates']],
            'final_epochs': len(self.training_history['train_losses'])
        }

        filename = f'training_history_{self.target_col}_seq{self.sequence_length}.json'
        with open(os.path.join(self.metrics_path, filename), 'w') as f:
            json.dump(history, f, indent=4)

    def predict(self, X):
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            predictions = self.model(X)
        return predictions.cpu().numpy()

    def inverse_transform(self, data):
        """Inverse transform scaled data"""
        return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()

    def grid_search(self, data_frame, params_grid, epochs=50):
        """Perform grid search for best parameters"""
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.prepare_data(data_frame)

        param_names = list(params_grid.keys())
        param_values = list(params_grid.values())
        param_combinations = list(product(*param_values))

        best_score = float('inf')
        best_params = None
        results = []

        print(f"\nStarting CNN-BiLSTM grid search with {len(param_combinations)} combinations...")

        for i, combination in enumerate(param_combinations):
            params = dict(zip(param_names, combination))
            print(f"\nTesting combination {i + 1}/{len(param_combinations)}: {params}")

            val_loss, train_losses, val_losses, learning_rates = self.train_model(
                X_train, y_train, X_val, y_val, params, epochs=epochs
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
                print(f"New best score: {val_loss:.6f}")

        print(f"\nGrid search complete!")
        print(f"Best validation loss: {best_score:.6f}")
        print(f"Best set of parameters: {best_params}")

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
            'model': 'CNN-BiLSTM',
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
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.prepare_data(df)

        X_train_full = torch.cat([X_train, X_val], dim=0)
        y_train_full = torch.cat([y_train, y_val], dim=0)

        print(f"\nTraining final CNN-BiLSTM model for {self.target_col} with best parameters...")
        final_val_loss, train_losses, val_losses, learning_rates = self.train_model(
            X_train_full, y_train_full, X_val, y_val, best_params, epochs=epochs
        )

        # make predictions
        test_predictions_scaled = self.predict(X_test)
        test_actual_scaled = y_test.numpy()

        test_predictions = self.inverse_transform(test_predictions_scaled)
        test_actual = self.inverse_transform(test_actual_scaled)

        # metrics
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
        print(f"MSE: {mse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")

        fig = plt.figure(figsize=(15, 10))

        gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.25)

        # 1. Time series prediction
        ax1 = fig.add_subplot(gs[0, 0])
        time_points = range(len(test_predictions))
        ax1.plot(time_points, test_actual, 'b-', label='Actual', alpha=0.7, linewidth=2)
        ax1.plot(time_points, test_predictions, 'r--', label='Predicted', alpha=0.7, linewidth=2)
        ax1.fill_between(time_points, test_actual, test_predictions,
                         alpha=0.2, color='gray', label='Error')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel(f'{self.target_col} Value')
        ax1.set_title(f'Test Set Predictions\nSequence Length: {self.sequence_length}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Scatter plot
        ax2 = fig.add_subplot(gs[0, 1])
        scatter = ax2.scatter(test_actual, test_predictions, alpha=0.6, edgecolors='k', s=50)
        min_val = min(test_actual.min(), test_predictions.min())
        max_val = max(test_actual.max(), test_predictions.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax2.set_xlabel('Actual Values')
        ax2.set_ylabel('Predicted Values')
        ax2.set_title(f'Actual vs Predicted\nR² = {r2:.4f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Error distribution
        ax3 = fig.add_subplot(gs[1, 0])
        errors = test_predictions - test_actual
        n, bins, patches = ax3.hist(errors, bins=30, edgecolor='black', alpha=0.7, density=True)
        ax3.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        ax3.axvline(x=np.mean(errors), color='g', linestyle='-',
                    linewidth=2, label=f'Mean: {np.mean(errors):.2f}')

        ax3.set_xlabel('Prediction Error')
        ax3.set_ylabel('Density')
        ax3.set_title(f'Error Distribution\nStd: {np.std(errors):.2f}, MAE: {mae:.2f}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Metrics summary table
        ax4 = fig.add_subplot(gs[1, 1])
        metrics_text = f"""
        CNN-BiLSTM Model Performance Summary
        {'=' * 40}
        Sequence Length: {self.sequence_length}
        Target Variable: {self.target_col}

        Performance Metrics:
        --------------------
        MSE:     {mse:.2f}
        MAE:     {mae:.2f}
        RMSE:    {rmse:.2f}
        R²:      {r2:.4f}
        MAPE:    {mape:.2f}%

        CNN Parameters:
        ----------------
        Conv Filters:   {best_params.get('conv_filters', 'N/A')}
        Kernel Size:    {best_params.get('kernel_size', 'N/A')}

        BiLSTM Parameters:
        ------------------
        Hidden Size:    {best_params.get('lstm_hidden_size', 'N/A')}
        Num Layers:     {best_params.get('lstm_num_layers', 'N/A')}

        Training Parameters:
        --------------------
        Dropout Rate:   {best_params.get('dropout_rate', 'N/A'):.3f}
        Learning Rate:  {best_params.get('learning_rate', 'N/A'):.5f}
        Batch Size:     {best_params.get('batch_size', 'N/A')}
        """
        ax4.text(0.05, 0.5, metrics_text, fontsize=9, family='monospace',
                 verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.axis('off')
        ax4.set_title('Model Summary')

        plt.suptitle(f'CNN-BiLSTM Results for {self.target_col} Prediction\nSequence Length: {self.sequence_length}',
                     fontsize=16, y=1.02)
        plt.tight_layout()

        plot_filename = f'CNN_BiLSTM_results_{self.target_col}_seq{self.sequence_length}.png'
        plt.savefig(os.path.join(self.plots_path, plot_filename), dpi=150, bbox_inches='tight')
        plt.show()

        self.plot_loss_curves(train_losses, val_losses, learning_rates)

        return test_predictions, test_actual, mse, mae, rmse, r2, mape

    def plot_loss_curves(self, train_losses, val_losses, learning_rates):
        """Plot and save loss curves separately"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        epochs_range = range(1, len(train_losses) + 1)
        axes[0].plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs_range, val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'Loss Curves - {self.target_col}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')

        # Learning rate
        axes[1].plot(epochs_range, learning_rates[:len(epochs_range)], 'g-', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title(f'Learning Rate Schedule - {self.target_col}')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')

        # Loss difference
        loss_diff = [abs(t - v) for t, v in zip(train_losses, val_losses)]
        axes[2].plot(epochs_range, loss_diff, 'purple', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('|Train Loss - Val Loss|')
        axes[2].set_title(f'Loss Difference - {self.target_col}')
        axes[2].grid(True, alpha=0.3)

        plt.suptitle(f'Training History - CNN-BiLSTM (Seq Len: {self.sequence_length})', fontsize=14)
        plt.tight_layout()

        loss_plot_filename = f'loss_curves_{self.target_col}_seq{self.sequence_length}.png'
        plt.savefig(os.path.join(self.plots_path, loss_plot_filename), dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_features(self, sample_batch):
        """Visualize CNN and LSTM features for analysis"""
        self.model.eval()
        with torch.no_grad():
            sample_batch = sample_batch.to(self.device)
            predictions, cnn_features, lstm_features = self.model(sample_batch, return_features=True)

            cnn_features = cnn_features.cpu().numpy()
            lstm_features = lstm_features.cpu().numpy()

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            # CNN features heatmap
            im1 = axes[0, 0].imshow(cnn_features[0].T, aspect='auto', cmap='viridis')
            axes[0, 0].set_xlabel('Time Step')
            axes[0, 0].set_ylabel('CNN Feature Channels')
            axes[0, 0].set_title('CNN Features (First Sample)')
            plt.colorbar(im1, ax=axes[0, 0])

            # LSTM features heatmap
            im2 = axes[0, 1].imshow(lstm_features[0].T, aspect='auto', cmap='plasma')
            axes[0, 1].set_xlabel('Time Step')
            axes[0, 1].set_ylabel('LSTM Hidden States')
            axes[0, 1].set_title('LSTM Features (First Sample)')
            plt.colorbar(im2, ax=axes[0, 1])

            # Feature statistics
            axes[1, 0].bar(['CNN Features', 'LSTM Features'],
                           [cnn_features.std(), lstm_features.std()])
            axes[1, 0].set_ylabel('Feature Standard Deviation')
            axes[1, 0].set_title('Feature Variation')
            axes[1, 0].grid(True, alpha=0.3)

            # Feature correlation
            cnn_flat = cnn_features[0].flatten()
            lstm_flat = lstm_features[0].flatten()
            min_len = min(len(cnn_flat), len(lstm_flat))
            correlation = np.corrcoef(cnn_flat[:min_len], lstm_flat[:min_len])[0, 1]

            axes[1, 1].scatter(cnn_flat[:500], lstm_flat[:500], alpha=0.6)
            axes[1, 1].set_xlabel('CNN Features')
            axes[1, 1].set_ylabel('LSTM Features')
            axes[1, 1].set_title(f'Feature Correlation: {correlation:.3f}')
            axes[1, 1].grid(True, alpha=0.3)

            plt.suptitle(f'Feature Visualization - CNN-BiLSTM (Seq Len: {self.sequence_length})',
                         fontsize=14, y=1.02)
            plt.tight_layout()

            # Save feature visualization
            feature_plot_filename = f'feature_visualization_{self.target_col}_seq{self.sequence_length}.png'
            plt.savefig(os.path.join(self.plots_path, feature_plot_filename), dpi=150, bbox_inches='tight')
            plt.close()

    def save_final_metrics(self, metrics_dict):
        """Save final metrics to CSV"""
        metrics_df = pd.DataFrame([metrics_dict])
        filename = f'final_metrics_{self.target_col}_seq{self.sequence_length}.csv'
        metrics_df.to_csv(os.path.join(self.metrics_path, filename), index=False)
        print(f"Final metrics saved to {filename}")


def run_cnn_bilstm_experiment(df, sequence_length, param_grid, epochs=50):
    """Run complete CNN-BiLSTM experiment for a specific sequence length"""
    print(f"\n{'=' * 80}")
    print(f"CNN-BiLSTM HYBRID EXPERIMENT - SEQUENCE LENGTH: {sequence_length}")
    print(f"{'=' * 80}")

    all_results = {}

    # Predict Residential Usage
    print(f"\n{'=' * 60}")
    print(f"PREDICTING RESIDENTIAL USAGE (Seq Len: {sequence_length})")
    print(f"{'=' * 60}")

    residential_predictor = CNNBiLSTMPredictor(
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
        'model': 'CNN-BiLSTM',
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

    commercial_predictor = CNNBiLSTMPredictor(
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
        'model': 'CNN-BiLSTM',
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
    print(f"CNN-BiLSTM SUMMARY FOR SEQUENCE LENGTH: {sequence_length}")
    print(f"{'=' * 60}")
    print(f"{'Metric':<15} {'Residential':<15} {'Commercial':<15}")
    print(f"{'-' * 45}")
    print(f"{'MSE':<15} {res_mse:<15.2f} {com_mse:<15.2f}")
    print(f"{'MAE':<15} {res_mae:<15.2f} {com_mae:<15.2f}")
    print(f"{'RMSE':<15} {res_rmse:<15.2f} {com_rmse:<15.2f}")
    print(f"{'R^2':<15} {res_r2:<15.4f} {com_r2:<15.4f}")
    print(f"{'MAPE':<15} {res_mape:<15.2f} {com_mape:<15.2f}")

    all_results[sequence_length] = {
        'residential': res_metrics,
        'commercial': com_metrics
    }

    return all_results


def create_cnn_bilstm_comparison_summary(all_results):
    """Create a summary comparison across all sequence lengths for CNN-BiLSTM"""
    print(f"\n{'=' * 80}")
    print("CNN-BiLSTM COMPARISON ACROSS ALL SEQUENCE LENGTHS")
    print(f"{'=' * 80}")

    summary_data = []

    for seq_len, results in all_results.items():
        # Residential
        summary_data.append({
            'Model': 'CNN-BiLSTM',
            'Sequence_Length': seq_len,
            'Target': 'Residential',
            'MSE': results['residential']['mse'],
            'MAE': results['residential']['mae'],
            'RMSE': results['residential']['rmse'],
            'R2': results['residential']['r2'],
            'MAPE': results['residential']['mape']
        })

        # Commercial
        summary_data.append({
            'Model': 'CNN-BiLSTM',
            'Sequence_Length': seq_len,
            'Target': 'Commercial',
            'MSE': results['commercial']['mse'],
            'MAE': results['commercial']['mae'],
            'RMSE': results['commercial']['rmse'],
            'R2': results['commercial']['r2'],
            'MAPE': results['commercial']['mape']
        })

    summary_df = pd.DataFrame(summary_data)

    summary_path = os.path.join(metrics_base_path, 'CNN_BiLSTM_summary_comparison.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nCNN-BiLSTM summary comparison saved to: {summary_path}")

    print(f"\n{'Model':<12} {'Seq Len':<10} {'Target':<12} {'MSE':<10} {'MAE':<10} {'RMSE':<10} {'R²':<8} {'MAPE':<10}")
    print(f"{'-' * 92}")

    for idx, row in summary_df.iterrows():
        print(f"{row['Model']:<12} {row['Sequence_Length']:<10} {row['Target']:<12} "
              f"{row['MSE']:<10.2f} {row['MAE']:<10.2f} {row['RMSE']:<10.2f} "
              f"{row['R2']:<8.4f} {row['MAPE']:<10.2f}")

    return summary_df


def plot_cnn_bilstm_comparison(summary_df):
    """Create comparison plots for CNN-BiLSTM across different sequence lengths"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics = ['MSE', 'MAE', 'RMSE', 'R2', 'MAPE']
    metric_titles = ['MSE (Lower is better)', 'MAE (Lower is better)',
                     'RMSE (Lower is better)', 'R² (Higher is better)',
                     'MAPE % (Lower is better)']

    for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[idx // 3, idx % 3]

        # Separate residential and commercial
        residential_data = summary_df[summary_df['Target'] == 'Residential']
        commercial_data = summary_df[summary_df['Target'] == 'Commercial']

        x_pos = np.arange(len(residential_data['Sequence_Length']))
        width = 0.35

        bars1 = ax.bar(x_pos - width / 2, residential_data[metric], width,
                       label='Residential', alpha=0.8, color='blue')
        bars2 = ax.bar(x_pos + width / 2, commercial_data[metric], width,
                       label='Commercial', alpha=0.8, color='orange')

        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

        autolabel(bars1)
        autolabel(bars2)

        ax.set_xlabel('Sequence Length')
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(residential_data['Sequence_Length'].astype(str))
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    axes[1, 2].axis('off')
    axes[1, 2].text(0.5, 0.5, 'CNN-BiLSTM Architecture:\n'
                              '• 3-layer CNN for spatial features\n'
                              '• Bidirectional LSTM for temporal\n'
                              '• Attention mechanism (optional)\n'
                              '• Combined feature extraction',
                    ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.suptitle('CNN-BiLSTM Hybrid Model Performance Comparison',
                 fontsize=16, y=1.02)
    plt.tight_layout()

    comparison_plot_path = os.path.join(plots_base_path, 'CNN_BiLSTM_sequence_length_comparison.png')
    plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nCNN-BiLSTM comparison plot saved to: {comparison_plot_path}")


def compare_all_models():
    """Compare results from all three models (CNN, BiLSTM, CNN-BiLSTM)"""
    print(f"\n{'=' * 80}")
    print("COMPREHENSIVE MODEL COMPARISON")
    print(f"{'=' * 80}")

    try:
        # load all model results
        cnn_summary_path = os.path.join(metrics_base_path.replace('CNN_BiLSTM', 'SA_CNN'), 'CNN_summary_comparison.csv')
        bilstm_summary_path = os.path.join(metrics_base_path.replace('CNN_BiLSTM', 'SA_BiLSTM'),
                                           'BiLSTM_summary_comparison.csv')
        cnn_bilstm_summary_path = os.path.join(metrics_base_path, 'CNN_BiLSTM_summary_comparison.csv')

        all_summaries = []

        if os.path.exists(cnn_summary_path):
            cnn_df = pd.read_csv(cnn_summary_path)
            all_summaries.append(cnn_df)
            print(f"Loaded CNN results")

        if os.path.exists(bilstm_summary_path):
            bilstm_df = pd.read_csv(bilstm_summary_path)
            all_summaries.append(bilstm_df)
            print(f"Loaded BiLSTM results")

        if os.path.exists(cnn_bilstm_summary_path):
            cnn_bilstm_df = pd.read_csv(cnn_bilstm_summary_path)
            all_summaries.append(cnn_bilstm_df)
            print(f"✓ Loaded CNN-BiLSTM results")

        if all_summaries:
            combined_df = pd.concat(all_summaries, ignore_index=True)

            combined_path = os.path.join(metrics_base_path, 'ALL_MODELS_COMPARISON.csv')
            combined_df.to_csv(combined_path, index=False)
            print(f"\nAll models comparison saved to: {combined_path}")

            print(
                f"\n{'Model':<12} {'Seq Len':<10} {'Target':<12} {'MSE':<10} {'MAE':<10} {'RMSE':<10} {'R²':<8} {'MAPE':<10}")
            print(f"{'-' * 92}")

            combined_df = combined_df.sort_values(['Model', 'Sequence_Length', 'Target'])

            for idx, row in combined_df.iterrows():
                print(f"{row['Model']:<12} {row['Sequence_Length']:<10} {row['Target']:<12} "
                      f"{row['MSE']:<10.2f} {row['MAE']:<10.2f} {row['RMSE']:<10.2f} "
                      f"{row['R2']:<8.4f} {row['MAPE']:<10.2f}")

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))

            metrics = ['MSE', 'MAE', 'RMSE', 'R2', 'MAPE']
            metric_titles = ['MSE Comparison', 'MAE Comparison',
                             'RMSE Comparison', 'R² Comparison',
                             'MAPE Comparison']

            models = ['CNN', 'BiLSTM', 'CNN-BiLSTM']
            colors = ['blue', 'orange', 'green']

            for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
                ax = axes[idx // 3, idx % 3]

                # Group by model and sequence length
                for model, color in zip(models, colors):
                    model_data = combined_df[(combined_df['Model'] == model) &
                                             (combined_df['Target'] == 'Residential')]
                    if not model_data.empty:
                        ax.plot(model_data['Sequence_Length'], model_data[metric],
                                marker='o', label=model, color=color, linewidth=2)

                ax.set_xlabel('Sequence Length')
                ax.set_ylabel(metric)
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xticks([1, 3, 12])

            axes[1, 2].axis('off')
            axes[1, 2].text(0.5, 0.5, 'Model Comparison Summary:\n'
                                      '• CNN: Spatial feature extraction\n'
                                      '• BiLSTM: Temporal dependencies\n'
                                      '• CNN-BiLSTM: Combined approach',
                            ha='center', va='center', fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

            plt.suptitle('Comprehensive Model Performance Comparison',
                         fontsize=16, y=1.02)
            plt.tight_layout()

            comparison_path = os.path.join(plots_base_path, 'ALL_MODELS_COMPARISON.png')
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.show()
            print(f"\nComprehensive comparison plot saved to: {comparison_path}")

        else:
            print("No model results found for comparison.")

    except Exception as e:
        print(f"\nNote: Could not compare all models. Error: {e}")
        print("Make sure all experiments (CNN, BiLSTM) were run first for complete comparison.")


if __name__ == "__main__":
    for base_path in [models_base_path, plots_base_path, metrics_base_path]:
        os.makedirs(base_path, exist_ok=True)

    print("Loading processed data...")
    df = pd.read_csv("../data/processed/dataset.csv", index_col='Date', parse_dates=True)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    cnn_bilstm_param_grid = {
        # up to attention and hidden size: [50]
        'conv_filters': [2, 4],
        'kernel_size': [3],
        'lstm_hidden_size': [50, 100],
        'lstm_num_layers': [3, 4],
        'dropout_rate': [0.1],
        'fc1_units': [16, 32],
        'fc2_units': [8, 16],
        'learning_rate': [0.01],
        'batch_size': [128],
        'weight_decay': [1e-6]  # L2 regularization
    }

    sequence_lengths = [1, 3, 12]
    TOTAL_EPOCHES = 20

    all_cnn_bilstm_results = {}

    for seq_len in sequence_lengths:
        seq_results = run_cnn_bilstm_experiment(
            df, seq_len, cnn_bilstm_param_grid, epochs=TOTAL_EPOCHES
        )
        all_cnn_bilstm_results.update(seq_results)

    cnn_bilstm_summary_df = create_cnn_bilstm_comparison_summary(all_cnn_bilstm_results)

    plot_cnn_bilstm_comparison(cnn_bilstm_summary_df)

    compare_all_models()

    print(f"\n{'=' * 80}")
    print("CNN-BiLSTM HYBRID EXPERIMENT COMPLETE!")
    print(f"{'=' * 80}")
    print("\nAll CNN-BiLSTM results have been saved in the following structure:")
    print(f"Models: {models_base_path}seq_*/")
    print(f"Plots: {plots_base_path}seq_*/")
    print(f"Metrics: {metrics_base_path}seq_*/")
    print(f"\nCNN-BiLSTM summary: {metrics_base_path}CNN_BiLSTM_summary_comparison.csv")