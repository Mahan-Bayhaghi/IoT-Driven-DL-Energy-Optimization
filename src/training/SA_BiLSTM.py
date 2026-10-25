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
models_base_path = base_results_path + "models/SA_BiLSTM/"
plots_base_path = base_results_path + "plots/SA_BiLSTM/"
metrics_base_path = base_results_path + "metrics/SA_BiLSTM/"


class BiLSTMModel(nn.Module):
    """
    Bidirectional LSTM for time series prediction
    """

    def __init__(self, input_size=1, hidden_size=32, num_layers=2,
                 dropout_rate=0.2, fc1_units=16, fc2_units=8):
        super(BiLSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout_rate)

        # for bidirectional LSTM, output size is hidden_size * 2
        self.fc1 = nn.Linear(hidden_size * 2, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)  # output: a single prediction

        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)

        batch_size = x.size(0)
        lstm_out, (hidden, cell) = self.lstm(x)

        # bidirectional LSTM, we use the concatenated forward and backward hidden states
        lstm_out = lstm_out[:, -1, :]  # Take last time step output

        lstm_out = self.dropout(lstm_out)

        x = self.relu(self.fc1(lstm_out))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class BiLSTMPredictor:
    def __init__(self, sequence_length=12, target_col='Residential_Usage'):
        self.sequence_length = sequence_length
        self.target_col = target_col
        self.scaler = MinMaxScaler(feature_range=(0, 1))
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

        # reshape for LSTM: (samples, seq_length, features)
        X_train = torch.FloatTensor(X_train).unsqueeze(-1)
        y_train = torch.FloatTensor(y_train).unsqueeze(-1)
        X_val = torch.FloatTensor(X_val).unsqueeze(-1)
        y_val = torch.FloatTensor(y_val).unsqueeze(-1)
        X_test = torch.FloatTensor(X_test).unsqueeze(-1)
        y_test = torch.FloatTensor(y_test).unsqueeze(-1)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def train_model(self, X_train, y_train, X_val, y_val, params, epochs=100, patience=15):
        """Train the model with given parameters"""
        self.model = BiLSTMModel(
            input_size=1,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout_rate=params['dropout_rate'],
            fc1_units=params['fc1_units'],
            fc2_units=params['fc2_units']
        ).to(self.device)

        # loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=params['learning_rate'])

        # lr scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
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

        train_losses = []
        val_losses = []
        learning_rates = []

        print(f"\nTraining BiLSTM model with parameters: {params}")
        # print(f"Model architecture: {self.model}")

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            batch_count = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()

                # Gradient clipping for LSTM stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()
                batch_count += 1

            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # validation
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val)
                val_loss = criterion(val_predictions, y_val).item()
                val_losses.append(val_loss)

            # lr scheduler step
            scheduler.step(val_loss)
            learning_rates.append(optimizer.param_groups[0]['lr'])

            # early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()

                model_filename = f'best_BiLSTM_model_{self.target_col}_seq{self.sequence_length}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'params': params
                }, os.path.join(self.models_path, model_filename))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # if (epoch + 1) % 10 == 0:
            #     print(f"Epoch [{epoch + 1}/{epochs}], "
            #           f"Train Loss: {avg_train_loss:.6f}, "
            #           f"Val Loss: {val_loss:.6f}, "
            #           f"LR: {learning_rates[-1]:.6f}")

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        self.save_training_history(train_losses, val_losses, learning_rates, params)

        return best_val_loss, train_losses, val_losses, learning_rates

    def save_training_history(self, train_losses, val_losses, learning_rates, params):
        """Save training history to file"""
        history = {
            'params': params,
            'train_losses': [float(loss) for loss in train_losses],
            'val_losses': [float(loss) for loss in val_losses],
            'learning_rates': [float(lr) for lr in learning_rates],
            'final_epochs': len(train_losses)
        }

        # filename = f'training_history_{self.target_col}_seq{self.sequence_length}.json'
        # with open(os.path.join(self.metrics_path, filename), 'w') as f:
        #     json.dump(history, f, indent=4)

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

        print(f"\nStarting grid search with {len(param_combinations)} combinations...")

        for i, combination in enumerate(param_combinations):
            params = dict(zip(param_names, combination))
            print(f"\nTesting combination {i + 1}/{len(param_combinations)}: {params}")

            # Train model with current parameters
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
                'learning_rates': [float(lr) for lr in result['learning_rates']] if 'learning_rates' in result else []
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
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.prepare_data(df)

        # Combine train and validation for final training
        X_train_full = torch.cat([X_train, X_val], dim=0)
        y_train_full = torch.cat([y_train, y_val], dim=0)

        print(f"\nTraining final BiLSTM model for {self.target_col} with best parameters...")
        final_val_loss, train_losses, val_losses, learning_rates = self.train_model(
            X_train_full, y_train_full, X_val, y_val, best_params, epochs=epochs
        )

        # make predictions
        test_predictions_scaled = self.predict(X_test)
        test_actual_scaled = y_test.numpy()

        test_predictions = self.inverse_transform(test_predictions_scaled)
        test_actual = self.inverse_transform(test_actual_scaled)

        # calculate metrics
        mse = np.mean((test_predictions - test_actual) ** 2)
        mae = np.mean(np.abs(test_predictions - test_actual))
        rmse = np.sqrt(mse)

        # R² (coefficient of determination)
        ss_res = np.sum((test_actual - test_predictions) ** 2)
        ss_tot = np.sum((test_actual - np.mean(test_actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # MAPE (Mean Absolute Percentage Error)
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

        # Time series prediction
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

        # Scatter plot
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

        # Error distribution
        ax3 = fig.add_subplot(gs[1, 0])
        errors = test_predictions - test_actual
        n, bins, patches = ax3.hist(errors, bins=30, edgecolor='black', alpha=0.7, density=True)
        ax3.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        ax3.axvline(x=np.mean(errors), color='g', linestyle='-',
                    linewidth=2, label=f'Mean: {np.mean(errors):.2f}')

        # # Add normal distribution fit
        # from scipy.stats import norm
        # mu, std = norm.fit(errors)
        # xmin, xmax = ax3.get_xlim()
        # x = np.linspace(xmin, xmax, 100)
        # p = norm.pdf(x, mu, std)
        # ax3.plot(x, p, 'k', linewidth=2, label=f'Normal fit\nμ={mu:.2f}, σ={std:.2f}')

        ax3.set_xlabel('Prediction Error')
        ax3.set_ylabel('Density')
        ax3.set_title(f'Error Distribution\nStd: {np.std(errors):.2f}, MAE: {mae:.2f}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Metrics summary table
        ax4 = fig.add_subplot(gs[1, 1])
        metrics_text = f"""
        BiLSTM Model Performance Summary
        {'=' * 30}
        Sequence Length: {self.sequence_length}
        Target Variable: {self.target_col}

        Performance Metrics:
        --------------------
        MSE:     {mse:.2f}
        MAE:     {mae:.2f}
        RMSE:    {rmse:.2f}
        R²:      {r2:.4f}
        MAPE:    {mape:.2f}%

        Best Model Parameters:
        ----------------------
        Hidden Size:   {best_params.get('hidden_size', 'N/A')}
        Num Layers:    {best_params.get('num_layers', 'N/A')}
        Dropout Rate:  {best_params.get('dropout_rate', 'N/A'):.3f}
        Learning Rate: {best_params.get('learning_rate', 'N/A'):.5f}
        Batch Size:    {best_params.get('batch_size', 'N/A')}
        """
        ax4.text(0.05, 0.5, metrics_text, fontsize=10, family='monospace',
                 verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.axis('off')
        ax4.set_title('Model Summary')

        plt.suptitle(f'BiLSTM Results for {self.target_col} Prediction\nSequence Length: {self.sequence_length}',
                     fontsize=16, y=1.02)
        plt.tight_layout()

        plot_filename = f'BiLSTM_results_{self.target_col}_seq{self.sequence_length}.png'
        plt.savefig(os.path.join(self.plots_path, plot_filename), dpi=150, bbox_inches='tight')
        plt.show()

        # # Save loss curves
        # self.plot_loss_curves(train_losses, val_losses, learning_rates)

        return test_predictions, test_actual, mse, mae, rmse, r2, mape

    # def plot_loss_curves(self, train_losses, val_losses, learning_rates):
    #     """Plot and save loss curves separately"""
    #     fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    #
    #     # Loss curves
    #     epochs_range = range(1, len(train_losses) + 1)
    #     axes[0].plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2)
    #     axes[0].plot(epochs_range, val_losses, 'r-', label='Val Loss', linewidth=2)
    #     axes[0].set_xlabel('Epoch')
    #     axes[0].set_ylabel('Loss')
    #     axes[0].set_title(f'Loss Curves - {self.target_col}')
    #     axes[0].legend()
    #     axes[0].grid(True, alpha=0.3)
    #     axes[0].set_yscale('log')
    #
    #     # Learning rate
    #     axes[1].plot(epochs_range, learning_rates[:len(epochs_range)], 'g-', linewidth=2)
    #     axes[1].set_xlabel('Epoch')
    #     axes[1].set_ylabel('Learning Rate')
    #     axes[1].set_title(f'Learning Rate Schedule - {self.target_col}')
    #     axes[1].grid(True, alpha=0.3)
    #     axes[1].set_yscale('log')
    #
    #     plt.suptitle(f'Training History - Sequence Length: {self.sequence_length}', fontsize=14)
    #     plt.tight_layout()
    #
    #     # Save loss curves plot
    #     loss_plot_filename = f'loss_curves_{self.target_col}_seq{self.sequence_length}.png'
    #     plt.savefig(os.path.join(self.plots_path, loss_plot_filename), dpi=150, bbox_inches='tight')
    #     plt.close()

    def save_final_metrics(self, metrics_dict):
        """Save final metrics to CSV"""
        metrics_df = pd.DataFrame([metrics_dict])
        filename = f'final_metrics_{self.target_col}_seq{self.sequence_length}.csv'
        metrics_df.to_csv(os.path.join(self.metrics_path, filename), index=False)
        print(f"Final metrics saved to {filename}")


def run_bilstm_experiment(df, sequence_length, param_grid, epochs=50):
    """Run complete BiLSTM experiment for a specific sequence length"""
    print(f"\n{'=' * 80}")
    print(f"BI-LSTM EXPERIMENT - SEQUENCE LENGTH: {sequence_length}")
    print(f"{'=' * 80}")

    all_results = {}

    # predict Residential Usage
    print(f"\n{'=' * 60}")
    print(f"PREDICTING RESIDENTIAL USAGE (Seq Len: {sequence_length})")
    print(f"{'=' * 60}")

    residential_predictor = BiLSTMPredictor(
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
        'model': 'BiLSTM',
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

    # predict Commercial Usage
    print(f"\n{'=' * 60}")
    print(f"PREDICTING COMMERCIAL USAGE (Seq Len: {sequence_length})")
    print(f"{'=' * 60}")

    commercial_predictor = BiLSTMPredictor(
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
        'model': 'BiLSTM',
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
    print(f"BI-LSTM SUMMARY FOR SEQUENCE LENGTH: {sequence_length}")
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


def create_bilstm_comparison_summary(all_results):
    """Create a summary comparison across all sequence lengths for BiLSTM"""
    print(f"\n{'=' * 80}")
    print("BI-LSTM COMPARISON ACROSS ALL SEQUENCE LENGTHS")
    print(f"{'=' * 80}")

    summary_data = []

    for seq_len, results in all_results.items():
        # Residential
        summary_data.append({
            'Model': 'BiLSTM',
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
            'Model': 'BiLSTM',
            'Sequence_Length': seq_len,
            'Target': 'Commercial',
            'MSE': results['commercial']['mse'],
            'MAE': results['commercial']['mae'],
            'RMSE': results['commercial']['rmse'],
            'R2': results['commercial']['r2'],
            'MAPE': results['commercial']['mape']
        })

    summary_df = pd.DataFrame(summary_data)

    summary_path = os.path.join(metrics_base_path, 'BiLSTM_summary_comparison.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nBiLSTM summary comparison saved to: {summary_path}")

    print(f"\n{'Model':<10} {'Seq Len':<10} {'Target':<12} {'MSE':<10} {'MAE':<10} {'RMSE':<10} {'R²':<8} {'MAPE':<10}")
    print(f"{'-' * 90}")

    for idx, row in summary_df.iterrows():
        print(f"{row['Model']:<10} {row['Sequence_Length']:<10} {row['Target']:<12} "
              f"{row['MSE']:<10.2f} {row['MAE']:<10.2f} {row['RMSE']:<10.2f} "
              f"{row['R2']:<8.4f} {row['MAPE']:<10.2f}")

    return summary_df


def plot_bilstm_comparison(summary_df):
    """Create comparison plots for BiLSTM across different sequence lengths"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics = ['MSE', 'MAE', 'RMSE', 'R2', 'MAPE']
    metric_titles = ['MSE (Lower is better)', 'MAE (Lower is better)',
                     'RMSE (Lower is better)', 'R² (Higher is better)',
                     'MAPE % (Lower is better)']

    for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[idx // 3, idx % 3]

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

    axes[1, 2].text(0.5, 0.5, 'BiLSTM Model Architecture:\n'
                              '• Bidirectional LSTM\n'
                              '• Multiple layers\n'
                              '• Dropout for regularization\n'
                              '• Fully connected output',
                    ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('BiLSTM Performance Comparison Across Different Sequence Lengths',
                 fontsize=16, y=1.02)
    plt.tight_layout()

    comparison_plot_path = os.path.join(plots_base_path, 'BiLSTM_sequence_length_comparison.png')
    plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nBiLSTM comparison plot saved to: {comparison_plot_path}")


# Main execution
if __name__ == "__main__":
    for base_path in [models_base_path, plots_base_path, metrics_base_path]:
        os.makedirs(base_path, exist_ok=True)

    print("Loading processed data...")
    df = pd.read_csv("../data/processed/dataset.csv", index_col='Date', parse_dates=True)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    bilstm_param_grid = {
        # up to hidden size: [50]
        'hidden_size': [50],
        'num_layers': [2],
        'dropout_rate': [0.1],
        'fc1_units': [8, 16],
        'fc2_units': [4, 8],
        'learning_rate': [0.05, 0.01],
        'batch_size': [128]
    }

    sequence_lengths = [1, 3, 12]
    TOTAL_EPOCHES = 20

    all_bilstm_results = {}

    for seq_len in sequence_lengths:
        seq_results = run_bilstm_experiment(
            df, seq_len, bilstm_param_grid, epochs=TOTAL_EPOCHES
        )
        all_bilstm_results.update(seq_results)

    bilstm_summary_df = create_bilstm_comparison_summary(all_bilstm_results)

    plot_bilstm_comparison(bilstm_summary_df)

    print(f"\n{'=' * 80}")
    print("BI-LSTM EXPERIMENT COMPLETE!")
    print(f"{'=' * 80}")
    print("\nAll BiLSTM results have been saved in the following structure:")
    print(f"Models: {models_base_path}seq_*/")
    print(f"Plots: {plots_base_path}seq_*/")
    print(f"Metrics: {metrics_base_path}seq_*/")
    print(f"\nBiLSTM summary comparison: {metrics_base_path}BiLSTM_summary_comparison.csv")

    # Compare with CNN results if available
    try:
        cnn_summary_path = os.path.join(metrics_base_path.replace('SA_BiLSTM', 'SA_CNN'), 'summary_comparison.csv')
        if os.path.exists(cnn_summary_path):
            print(f"\n{'=' * 80}")
            print("COMPARING CNN vs BiLSTM RESULTS")
            print(f"{'=' * 80}")

            cnn_summary = pd.read_csv(cnn_summary_path)
            cnn_summary['Model'] = 'CNN'

            combined_summary = pd.concat([cnn_summary, bilstm_summary_df], ignore_index=True)

            combined_path = os.path.join(metrics_base_path, 'CNN_vs_BiLSTM_comparison.csv')
            combined_summary.to_csv(combined_path, index=False)
            print(f"CNN vs BiLSTM comparison saved to: {combined_path}")

            print(
                f"\n{'Model':<10} {'Seq Len':<10} {'Target':<12} {'MSE':<10} {'MAE':<10} {'RMSE':<10} {'R²':<8} {'MAPE':<10}")
            print(f"{'-' * 90}")

            for idx, row in combined_summary.iterrows():
                print(f"{row['Model']:<10} {row['Sequence_Length']:<10} {row['Target']:<12} "
                      f"{row['MSE']:<10.2f} {row['MAE']:<10.2f} {row['RMSE']:<10.2f} "
                      f"{row['R2']:<8.4f} {row['MAPE']:<10.2f}")
    except Exception as e:
        print(f"\nNote: Could not compare with CNN results. Error: {e}")
        print("Make sure CNN experiments were run first for comparison.")