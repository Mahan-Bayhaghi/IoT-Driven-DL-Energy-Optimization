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

warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

base_results_path = "../../results/"
models_base_path = base_results_path + "models/SA_CNN/"
plots_base_path = base_results_path + "plots/SA_CNN/"
metrics_base_path = base_results_path + "metrics/SA_CNN/"


class Simple1DCNN(nn.Module):
    """
    Simple 1D CNN for time series prediction
    """

    def __init__(self, input_size=1, seq_length=12, conv_filters=32, kernel_size=3,
                 fc1_units=50, fc2_units=25, dropout_rate=0.2):
        super(Simple1DCNN, self).__init__()

        self.seq_length = seq_length

        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_size,
                               out_channels=conv_filters,
                               kernel_size=kernel_size,
                               padding='same')
        self.bn1 = nn.BatchNorm1d(conv_filters)
        self.conv2 = nn.Conv1d(in_channels=conv_filters,
                               out_channels=conv_filters * 2,
                               kernel_size=kernel_size,
                               padding='same')
        self.bn2 = nn.BatchNorm1d(conv_filters * 2)

        conv_output_size = conv_filters * 2 * seq_length

        self.fc1 = nn.Linear(conv_output_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)  # output: a single prediction

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        # for Conv1d, we need (batch_size, input_size, seq_length)
        x = x.transpose(1, 2)

        # convolutional layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        # flatten
        x = x.flatten(start_dim=1)

        # fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class TimeSeriesPredictor:
    def __init__(self, sequence_length=2, target_col='Residential_Usage',
                 results_suffix=''):
        self.sequence_length = sequence_length
        self.target_col = target_col
        self.results_suffix = results_suffix
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create directories for this sequence length
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

        X_train = torch.FloatTensor(X_train).unsqueeze(-1)  # Add channel dimension
        y_train = torch.FloatTensor(y_train).unsqueeze(-1)
        X_val = torch.FloatTensor(X_val).unsqueeze(-1)
        y_val = torch.FloatTensor(y_val).unsqueeze(-1)
        X_test = torch.FloatTensor(X_test).unsqueeze(-1)
        y_test = torch.FloatTensor(y_test).unsqueeze(-1)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def train_model(self, X_train, y_train, X_val, y_val, params, epochs=100, patience=10):
        """Train the model with given parameters"""
        # Initialize model
        self.model = Simple1DCNN(
            input_size=1,
            seq_length=self.sequence_length,
            conv_filters=params['conv_filters'],
            kernel_size=params['kernel_size'],
            fc1_units=params['fc1_units'],
            fc2_units=params['fc2_units'],
            dropout_rate=params['dropout_rate']
        ).to(self.device)

        # loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=params['learning_rate'])

        # early stopping
        best_val_loss = float('inf')
        patience_counter = 0

        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # validation
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val)
                val_loss = criterion(val_predictions, y_val).item()
                val_losses.append(val_loss)

            # early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # saving best model
                model_filename = f'best_SA_CNN_model_{self.target_col}_seq{self.sequence_length}.pth'
                torch.save(self.model.state_dict(), os.path.join(self.models_path, model_filename))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Load best model
        model_filename = f'best_SA_CNN_model_{self.target_col}_seq{self.sequence_length}.pth'
        self.model.load_state_dict(torch.load(os.path.join(self.models_path, model_filename)))

        return best_val_loss, train_losses, val_losses

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

    def grid_search(self, data_frame, params_grid, epochs=20):
        """Perform grid search for best parameters"""
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.prepare_data(data_frame)

        param_names = list(params_grid.keys())
        param_values = list(params_grid.values())
        param_combinations = list(product(*param_values))

        best_score = float('inf')
        best_params = None
        results = []

        print(f"Starting grid search with {len(param_combinations)} combinations...")

        for i, combination in enumerate(param_combinations):
            params = dict(zip(param_names, combination))
            print(f"\nTesting combination {i + 1}/{len(param_combinations)}: {params}")

            # Train model with current parameters
            val_loss, train_losses, val_losses = self.train_model(
                X_train, y_train, X_val, y_val, params, epochs=epochs
            )

            results.append({
                'params': params,
                'val_loss': val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
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
                'val_losses': [float(loss) for loss in result['val_losses']]
            }
            serializable_results.append(serializable_result)

        results_data = {
            'sequence_length': self.sequence_length,
            'target_column': self.target_col,
            'best_params': best_params,
            'all_results': serializable_results
        }

        filename = f'grid_search_results_{self.target_col}_seq{self.sequence_length}.json'
        with open(os.path.join(self.metrics_path, filename), 'w') as f:
            json.dump(results_data, f, indent=4)

        print(f"Grid search results saved to {filename}")

    def plot_results(self, df, best_params, epochs=100):
        """Train with the best parameters and plot results"""
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.prepare_data(df)

        # combining train and validation for final training
        X_train_full = torch.cat([X_train, X_val], dim=0)
        y_train_full = torch.cat([y_train, y_val], dim=0)

        print(f"\nTraining final model for {self.target_col} with best parameters...")
        final_val_loss, _, val_losses = self.train_model(
            X_train_full, y_train_full, X_val, y_val, best_params, epochs=epochs
        )

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

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Time series prediction
        time_points = range(len(test_predictions))
        axes[0, 0].plot(time_points, test_actual, 'b-', label='Actual', alpha=0.7, linewidth=2)
        axes[0, 0].plot(time_points, test_predictions, 'r--', label='Predicted', alpha=0.7, linewidth=2)
        axes[0, 0].fill_between(time_points, test_actual, test_predictions,
                                alpha=0.2, color='gray', label='Error')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel(f'{self.target_col} Value')
        axes[0, 0].set_title(f'Test Set Predictions\nSequence Length: {self.sequence_length}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Scatter plot of actual vs predicted
        axes[0, 1].scatter(test_actual, test_predictions, alpha=0.6, edgecolors='k')
        min_val = min(test_actual.min(), test_predictions.min())
        max_val = max(test_actual.max(), test_predictions.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        axes[0, 1].set_xlabel('Actual Values')
        axes[0, 1].set_ylabel('Predicted Values')
        axes[0, 1].set_title(f'Actual vs Predicted\nR² = {r2:.3f}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Error distribution
        errors = test_predictions - test_actual
        axes[1, 0].hist(errors, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].axvline(x=np.mean(errors), color='g', linestyle='-', linewidth=2,
                           label=f'Mean: {np.mean(errors):.2f}')
        axes[1, 0].set_xlabel('Prediction Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Error Distribution\nStd: {np.std(errors):.2f}, MAE: {mae:.2f}')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Metrics summary table
        metrics_text = f"""
        Sequence Length: {self.sequence_length}
        -------------------------
        MSE:  {mse:.2f}
        MAE:  {mae:.2f}
        RMSE: {rmse:.2f}
        R²:   {r2:.4f}
        MAPE: {mape:.2f}%
        """
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Performance Metrics')

        plt.suptitle(f'1D CNN Results for {self.target_col} Prediction\nSequence Length: {self.sequence_length}',
                     fontsize=16, y=1.02)
        plt.tight_layout()

        plot_filename = f'cnn_results_{self.target_col}_seq{self.sequence_length}.png'
        plt.savefig(os.path.join(self.plots_path, plot_filename), dpi=150, bbox_inches='tight')
        plt.show()

        return test_predictions, test_actual, mse, mae, rmse, r2, mape

    def save_final_metrics(self, metrics_dict):
        """Save final metrics to CSV"""
        metrics_df = pd.DataFrame([metrics_dict])
        filename = f'final_metrics_{self.target_col}_seq{self.sequence_length}.csv'
        metrics_df.to_csv(os.path.join(self.metrics_path, filename), index=False)
        print(f"Final metrics saved to {filename}")


def run_experiment_for_sequence_length(df, sequence_length, param_grid, epochs=20):
    """Run complete experiment for a specific sequence length"""
    print(f"\n{'=' * 80}")
    print(f"EXPERIMENT WITH SEQUENCE LENGTH: {sequence_length}")
    print(f"{'=' * 80}")

    all_results = {}

    # Predict Residential Usage
    print(f"\n{'=' * 60}")
    print(f"PREDICTING RESIDENTIAL USAGE (Seq Len: {sequence_length})")
    print(f"{'=' * 60}")

    residential_predictor = TimeSeriesPredictor(
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
        'target': 'Residential_Usage',
        'sequence_length': sequence_length,
        'mse': res_mse,
        'mae': res_mae,
        'rmse': res_rmse,
        'r2': res_r2,
        'mape': res_mape,
        'best_params': str(best_params_res)
    }
    residential_predictor.save_final_metrics(res_metrics)

    # Predict Commercial Usage
    print(f"\n{'=' * 60}")
    print(f"PREDICTING COMMERCIAL USAGE (Seq Len: {sequence_length})")
    print(f"{'=' * 60}")

    commercial_predictor = TimeSeriesPredictor(
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
        'target': 'Commercial_Usage',
        'sequence_length': sequence_length,
        'mse': com_mse,
        'mae': com_mae,
        'rmse': com_rmse,
        'r2': com_r2,
        'mape': com_mape,
        'best_params': str(best_params_com)
    }
    commercial_predictor.save_final_metrics(com_metrics)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY FOR SEQUENCE LENGTH: {sequence_length}")
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


def create_comparison_summary(all_results):
    """Create a summary comparison across all sequence lengths"""
    print(f"\n{'=' * 80}")
    print("COMPARISON ACROSS ALL SEQUENCE LENGTHS")
    print(f"{'=' * 80}")

    summary_data = []

    for seq_len, results in all_results.items():
        # Residential
        summary_data.append({
            'Model': 'CNN',
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
            'Model': 'CNN',
            'Sequence_Length': seq_len,
            'Target': 'Commercial',
            'MSE': results['commercial']['mse'],
            'MAE': results['commercial']['mae'],
            'RMSE': results['commercial']['rmse'],
            'R2': results['commercial']['r2'],
            'MAPE': results['commercial']['mape']
        })

    summary_df = pd.DataFrame(summary_data)

    summary_path = os.path.join(metrics_base_path, 'CNN_summary_comparison.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary comparison saved to: {summary_path}")

    print(f"\n{'Sequence Length':<15} {'Target':<12} {'MSE':<10} {'MAE':<10} {'RMSE':<10} {'R²':<8} {'MAPE':<10}")
    print(f"{'-' * 85}")

    for idx, row in summary_df.iterrows():
        print(f"{row['Sequence_Length']:<15} {row['Target']:<12} {row['MSE']:<10.2f} {row['MAE']:<10.2f} "
              f"{row['RMSE']:<10.2f} {row['R2']:<8.4f} {row['MAPE']:<10.2f}")

    return summary_df


def plot_comparison_across_sequences(summary_df):
    """Create comparison plots across different sequence lengths"""
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

        ax.bar(x_pos - width / 2, residential_data[metric], width,
               label='Residential', alpha=0.8, color='blue')
        ax.bar(x_pos + width / 2, commercial_data[metric], width,
               label='Commercial', alpha=0.8, color='orange')

        ax.set_xlabel('Sequence Length')
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(residential_data['Sequence_Length'].astype(str))
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[1, 2].axis('off')

    plt.suptitle('Performance Comparison Across Different Sequence Lengths',
                 fontsize=16, y=1.02)
    plt.tight_layout()

    comparison_plot_path = os.path.join(plots_base_path, 'sequence_length_comparison.png')
    plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nComparison plot saved to: {comparison_plot_path}")


if __name__ == "__main__":
    for base_path in [models_base_path, plots_base_path, metrics_base_path]:
        os.makedirs(base_path, exist_ok=True)

    print("Loading processed data...")
    df = pd.read_csv("../data/processed/dataset.csv", index_col='Date', parse_dates=True)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    param_grid = {
        'conv_filters': [1],
        'kernel_size': [3],
        'fc1_units': [4, 8],
        'fc2_units': [2, 4],
        'dropout_rate': [0.05, 0.1],
        'learning_rate': [0.05, 0.1],
        'batch_size': [128]
    }

    sequence_lengths = [1, 3, 12]
    TOTAL_EPOCHES = 20

    all_results = {}

    for seq_len in sequence_lengths:
        seq_results = run_experiment_for_sequence_length(
            df, seq_len, param_grid, epochs=TOTAL_EPOCHES
        )
        all_results.update(seq_results)

    summary_df = create_comparison_summary(all_results)

    plot_comparison_across_sequences(summary_df)

    print(f"\n{'=' * 80}")
    print("EXPERIMENT COMPLETE!")
    print(f"{'=' * 80}")
    print("\nAll results have been saved in the following structure:")
    print(f"Models: {models_base_path}seq_*/")
    print(f"Plots: {plots_base_path}seq_*/")
    print(f"Metrics: {metrics_base_path}seq_*/")
    print(f"\nSummary comparison: {metrics_base_path}CNN_summary_comparison.csv")
