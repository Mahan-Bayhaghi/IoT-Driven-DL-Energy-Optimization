import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from itertools import product
import warnings

warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

results_path = "../../../results/"
models_path = results_path + "models/SA_CNN/"
plots_path = results_path + "plots/SA_CNN/"
metrics_path = results_path + "metrics/"


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
    def __init__(self, sequence_length=2, target_col='Residential_Usage'):
        self.sequence_length = sequence_length
        self.target_col = target_col
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

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
                torch.save(self.model.state_dict(), models_path + f'best_SA_CNN_model_{self.target_col}.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")

        self.model.load_state_dict(torch.load(models_path + f'best_SA_CNN_model_{self.target_col}.pth'))

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

                # train final model with best parameters on combined train+val
                print(f"New best score: {val_loss:.6f}")

        print(f"\nGrid search complete!")
        print(f"Best validation loss: {best_score:.6f}")
        print(f"Best set of parameters: {best_params}")

        return best_params, results

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

        # Calculate metrics
        mse = np.mean((test_predictions - test_actual) ** 2)
        mae = np.mean(np.abs(test_predictions - test_actual))
        rmse = np.sqrt(mse)

        # Calculate R² (coefficient of determination)
        ss_res = np.sum((test_actual - test_predictions) ** 2)
        ss_tot = np.sum((test_actual - np.mean(test_actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Calculate MAPE (Mean Absolute Percentage Error)
        # Handle zero values to avoid division by zero
        mask = test_actual != 0
        if np.any(mask):
            mape = np.mean(np.abs((test_actual[mask] - test_predictions[mask]) / test_actual[mask])) * 100
        else:
            mape = np.nan  # or 0, or handle as appropriate for your use case
            print("Warning: All actual values are zero, MAPE cannot be calculated")

        print(f"\nTest Set Metrics for {self.target_col}:")
        print(f"MSE: {mse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Time series prediction
        time_points = range(len(test_predictions))
        axes[0].plot(time_points, test_actual, 'b-', label='Actual', alpha=0.7)
        axes[0].plot(time_points, test_predictions, 'r--', label='Predicted', alpha=0.7)
        axes[0].fill_between(time_points, test_actual, test_predictions,
                             alpha=0.2, color='gray', label='Error')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel(f'{self.target_col} Value')
        axes[0].set_title(f'Test Set Predictions - {self.target_col}\nR²={r2:.3f}, MAPE={mape:.1f}%')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Error distribution
        errors = test_predictions - test_actual
        axes[1].hist(errors, bins=20, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1].axvline(x=np.mean(errors), color='g', linestyle='-', linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
        axes[1].set_xlabel('Prediction Error')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Error Distribution - {self.target_col}\nStd: {np.std(errors):.2f}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f'1D CNN Results for {self.target_col} Prediction', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(plots_path + f'cnn_results_{self.target_col}.png', dpi=150, bbox_inches='tight')
        plt.show()

        return test_predictions, test_actual, mse, mae, rmse, r2, mape

# Main execution
if __name__ == "__main__":
    # Load processed data
    print("Loading processed data...")
    df = pd.read_csv("../../data/processed/dataset.csv", index_col='Date', parse_dates=True)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    # Define parameter grid for grid search
    param_grid = {
        'conv_filters': [2],
        'kernel_size': [1, 2],
        'fc1_units': [4, 8],
        'fc2_units': [2, 4],
        'dropout_rate': [0.05, 0.1],
        'learning_rate': [0.005, 0.01],
        'batch_size': [128]
    }
    TOTAL_SEQ_LEN = 1
    TOTAL_EPOCHES = 20

    # Predict Residential Usage
    print("\n" + "=" * 60)
    print("PREDICTING RESIDENTIAL USAGE")
    print("=" * 60)
    residential_predictor = TimeSeriesPredictor(
        sequence_length=TOTAL_SEQ_LEN,
        target_col='Residential_Usage'
    )

    best_params_res, results_res = residential_predictor.grid_search(
        df, param_grid, epochs=TOTAL_EPOCHES
    )

    res_predictions, res_actual, res_mse, res_mae, res_rmse, res_r2, res_mape = residential_predictor.plot_results(
        df, best_params_res, epochs=TOTAL_EPOCHES
    )

    # Predict Commercial Usage
    print("\n" + "=" * 60)
    print("PREDICTING COMMERCIAL USAGE")
    print("=" * 60)
    commercial_predictor = TimeSeriesPredictor(
        sequence_length=TOTAL_SEQ_LEN,
        target_col='Commercial_Usage'
    )

    best_params_com, results_com = commercial_predictor.grid_search(
        df, param_grid, epochs=TOTAL_EPOCHES
    )

    com_predictions, com_actual, com_mse, com_mae, com_rmse, com_r2, com_mape = commercial_predictor.plot_results(
        df, best_params_com, epochs=TOTAL_EPOCHES
    )

    # Print summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<15} {'Residential':<15} {'Commercial':<15}")
    print(f"{'-' * 45}")
    print(f"{'MSE':<15} {res_mse:<15.2f} {com_mse:<15.2f}")
    print(f"{'MAE':<15} {res_mae:<15.2f} {com_mae:<15.2f}")
    print(f"{'RMSE':<15} {res_rmse:<15.2f} {com_rmse:<15.2f}")
    print(f"{'R^2':<15} {res_r2:<15.2f} {com_r2:<15.2f}")
    print(f"{'MAPE':<15} {res_mape:<15.2f} {com_mape:<15.2f}")

    results_df = pd.DataFrame({
        'Target': ['Residential_Usage', 'Commercial_Usage'],
        'Best_Val_Loss': [results_res[0]['val_loss'], results_com[0]['val_loss']],
        'Test_MSE': [res_mse, com_mse],
        'Test_MAE': [res_mae, com_mae],
        'Test_RMSE': [res_rmse, com_rmse],
        'Test_R^2': [res_r2, com_r2],
        'Test_MAPE': [res_mape, com_mape],
        'Best_Params': [str(best_params_res), str(best_params_com)]
    })

    results_df.to_csv(metrics_path + 'cnn_prediction_results.csv', index=False)
    print(f"\nResults saved to 'cnn_prediction_results.csv'")
