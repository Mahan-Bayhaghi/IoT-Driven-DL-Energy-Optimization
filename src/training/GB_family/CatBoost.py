import json
import os
import time
import warnings
from datetime import datetime

import catboost as cb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Suppress ALL warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


class CatBoostPredictor:
    """CatBoost model for time series prediction"""

    def __init__(self, sequence_length=12, target_col='Residential_Usage'):
        self.sequence_length = sequence_length
        self.target_col = target_col
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.feature_names = None

        # Base paths
        self.base_results_path = "../../../results/"
        self.models_base_path = self.base_results_path + "models/CatBoost/"
        self.plots_base_path = self.base_results_path + "plots/CatBoost/"
        self.metrics_base_path = self.base_results_path + "metrics/CatBoost/"

        # Create directories for this sequence length
        self.create_directories()
        print(f"Initializing CatBoost predictor with sequence length: {sequence_length}")

    def create_directories(self):
        """Create directories for saving results"""
        self.models_path = os.path.join(self.models_base_path, f"seq_{self.sequence_length}")
        self.plots_path = os.path.join(self.plots_base_path, f"seq_{self.sequence_length}")
        self.metrics_path = os.path.join(self.metrics_base_path, f"seq_{self.sequence_length}")

        for path in [self.models_base_path, self.plots_base_path, self.metrics_base_path,
                     self.models_path, self.plots_path, self.metrics_path]:
            os.makedirs(path, exist_ok=True)

    def create_features(self, data_frame, include_time_features=True):
        """Create features for time series prediction"""
        df = data_frame.copy()

        # Target variable
        target_series = df[self.target_col]

        # Create lag features
        for lag in range(1, self.sequence_length + 1):
            df[f'lag_{lag}'] = target_series.shift(lag)

        # Rolling statistics
        windows = [3, 6, 12, 60]
        for window in windows:
            df[f'rolling_mean_{window}'] = target_series.rolling(window=window, min_periods=1).mean()
            # df[f'rolling_std_{window}'] = target_series.rolling(window=window, min_periods=1).std()
            # df[f'rolling_min_{window}'] = target_series.rolling(window=window, min_periods=1).min()
            # df[f'rolling_max_{window}'] = target_series.rolling(window=window, min_periods=1).max()

        # Exponential weighted moving average
        df['ewm_alpha_0.3'] = target_series.ewm(alpha=0.3).mean()
        # df['ewm_alpha_0.5'] = target_series.ewm(alpha=0.5).mean()
        # df['ewm_alpha_0.7'] = target_series.ewm(alpha=0.7).mean()

        # Time-based features
        if include_time_features:
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['week_of_year'] = df.index.isocalendar().week
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            df['year'] = df.index.year
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

            # Cyclical encoding for time features
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            # df['day_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
            # df['day_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)

        # Drop rows with NaN values (from lag features)
        df = df.dropna()

        # Separate features and target
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        # Store feature names
        self.feature_names = X.columns.tolist()

        return X, y

    def prepare_data(self, data_frame):
        """Prepare data for training with time series split"""
        X, y = self.create_features(data_frame)

        # Time-based train/val/test split (80/10/10)
        split_idx1 = int(len(X) * 0.8)
        split_idx2 = int(len(X) * 0.9)

        X_train, X_val, X_test = X.iloc[:split_idx1], X.iloc[split_idx1:split_idx2], X.iloc[split_idx2:]
        y_train, y_val, y_test = y.iloc[:split_idx1], y.iloc[split_idx1:split_idx2], y.iloc[split_idx2:]

        # Scale the target variable if needed
        y_train_scaled = self.scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_val_scaled = self.scaler.transform(y_val.values.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler.transform(y_test.values.reshape(-1, 1)).flatten()

        return (X_train, y_train_scaled), (X_val, y_val_scaled), (X_test, y_test_scaled), y_test.values

    def train_model(self, X_train, y_train, X_val, y_val, params):
        """Train the CatBoost model"""
        start_time = time.time()

        # CatBoost specific parameters - create a clean copy
        cb_params = params.copy()

        # Remove parameters that will be passed separately
        cb_params.pop('iterations', None)
        cb_params.pop('n_estimators', None)  # Also remove n_estimators if it exists
        cb_params.pop('early_stopping_rounds', None)
        cb_params.pop('verbose', None)
        cb_params.pop('logging_level', None)
        cb_params.pop('verbose_eval', None)
        cb_params.pop('silent', None)

        # Ensure iterations and early_stopping_rounds are set
        iterations = params.get('iterations', params.get('n_estimators', 50))
        early_stopping_rounds = params.get('early_stopping_rounds', 20)

        # Train model with no verbosity - use logging_level='Silent' (not verbose=False)
        self.model = cb.CatBoostRegressor(
            **cb_params,
            iterations=iterations,
            early_stopping_rounds=early_stopping_rounds,
            logging_level='Silent'  # Use only one verbosity parameter
        )

        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=False  # This is for the fit method, not the constructor
        )

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        return self.model

    def predict(self, X):
        """Make predictions"""
        predictions_scaled = self.model.predict(X)
        predictions = self.scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        return predictions

    def calculate_metrics(self, y_true, y_pred):
        """Calculate all evaluation metrics"""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        # Calculate MAPE (handle zero values)
        mask = y_true != 0
        if np.any(mask):
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan

        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }

    def optimize_hyperparameters(self, data_frame, n_trials=50):
        """Perform hyperparameter optimization using Optuna"""
        (X_train, y_train), (X_val, y_val), (X_test, y_test), y_test_original = self.prepare_data(data_frame)

        print(f"\nStarting CatBoost hyperparameter optimization with {n_trials} trials...")

        def objective(trial):
            # Suggest hyperparameters for CatBoost
            params = {
                'loss_function': 'RMSE',
                'iterations': 50,
                'depth': trial.suggest_int('depth', 5, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'early_stopping_rounds': 20,
                'verbose': False,
                'logging_level': 'Silent'
            }

            # Train model
            model = self.train_model(X_train, y_train, X_val, y_val, params)

            # Predict on validation set
            y_val_pred = self.predict(X_val)
            y_val_original = self.scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

            # Calculate RMSE as objective
            rmse = np.sqrt(mean_squared_error(y_val_original, y_val_pred))

            return rmse

        # Create and run Optuna study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )

        # Run optimization with progress bar but no other output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Get best parameters
        best_params = study.best_params

        # Add fixed parameters
        best_params.update({
            'loss_function': 'RMSE',
            'iterations': 50,
            'early_stopping_rounds': 20,
            'verbose': False,
            'logging_level': 'Silent'
        })

        print(f"\nBest CatBoost parameters found:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")

        # Train final model with best parameters
        print(f"\nTraining final CatBoost model with best parameters...")
        final_model = self.train_model(X_train, y_train, X_val, y_val, best_params)

        # Evaluate on test set
        y_test_pred = self.predict(X_test)
        test_metrics = self.calculate_metrics(y_test_original, y_test_pred)

        print(f"\nTest Set Metrics for CatBoost:")
        print(f"MSE: {test_metrics['mse']:.2f}")
        print(f"MAE: {test_metrics['mae']:.2f}")
        print(f"RMSE: {test_metrics['rmse']:.2f}")
        print(f"R²: {test_metrics['r2']:.4f}")
        print(f"MAPE: {test_metrics['mape']:.2f}%")

        # Save optimization results
        self.save_optimization_results(study, best_params, test_metrics)

        # Save model
        self.save_model()

        return best_params, test_metrics, X_test, y_test_original, y_test_pred

    def save_optimization_results(self, study, best_params, test_metrics):
        """Save optimization results to file"""
        results_data = {
            'model_type': 'CatBoost',
            'sequence_length': self.sequence_length,
            'target_column': self.target_col,
            'best_params': best_params,
            'test_metrics': test_metrics,
            'optimization_history': [
                {
                    'trial': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                }
                for trial in study.trials
            ],
            'best_trial': study.best_trial.number,
            'best_value': study.best_value,
            'timestamp': datetime.now().isoformat()
        }

        filename = f'optimization_results_{self.target_col}_seq{self.sequence_length}.json'
        with open(os.path.join(self.metrics_path, filename), 'w') as f:
            json.dump(results_data, f, indent=4)

        print(f"Optimization results saved to {filename}")

    def save_model(self):
        """Save the trained model"""
        if self.model:
            model_filename = f'catboost_model_{self.target_col}_seq{self.sequence_length}.cbm'
            self.model.save_model(os.path.join(self.models_path, model_filename))
            print(f"Model saved to {model_filename}")

    def load_model(self, model_path):
        """Load a trained model"""
        self.model = cb.CatBoostRegressor()
        self.model.load_model(model_path)
        print(f"Model loaded from {model_path}")

    def plot_results(self, X_test, y_test, y_pred, metrics, best_params):
        """Plot results with feature importance"""
        # Create figure
        fig = plt.figure(figsize=(18, 12))

        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Time series prediction
        ax1 = fig.add_subplot(gs[0, :2])
        time_points = range(len(y_test))
        ax1.plot(time_points, y_test, 'b-', label='Actual', alpha=0.7, linewidth=2)
        ax1.plot(time_points, y_pred, 'r--', label='Predicted', alpha=0.7, linewidth=2)
        ax1.fill_between(time_points, y_test, y_pred,
                         alpha=0.2, color='gray', label='Error')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel(f'{self.target_col} Value')
        ax1.set_title(f'CatBoost Predictions\nSequence Length: {self.sequence_length}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Scatter plot
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', s=50)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax2.set_xlabel('Actual Values')
        ax2.set_ylabel('Predicted Values')
        ax2.set_title(f'Actual vs Predicted\nR² = {metrics["r2"]:.4f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Feature importance
        ax3 = fig.add_subplot(gs[1, :])
        importances = self.model.get_feature_importance()
        feature_names = self.feature_names

        # Sort features by importance
        sorted_idx = np.argsort(importances)[-20:]  # Top 20 features
        sorted_importances = importances[sorted_idx]
        sorted_features = [feature_names[i] for i in sorted_idx]

        ax3.barh(range(len(sorted_importances)), sorted_importances, color='steelblue')
        ax3.set_yticks(range(len(sorted_importances)))
        ax3.set_yticklabels(sorted_features)
        ax3.set_xlabel('Feature Importance')
        ax3.set_title(f'Top 20 Feature Importance - CatBoost')
        ax3.grid(True, alpha=0.3, axis='x')

        # 4. Error distribution
        ax4 = fig.add_subplot(gs[2, 0])
        errors = y_pred - y_test
        ax4.hist(errors, bins=30, edgecolor='black', alpha=0.7, density=True)
        ax4.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        ax4.axvline(x=np.mean(errors), color='g', linestyle='-',
                    linewidth=2, label=f'Mean: {np.mean(errors):.2f}')

        # # Add normal distribution fit
        # mu, std = norm.fit(errors)
        # xmin, xmax = ax4.get_xlim()
        # x = np.linspace(xmin, xmax, 100)
        # p = norm.pdf(x, mu, std)
        # ax4.plot(x, p, 'k', linewidth=2, label=f'Normal fit\nμ={mu:.2f}, σ={std:.2f}')

        ax4.set_xlabel('Prediction Error')
        ax4.set_ylabel('Density')
        ax4.set_title(f'Error Distribution\nStd: {np.std(errors):.2f}, MAE: {metrics["mae"]:.2f}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Metrics summary table
        ax5 = fig.add_subplot(gs[2, 1:])
        metrics_text = f"""
        CatBoost Model Performance Summary
        {'=' * 40}
        Sequence Length: {self.sequence_length}
        Target Variable: {self.target_col}

        Performance Metrics:
        --------------------
        MSE:     {metrics['mse']:.2f}
        MAE:     {metrics['mae']:.2f}
        RMSE:    {metrics['rmse']:.2f}
        R²:      {metrics['r2']:.4f}
        MAPE:    {metrics['mape']:.2f}%

        Key Hyperparameters:
        --------------------
        Depth:          {best_params.get('depth', 'N/A')}
        Learning Rate:  {best_params.get('learning_rate', 'N/A'):.5f}
        L2 Leaf Reg:    {best_params.get('l2_leaf_reg', 'N/A'):.5f}
        Border Count:   {best_params.get('border_count', 'N/A')}
        """

        ax5.text(0.05, 0.5, metrics_text, fontsize=10, family='monospace',
                 verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax5.axis('off')
        ax5.set_title('Model Summary')

        plt.suptitle(
            f'CatBoost Results for {self.target_col} Prediction\nSequence Length: {self.sequence_length}',
            fontsize=16, y=1.02)
        plt.tight_layout()

        # Save plot
        plot_filename = f'catboost_results_{self.target_col}_seq{self.sequence_length}.png'
        plt.savefig(os.path.join(self.plots_path, plot_filename), dpi=150, bbox_inches='tight')
        plt.show()

        # Save feature importance separately
        self.plot_detailed_feature_importance(importances, self.feature_names, metrics)

    def plot_detailed_feature_importance(self, importances, feature_names, metrics):
        """Plot detailed feature importance analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Feature importance by type
        lag_features = [f for f in feature_names if f.startswith('lag_')]
        rolling_features = [f for f in feature_names if f.startswith('rolling_')]
        time_features = [f for f in feature_names if f in ['day_of_week', 'month', 'year', 'is_weekend',
                                                           'month_sin', 'month_cos', 'day_sin', 'day_cos']]
        other_features = [f for f in feature_names if f not in lag_features + rolling_features + time_features]

        feature_types = ['Lag Features', 'Rolling Stats', 'Time Features', 'Other Features']
        type_counts = [len(lag_features), len(rolling_features), len(time_features), len(other_features)]

        # Get average importance by type
        def avg_importance(feature_list):
            if not feature_list:
                return 0
            indices = [feature_names.index(f) for f in feature_list]
            return np.mean([importances[i] for i in indices])

        type_importance = [
            avg_importance(lag_features),
            avg_importance(rolling_features),
            avg_importance(time_features),
            avg_importance(other_features)
        ]

        # Plot 1: Feature type importance
        x = np.arange(len(feature_types))
        width = 0.35

        axes[0, 0].bar(x - width / 2, type_counts, width, label='Count', alpha=0.7, color='blue')
        axes[0, 0].bar(x + width / 2, type_importance, width, label='Avg Importance', alpha=0.7, color='red')
        axes[0, 0].set_xlabel('Feature Type')
        axes[0, 0].set_ylabel('Count / Importance')
        axes[0, 0].set_title('Feature Type Analysis')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(feature_types, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Top 15 features
        sorted_idx = np.argsort(importances)[-15:]
        sorted_importances = importances[sorted_idx]
        sorted_features = [feature_names[i] for i in sorted_idx]

        axes[0, 1].barh(range(len(sorted_importances)), sorted_importances, color='green')
        axes[0, 1].set_yticks(range(len(sorted_importances)))
        axes[0, 1].set_yticklabels(sorted_features)
        axes[0, 1].set_xlabel('Importance')
        axes[0, 1].set_title('Top 15 Features')
        axes[0, 1].grid(True, alpha=0.3, axis='x')

        # 3. Importance distribution
        axes[1, 0].hist(importances, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=np.mean(importances), color='r', linestyle='--',
                           linewidth=2, label=f'Mean: {np.mean(importances):.3f}')
        axes[1, 0].set_xlabel('Feature Importance')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Importance Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Cumulative importance
        sorted_importances_desc = np.sort(importances)[::-1]
        cumulative_importance = np.cumsum(sorted_importances_desc) / np.sum(importances)

        axes[1, 1].plot(range(1, len(cumulative_importance) + 1), cumulative_importance,
                        'b-', linewidth=2)
        axes[1, 1].axhline(y=0.8, color='r', linestyle='--', linewidth=1, alpha=0.7)
        axes[1, 1].axhline(y=0.9, color='g', linestyle='--', linewidth=1, alpha=0.7)
        axes[1, 1].axhline(y=0.95, color='orange', linestyle='--', linewidth=1, alpha=0.7)

        # Find number of features needed for each threshold
        for threshold, color in [(0.8, 'red'), (0.9, 'green'), (0.95, 'orange')]:
            idx = np.where(cumulative_importance >= threshold)[0]
            if len(idx) > 0:
                n_features = idx[0] + 1
                axes[1, 1].scatter(n_features, threshold, color=color, s=50, zorder=5)
                axes[1, 1].text(n_features, threshold + 0.02, f'{n_features} features',
                                fontsize=8, ha='center')

        axes[1, 1].set_xlabel('Number of Features')
        axes[1, 1].set_ylabel('Cumulative Importance')
        axes[1, 1].set_title('Cumulative Feature Importance')
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(
            f'CatBoost Feature Importance Analysis\nR² = {metrics["r2"]:.4f}, RMSE = {metrics["rmse"]:.2f}',
            fontsize=14, y=1.02)
        plt.tight_layout()

        # Save detailed feature importance plot
        feature_plot_filename = f'feature_importance_catboost_{self.target_col}_seq{self.sequence_length}.png'
        plt.savefig(os.path.join(self.plots_path, feature_plot_filename), dpi=150, bbox_inches='tight')
        plt.close()

    def save_final_metrics(self, metrics_dict):
        """Save final metrics to CSV"""
        metrics_df = pd.DataFrame([metrics_dict])
        filename = f'final_metrics_{self.target_col}_seq{self.sequence_length}.csv'
        metrics_df.to_csv(os.path.join(self.metrics_path, filename), index=False)
        print(f"Final metrics saved to {filename}")

    def run_experiment(self, df, n_trials=50):
        """Run complete CatBoost experiment"""
        print(f"\n{'=' * 80}")
        print(f"CATBOOST EXPERIMENT - SEQUENCE LENGTH: {self.sequence_length}")
        print(f"{'=' * 80}")

        print(f"\n{'=' * 60}")
        print(f"PREDICTING {self.target_col.upper()} with CatBoost (Seq Len: {self.sequence_length})")
        print(f"{'=' * 60}")

        best_params, test_metrics, X_test, y_test, y_pred = self.optimize_hyperparameters(
            df, n_trials=n_trials
        )

        self.plot_results(X_test, y_test, y_pred, test_metrics, best_params)

        # Save metrics
        metrics = {
            'model': 'CatBoost',
            'target': self.target_col,
            'sequence_length': self.sequence_length,
            'mse': float(test_metrics['mse']),
            'mae': float(test_metrics['mae']),
            'rmse': float(test_metrics['rmse']),
            'r2': float(test_metrics['r2']),
            'mape': float(test_metrics['mape']) if not np.isnan(test_metrics['mape']) else None,
            'best_params': str(best_params),
            'timestamp': datetime.now().isoformat()
        }
        self.save_final_metrics(metrics)

        print(f"\n{'=' * 60}")
        print(f"CatBoost SUMMARY FOR {self.target_col.upper()} - SEQUENCE LENGTH: {self.sequence_length}")
        print(f"{'=' * 60}")
        print(f"MSE: {test_metrics['mse']:.2f}")
        print(f"MAE: {test_metrics['mae']:.2f}")
        print(f"RMSE: {test_metrics['rmse']:.2f}")
        print(f"R²: {test_metrics['r2']:.4f}")
        print(f"MAPE: {test_metrics['mape']:.2f}%")

        return {
            'best_params': best_params,
            'test_metrics': test_metrics,
            'predictions': y_pred,
            'actuals': y_test
        }


def run_catboost_complete_experiment(df, sequence_lengths=[1, 3, 12], n_trials=30):
    """Run complete CatBoost experiments for residential and commercial usage"""
    all_results = {}

    for seq_len in sequence_lengths:
        print(f"\n{'=' * 80}")
        print(f"STARTING CATBOOST EXPERIMENTS FOR SEQUENCE LENGTH: {seq_len}")
        print(f"{'=' * 80}")

        # Residential Usage
        residential_predictor = CatBoostPredictor(
            sequence_length=seq_len,
            target_col='Residential_Usage'
        )
        res_results = residential_predictor.run_experiment(df, n_trials=n_trials)

        # Commercial Usage
        commercial_predictor = CatBoostPredictor(
            sequence_length=seq_len,
            target_col='Commercial_Usage'
        )
        com_results = commercial_predictor.run_experiment(df, n_trials=n_trials)

        all_results[seq_len] = {
            'residential': res_results,
            'commercial': com_results
        }

        # Print comparison for this sequence length
        print(f"\n{'=' * 60}")
        print(f"CATBOOST COMPARISON FOR SEQUENCE LENGTH: {seq_len}")
        print(f"{'=' * 60}")
        print(f"{'Metric':<15} {'Residential':<15} {'Commercial':<15}")
        print(f"{'-' * 45}")
        print(f"{'MSE':<15} {res_results['test_metrics']['mse']:<15.2f} {com_results['test_metrics']['mse']:<15.2f}")
        print(f"{'MAE':<15} {res_results['test_metrics']['mae']:<15.2f} {com_results['test_metrics']['mae']:<15.2f}")
        print(f"{'RMSE':<15} {res_results['test_metrics']['rmse']:<15.2f} {com_results['test_metrics']['rmse']:<15.2f}")
        print(f"{'R²':<15} {res_results['test_metrics']['r2']:<15.4f} {com_results['test_metrics']['r2']:<15.4f}")
        print(f"{'MAPE':<15} {res_results['test_metrics']['mape']:<15.2f} {com_results['test_metrics']['mape']:<15.2f}")

    # Create summary comparison
    create_summary_comparison(all_results)

    return all_results


def create_summary_comparison(all_results):
    """Create summary comparison across all sequence lengths"""
    print(f"\n{'=' * 80}")
    print("CATBOOST MODELS COMPARISON")
    print(f"{'=' * 80}")

    summary_data = []

    for seq_len, results in all_results.items():
        # Residential
        summary_data.append({
            'Model': 'CatBoost',
            'Sequence_Length': seq_len,
            'Target': 'Residential',
            'MSE': results['residential']['test_metrics']['mse'],
            'MAE': results['residential']['test_metrics']['mae'],
            'RMSE': results['residential']['test_metrics']['rmse'],
            'R2': results['residential']['test_metrics']['r2'],
            'MAPE': results['residential']['test_metrics']['mape']
        })

        # Commercial
        summary_data.append({
            'Model': 'CatBoost',
            'Sequence_Length': seq_len,
            'Target': 'Commercial',
            'MSE': results['commercial']['test_metrics']['mse'],
            'MAE': results['commercial']['test_metrics']['mae'],
            'RMSE': results['commercial']['test_metrics']['rmse'],
            'R2': results['commercial']['test_metrics']['r2'],
            'MAPE': results['commercial']['test_metrics']['mape']
        })

    summary_df = pd.DataFrame(summary_data)

    # Save summary
    summary_path = "../../../results/metrics/CatBoost/CatBoost_summary_comparison.csv"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"\nCatBoost summary comparison saved to: {summary_path}")

    # Print summary table
    print(f"\n{'Seq Len':<10} {'Target':<12} {'MSE':<10} {'MAE':<10} {'RMSE':<10} {'R²':<8} {'MAPE':<10}")
    print(f"{'-' * 80}")

    for idx, row in summary_df.iterrows():
        print(f"{row['Sequence_Length']:<10} {row['Target']:<12} "
              f"{row['MSE']:<10.2f} {row['MAE']:<10.2f} {row['RMSE']:<10.2f} "
              f"{row['R2']:<8.4f} {row['MAPE']:<10.2f}")

    return summary_df


# Main execution for standalone use
if __name__ == "__main__":
    # Load processed data
    print("Loading processed data...")
    df = pd.read_csv("../../data/processed/dataset.csv", index_col='Date', parse_dates=True)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    # Run CatBoost experiments
    sequence_lengths = [1, 3, 12]
    n_trials = 30  # Number of Optuna trials for hyperparameter optimization

    all_results = run_catboost_complete_experiment(
        df, sequence_lengths=sequence_lengths, n_trials=n_trials
    )

    print(f"\n{'=' * 80}")
    print("CATBOOST EXPERIMENTS COMPLETE!")
    print(f"{'=' * 80}")
    print("\nAll CatBoost results have been saved")
