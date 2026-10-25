import glob
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

# Muted colors for non-highlight models
base_colors = sns.color_palette("muted", 5)

# Highlight color for CNN-BiLSTM
highlight_color = sns.color_palette("tab10")[3]  # strong red

# Build palette in model order
custom_palette = [
    base_colors[0],  # SA_CNN
    base_colors[1],  # SA_BiLSTM
    highlight_color,  # CNN_BiLSTM (highlight)
    base_colors[2],  # LGBM
    base_colors[3],  # CatBoost
    base_colors[4],  # XGBoost
]

sns.set_palette(custom_palette)

# Base paths
base_results_path = "../../results/metrics/"
models = ['SA_CNN', 'SA_BiLSTM', 'CNN_BiLSTM', 'LGBM', 'CatBoost', 'XGBoost']
model_labels = ['CNN', 'BiLSTM', 'CNN-BiLSTM', 'LGBM', 'CatBoost', 'XGBoost']
targets = ['Residential', 'Commercial']
sequence_lengths = [1, 3, 12]
# metrics = ['mse', 'mae', 'rmse', 'r2', 'mape']
metrics = ['MSE', 'MAE', 'RMSE', 'R2', 'MAPE']
metric_labels = ['MSE', 'MAE', 'RMSE', 'R²', 'MAPE']


def load_all_metrics():
    """Load all metrics from all models"""
    all_data = []

    for model in models:
        model_path = os.path.join(base_results_path, model)

        if not os.path.exists(model_path):
            print(f"Warning: {model_path} does not exist")
            continue

        # Load summary file if exists
        summary_files = glob.glob(os.path.join(model_path, '*summary_comparison.csv'))
        if summary_files:
            try:
                summary_df = pd.read_csv(summary_files[0])
                summary_df['model'] = model
                all_data.append(summary_df)
                print(f"Loaded summary for {model}")
            except Exception as e:
                print(f"Error loading summary for {model}: {e}")

        # Also load individual final metrics files
        for seq_len in sequence_lengths:
            seq_path = os.path.join(model_path, f'seq_{seq_len}')
            if not os.path.exists(seq_path):
                continue

            for target in targets:
                metric_file = os.path.join(seq_path, f'final_metrics_{target}_Usage_seq{seq_len}.csv')
                if os.path.exists(metric_file):
                    try:
                        df = pd.read_csv(metric_file)
                        df['model'] = model
                        df['sequence_length'] = seq_len
                        df['target'] = f'{target}_Usage'
                        all_data.append(df)
                    except Exception as e:
                        print(f"Error loading {metric_file}: {e}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)

        # Clean up column names
        if 'Model' in combined_df.columns:
            combined_df = combined_df.drop(columns=['Model'])
        if 'Target' in combined_df.columns:
            combined_df['target'] = combined_df['Target']
            combined_df = combined_df.drop(columns=['Target'])
        if 'Sequence_Length' in combined_df.columns:
            combined_df['sequence_length'] = combined_df['Sequence_Length']
            combined_df = combined_df.drop(columns=['Sequence_Length'])

        # Standardize target names
        combined_df['target'] = combined_df['target'].replace({
            'Residential': 'Residential_Usage',
            'Commercial': 'Commercial_Usage',
            'Residential_Usage': 'Residential_Usage',
            'Commercial_Usage': 'Commercial_Usage'
        })

        # Extract target type
        combined_df['target_type'] = combined_df['target'].apply(
            lambda x: 'Residential' if 'Residential' in str(x) else 'Commercial'
        )

        # Standardize model names
        model_mapping = {
            'SA_CNN': 'CNN',
            'SA_BiLSTM': 'BiLSTM',
            'CNN_BiLSTM': 'CNN-BiLSTM',
            'LGBM': 'LGBM',
            'CatBoost': 'CatBoost',
            'XGBoost': 'XGBoost'
        }
        combined_df['model_name'] = combined_df['model'].map(model_mapping)

        return combined_df
    else:
        print("No data loaded!")
        return pd.DataFrame()


def create_comparison_plots(df):
    """Create comparison plots for each model"""

    # Filter out any rows with missing metrics
    for metric in metrics:
        if metric in df.columns:
            df = df[~df[metric].isna()]

    # Plot 1: Comparison for each model (separate plots for each metric)
    create_model_metric_comparison(df)

    # Plot 2: Best of each model comparison
    create_best_model_comparison(df)


def create_model_metric_comparison(df):
    """Create separate plots for each model showing performance on each metric"""

    print("\n" + "=" * 80)
    print("CREATING MODEL-METRIC COMPARISON PLOTS")
    print("=" * 80)

    # Create a figure for each model
    for model in model_labels:
        model_df = df[df['model_name'] == model].copy()

        if model_df.empty:
            print(f"No data for model: {model}")
            continue

        # Create figure with subplots for each metric
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
            if idx >= len(axes):
                break

            ax = axes[idx]

            # Prepare data for this metric
            plot_data = []

            for target_type in ['Residential', 'Commercial']:
                for seq_len in sequence_lengths:
                    subset = model_df[
                        (model_df['target_type'] == target_type) &
                        (model_df['sequence_length'] == seq_len)
                        ]

                    if not subset.empty and metric in subset.columns:
                        metric_values = subset[metric].values
                        if len(metric_values) > 0:
                            # Take the mean if multiple values
                            value = np.mean(metric_values)
                            plot_data.append({
                                'target': target_type,
                                'sequence_length': seq_len,
                                'metric': metric_label,
                                'value': value
                            })

            if plot_data:
                plot_df = pd.DataFrame(plot_data)

                # Create bar plot
                x = np.arange(len(sequence_lengths))
                width = 0.35

                for i, target_type in enumerate(['Residential', 'Commercial']):
                    target_data = plot_df[plot_df['target'] == target_type]
                    if not target_data.empty:
                        # Sort by sequence length
                        target_data = target_data.sort_values('sequence_length')
                        values = target_data['value'].values

                        bars = ax.bar(x + (i - 0.5) * width, values, width,
                                      label=target_type, alpha=0.8)

                        # Add value labels
                        for bar, val in zip(bars, values):
                            if not np.isnan(val):
                                # Format based on metric
                                if metric in ['r2']:
                                    label = f'{val:.3f}'
                                elif metric in ['mape']:
                                    label = f'{val:.1f}%'
                                else:
                                    label = f'{val:.1f}'

                                height = bar.get_height()
                                if metric in ['mse', 'mae', 'rmse']:
                                    # Put label above bar for error metrics
                                    ax.text(bar.get_x() + bar.get_width() / 2., height * 1.02,
                                            label, ha='center', va='bottom', fontsize=8)
                                else:
                                    # Put label inside bar for other metrics
                                    ax.text(bar.get_x() + bar.get_width() / 2., height / 2,
                                            label, ha='center', va='center', fontsize=8,
                                            color='white' if val > np.max(values) / 2 else 'black')

                ax.set_xlabel('Sequence Length')
                ax.set_ylabel(metric_label)
                ax.set_title(f'{model}: {metric_label}')
                ax.set_xticks(x)
                ax.set_xticklabels([f'Seq {sl}' for sl in sequence_lengths])
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')

                # Set y-axis scale for better visualization
                if metric in ['mse', 'mae', 'rmse']:
                    ax.set_yscale('log')

        # Hide empty subplots
        for idx in range(len(plot_data) // 2 + 1 if plot_data else len(axes), len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'{model} Performance Comparison\nAcross Sequence Lengths and Targets',
                     fontsize=16, y=1.02)
        plt.tight_layout()

        # Save plot
        plot_filename = f'{model}_performance_comparison.png'
        plots_dir = '../../results/analysis_plots/'
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, plot_filename), dpi=150, bbox_inches='tight')
        print(f"Saved plot: {plot_filename}")
        plt.show()


def create_best_model_comparison(df):
    """Create comparison plot of best performance for each model"""

    print("\n" + "=" * 80)
    print("CREATING BEST MODEL COMPARISON PLOTS")
    print("=" * 80)

    # For each model and metric, find the best value (lowest for error metrics, highest for R²)
    best_performance = {}

    for model in model_labels:
        model_df = df[df['model_name'] == model].copy()

        if model_df.empty:
            continue

        model_best = {}

        for metric, metric_label in zip(metrics, metric_labels):
            if metric not in model_df.columns:
                continue

            # Filter out NaN values
            valid_data = model_df[~model_df[metric].isna()]

            if valid_data.empty:
                continue

            if metric == 'R2':
                # For R², we want the highest value (closest to 1)
                best_idx = valid_data[metric].idxmax()
                best_value = valid_data.loc[best_idx, metric]
                best_seq = valid_data.loc[best_idx, 'sequence_length']
                best_target = valid_data.loc[best_idx, 'target_type']
            elif metric == 'MAPE':
                # For MAPE, we want the lowest value
                best_idx = valid_data[metric].idxmin()
                best_value = valid_data.loc[best_idx, metric]
                best_seq = valid_data.loc[best_idx, 'sequence_length']
                best_target = valid_data.loc[best_idx, 'target_type']
            else:
                # For MSE, MAE, RMSE, we want the lowest value
                best_idx = valid_data[metric].idxmin()
                best_value = valid_data.loc[best_idx, metric]
                best_seq = valid_data.loc[best_idx, 'sequence_length']
                best_target = valid_data.loc[best_idx, 'target_type']

            model_best[metric] = {
                'value': best_value,
                'sequence_length': best_seq,
                'target': best_target,
                'metric_label': metric_label
            }

        best_performance[model] = model_best

    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Collect best values for this metric across all models
        model_names = []
        best_values = []
        bar_colors = []
        annotations = []

        for model in model_labels:
            if model in best_performance and metric in best_performance[model]:
                data = best_performance[model][metric]
                model_names.append(model)
                best_values.append(data['value'])

                # Color code by target type
                if data['target'] == 'Residential':
                    bar_colors.append('lightblue')
                else:
                    bar_colors.append('lightcoral')

                # Annotation with sequence length
                annotations.append(f"Seq {data['sequence_length']}")

        if model_names:
            # Create bar plot
            x = np.arange(len(model_names))
            bars = ax.bar(x, best_values, color=bar_colors, alpha=0.8, edgecolor='black')

            # Add value labels
            for bar, val, ann in zip(bars, best_values, annotations):
                height = bar.get_height()

                # Format value based on metric
                if metric in ['r2']:
                    label = f'{val:.3f}'
                elif metric in ['mape']:
                    label = f'{val:.1f}%'
                else:
                    label = f'{val:.1f}'

                # Position label
                if metric in ['mse', 'mae', 'rmse']:
                    # Error metrics: label above
                    ax.text(bar.get_x() + bar.get_width() / 2., height * 1.02,
                            label, ha='center', va='bottom', fontsize=9)
                else:
                    # Other metrics: label inside
                    y_pos = height / 2 if height > 0 else height
                    ax.text(bar.get_x() + bar.get_width() / 2., y_pos,
                            label, ha='center', va='center', fontsize=9,
                            color='white' if abs(height) > abs(np.max(best_values)) / 2 else 'black')

                # Add sequence length annotation below
                ax.text(bar.get_x() + bar.get_width() / 2., -0.02 if height >= 0 else height - 0.02,
                        ann, ha='center', va='top', fontsize=8, style='italic')

            ax.set_xlabel('Model')
            ax.set_ylabel(metric_label)
            ax.set_title(f'Best {metric_label} Comparison\n(Colored by target: Residential=blue, Commercial=red)')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')

            # Set y-axis scale for error metrics
            if metric in ['mse', 'mae', 'rmse']:
                ax.set_yscale('log')

    # Hide empty subplots
    for idx in range(len(metrics), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Best Performance Comparison Across All Models', fontsize=16, y=1.02)
    plt.tight_layout()

    # Save plot
    plots_dir = '../../results/analysis_plots/'
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'best_model_comparison.png'), dpi=150, bbox_inches='tight')
    print("Saved plot: best_model_comparison.png")
    plt.show()

    # Also create a summary table
    create_summary_table(best_performance)


def create_summary_table(best_performance):
    """Create a summary table of best performances"""

    print("\n" + "=" * 80)
    print("BEST PERFORMANCE SUMMARY TABLE")
    print("=" * 80)

    # Create summary DataFrame
    summary_data = []

    for model in model_labels:
        if model not in best_performance:
            continue

        model_data = {'Model': model}

        for metric, metric_label in zip(metrics, metric_labels):
            if metric in best_performance[model]:
                data = best_performance[model][metric]
                if metric in ['r2']:
                    model_data[f'{metric_label}'] = f"{data['value']:.4f}"
                elif metric in ['mape']:
                    model_data[f'{metric_label}'] = f"{data['value']:.2f}%"
                else:
                    model_data[f'{metric_label}'] = f"{data['value']:.2f}"
                model_data[f'{metric_label}_Seq'] = data['sequence_length']
                model_data[f'{metric_label}_Target'] = data['target'][:3]  # First 3 chars
            else:
                model_data[f'{metric_label}'] = 'N/A'
                model_data[f'{metric_label}_Seq'] = 'N/A'
                model_data[f'{metric_label}_Target'] = 'N/A'

        summary_data.append(model_data)

    if summary_data:
        summary_df = pd.DataFrame(summary_data)

        # Display table
        print("\nBest Performance Summary:")
        print("-" * 100)

        # Print header
        header = "Model".ljust(15)
        for metric_label in metric_labels:
            header += f"{metric_label:>12}"
        print(header)
        print("-" * 100)

        # Print data rows
        for _, row in summary_df.iterrows():
            line = f"{row['Model']:15}"
            for metric_label in metric_labels:
                if metric_label in row:
                    line += f"{row[metric_label]:>12}"
                else:
                    line += f"{'N/A':>12}"
            print(line)

        print("-" * 100)

        # Save to CSV
        summary_csv_path = '../../results/analysis_plots/best_performance_summary.csv'
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\nSummary saved to: {summary_csv_path}")

        # Create a visual summary table
        create_visual_summary_table(summary_df)


def create_visual_summary_table(summary_df):
    """Create a visual table plot of the summary"""

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')

    # Prepare data for table
    table_data = []
    headers = ['Model'] + metric_labels

    for _, row in summary_df.iterrows():
        table_row = [row['Model']]
        for metric_label in metric_labels:
            if metric_label in row:
                table_row.append(row[metric_label])
            else:
                table_row.append('N/A')
        table_data.append(table_row)

    # Create table
    table = ax.table(cellText=table_data,
                     colLabels=headers,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15] + [0.17] * len(metric_labels))

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4F81BD')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        if i % 2 == 0:
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#DCE6F1')

    plt.title('Best Performance Summary Across All Models', fontsize=14, pad=20)

    # Save the table
    plots_dir = '../../results/analysis_plots/'
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'best_performance_summary_table.png'),
                dpi=150, bbox_inches='tight')
    print("Saved visual summary table: best_performance_summary_table.png")
    plt.show()


def create_comprehensive_overview(df):
    """Create a comprehensive overview plot showing all models and metrics"""

    print("\n" + "=" * 80)
    print("CREATING COMPREHENSIVE OVERVIEW")
    print("=" * 80)

    # Prepare data for heatmap
    heatmap_data = []

    for model in model_labels:
        model_df = df[df['model_name'] == model].copy()

        if model_df.empty:
            continue

        for metric, metric_label in zip(metrics, metric_labels):
            if metric not in model_df.columns:
                continue

            # Get all values for this model and metric
            values = model_df[metric].dropna().values

            if len(values) > 0:
                # For error metrics, take the minimum (best)
                # For R², take the maximum (best)
                if metric == 'r2':
                    best_value = np.max(values)
                elif metric == 'mape':
                    best_value = np.min(values)
                else:
                    best_value = np.min(values)

                heatmap_data.append({
                    'model': model,
                    'metric': metric_label,
                    'value': best_value
                })

    if not heatmap_data:
        print("No data for heatmap")
        return

    heatmap_df = pd.DataFrame(heatmap_data)

    # Pivot for heatmap
    pivot_df = heatmap_df.pivot(index='model', columns='metric', values='value')

    # Reorder models and metrics
    pivot_df = pivot_df.reindex(index=model_labels, columns=metric_labels)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create mask for better value display
    cmap = plt.cm.RdYlGn_r  # Reverse so green=good, red=bad

    # Normalize differently for each metric
    im = ax.imshow(pivot_df.values, cmap=cmap, aspect='auto')

    # Add text annotations
    for i in range(len(model_labels)):
        for j in range(len(metric_labels)):
            if model_labels[i] in pivot_df.index and metric_labels[j] in pivot_df.columns:
                value = pivot_df.loc[model_labels[i], metric_labels[j]]
                if not np.isnan(value):
                    # Format based on metric
                    if metric_labels[j] == 'R²':
                        text = f'{value:.3f}'
                        color = 'white' if value < 0.5 else 'black'
                    elif metric_labels[j] == 'MAPE':
                        text = f'{value:.1f}%'
                        color = 'white' if value > 50 else 'black'
                    else:
                        text = f'{value:.1f}'
                        color = 'white' if value > np.nanmax(pivot_df.values[:, j]) / 2 else 'black'

                    ax.text(j, i, text, ha='center', va='center',
                            color=color, fontsize=10, fontweight='bold')

    # Set labels
    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_yticks(np.arange(len(model_labels)))
    ax.set_xticklabels(metric_labels)
    ax.set_yticklabels(model_labels)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Performance (lower is better for MSE/MAE/RMSE/MAPE,\nhigher is better for R²)',
                       rotation=-90, va="bottom")

    ax.set_title("Comprehensive Model Performance Overview\n(Green = Better, Red = Worse)",
                 fontsize=14, pad=20)

    plt.tight_layout()

    # Save heatmap
    plots_dir = '../../results/analysis_plots/'
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'comprehensive_performance_heatmap.png'),
                dpi=150, bbox_inches='tight')
    print("Saved comprehensive heatmap: comprehensive_performance_heatmap.png")
    plt.show()


def analyze_trends(df):
    """Analyze trends across sequence lengths"""

    print("\n" + "=" * 80)
    print("ANALYZING TRENDS ACROSS SEQUENCE LENGTHS")
    print("=" * 80)

    # Create trend plots for each metric
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Group by model and sequence length
        for model in model_labels:
            model_data = []

            for seq_len in sequence_lengths:
                subset = df[
                    (df['model_name'] == model) &
                    (df['sequence_length'] == seq_len)
                    ]

                if not subset.empty and metric in subset.columns:
                    values = subset[metric].dropna().values
                    if len(values) > 0:
                        avg_value = np.mean(values)
                        model_data.append((seq_len, avg_value))

            if model_data:
                # Sort by sequence length
                model_data.sort(key=lambda x: x[0])
                seq_lengths = [x[0] for x in model_data]
                avg_values = [x[1] for x in model_data]

                # Plot line with markers
                ax.plot(seq_lengths, avg_values, marker='o', label=model, linewidth=2)

        ax.set_xlabel('Sequence Length')
        ax.set_ylabel(metric_label)
        ax.set_title(f'{metric_label} Trends')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Set log scale for error metrics
        if metric in ['mse', 'mae', 'rmse']:
            ax.set_yscale('log')

        # Set x-axis to show sequence lengths
        ax.set_xticks(sequence_lengths)

    # Hide empty subplots
    for idx in range(len(metrics), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Performance Trends Across Sequence Lengths', fontsize=16, y=1.02)
    plt.tight_layout()

    # Save trend plot
    plots_dir = '../../results/analysis_plots/'
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'performance_trends.png'), dpi=150, bbox_inches='tight')
    print("Saved trend analysis: performance_trends.png")
    plt.show()


def create_seq3_comparison(df):
    """Create comparison plots for sequence length 3"""

    print("\n" + "=" * 80)
    print("CREATING SEQUENCE LENGTH 3 COMPARISON PLOTS")
    print("=" * 80)

    # Filter data for sequence length 3
    df_seq3 = df[df['sequence_length'] == 3].copy()

    if df_seq3.empty:
        print("No data for sequence length 3")
        return

    # Create figure with subplots for each metric
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Prepare data for this metric
        plot_data = []

        for model in model_labels:
            for target_type in ['Residential', 'Commercial']:
                subset = df_seq3[
                    (df_seq3['model_name'] == model) &
                    (df_seq3['target_type'] == target_type)
                    ]

                if not subset.empty and metric in subset.columns:
                    metric_values = subset[metric].values
                    if len(metric_values) > 0:
                        # Take the mean if multiple values
                        value = np.mean(metric_values)
                        plot_data.append({
                            'model': model,
                            'target': target_type,
                            'metric': metric_label,
                            'value': value
                        })

        if plot_data:
            plot_df = pd.DataFrame(plot_data)

            # Create grouped bar plot
            x = np.arange(len(model_labels))
            width = 0.35

            for i, target_type in enumerate(['Residential', 'Commercial']):
                target_data = plot_df[plot_df['target'] == target_type]
                if not target_data.empty:
                    # Sort by model order
                    target_data = target_data.set_index('model').reindex(model_labels).reset_index()
                    values = target_data['value'].values

                    bars = ax.bar(x + (i - 0.5) * width, values, width,
                                  label=target_type, alpha=0.8,
                                  color='lightblue' if target_type == 'Residential' else 'lightcoral')

                    # Add value labels
                    for bar, val in zip(bars, values):
                        if not np.isnan(val):
                            # Format based on metric
                            if metric in ['r2']:
                                label = f'{val:.3f}'
                            elif metric in ['mape']:
                                label = f'{val:.1f}%'
                            else:
                                label = f'{val:.1f}'

                            height = bar.get_height()
                            if metric in ['mse', 'mae', 'rmse']:
                                # Put label above bar for error metrics
                                ax.text(bar.get_x() + bar.get_width() / 2., height * 1.02,
                                        label, ha='center', va='bottom', fontsize=8)
                            else:
                                # Put label inside bar for other metrics
                                y_pos = height / 2 if height > 0 else height
                                ax.text(bar.get_x() + bar.get_width() / 2., y_pos,
                                        label, ha='center', va='center', fontsize=8,
                                        color='white' if abs(height) > abs(np.max(values)) / 2 else 'black')

            ax.set_xlabel('Model')
            ax.set_ylabel(metric_label)
            ax.set_title(f'Seq Length 3: {metric_label}')
            ax.set_xticks(x)
            ax.set_xticklabels(model_labels, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

            # Set y-axis scale for better visualization
            if metric in ['mse', 'mae', 'rmse']:
                ax.set_yscale('log')

    # Hide empty subplots
    for idx in range(len(plot_data) // 2 + 1 if plot_data else len(axes), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Model Performance Comparison for Sequence Length 3\n(Residential vs Commercial)',
                 fontsize=16, y=1.02)
    plt.tight_layout()

    # Save plot
    plot_filename = 'seq3_performance_comparison.png'
    plots_dir = '../../results/analysis_plots/'
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, plot_filename), dpi=150, bbox_inches='tight')
    print(f"Saved plot: {plot_filename}")
    plt.show()

    # Also create a side-by-side comparison (one plot for Residential, one for Commercial)
    create_seq3_side_by_side(df_seq3)


def create_seq3_side_by_side(df_seq3):
    """Create side-by-side comparison plots for Residential and Commercial"""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, target_type in enumerate(['Residential', 'Commercial']):
        ax = axes[ax_idx]

        # Prepare data for this target type
        plot_data = []

        for model in model_labels:
            subset = df_seq3[
                (df_seq3['model_name'] == model) &
                (df_seq3['target_type'] == target_type)
                ]

            if not subset.empty:
                for metric, metric_label in zip(metrics, metric_labels):
                    if metric in subset.columns:
                        values = subset[metric].values
                        if len(values) > 0:
                            avg_value = np.mean(values)
                            plot_data.append({
                                'model': model,
                                'metric': metric_label,
                                'value': avg_value
                            })

        if plot_data:
            plot_df = pd.DataFrame(plot_data)

            # Pivot for grouped bar plot
            pivot_df = plot_df.pivot(index='model', columns='metric', values='value')
            pivot_df = pivot_df.reindex(index=model_labels, columns=metric_labels)

            # Create grouped bar plot WITHOUT log scale initially
            x = np.arange(len(model_labels))
            width = 0.15

            for i, metric_label in enumerate(metric_labels):
                if metric_label in pivot_df.columns:
                    values = pivot_df[metric_label].values
                    offset = (i - len(metric_labels) / 2) * width + width / 2

                    # Replace NaN with 0 for plotting
                    values_clean = np.nan_to_num(values, nan=0)

                    bars = ax.bar(x + offset, values_clean, width, label=metric_label, alpha=0.8)

                    # Add value labels
                    for bar_idx, (bar, val) in enumerate(zip(bars, values)):
                        if not np.isnan(val):
                            # Format based on metric
                            if metric_label == 'R²':
                                label = f'{val:.3f}'
                                # Position inside bar for R²
                                y_pos = val / 2 if val > 0 else val
                                color = 'white' if val > 0.5 else 'black'
                                ax.text(bar.get_x() + bar.get_width() / 2., y_pos,
                                        label, ha='center', va='center', fontsize=7,
                                        color=color)
                            elif metric_label == 'MAPE':
                                label = f'{val:.1f}%'
                                # Position above bar for MAPE
                                ax.text(bar.get_x() + bar.get_width() / 2., val * 1.02,
                                        label, ha='center', va='bottom', fontsize=7)
                            else:
                                label = f'{val:.1f}'
                                # Position above bar for error metrics
                                ax.text(bar.get_x() + bar.get_width() / 2., val * 1.02,
                                        label, ha='center', va='bottom', fontsize=7)

            ax.set_xlabel('Model')
            ax.set_ylabel('Metric Value')
            ax.set_title(f'Sequence Length 3: {target_type} Usage')
            ax.set_xticks(x)
            ax.set_xticklabels(model_labels, rotation=45, ha='right')
            ax.legend(loc='upper left', fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3, axis='y')

            # Set appropriate y-axis limits
            # Get all values for this subplot
            all_values = []
            for metric_label in metric_labels:
                if metric_label in pivot_df.columns:
                    vals = pivot_df[metric_label].values
                    vals_clean = vals[~np.isnan(vals)]
                    if len(vals_clean) > 0:
                        all_values.extend(vals_clean)

            if all_values:
                y_min = np.min(all_values)
                y_max = np.max(all_values)

                # Add some padding
                if y_min == y_max:
                    y_min -= 0.1 * abs(y_min) if y_min != 0 else -0.1
                    y_max += 0.1 * abs(y_max) if y_max != 0 else 0.1
                else:
                    padding = 0.1 * (y_max - y_min)
                    y_min -= padding
                    y_max += padding

                # Ensure y_min is not negative for error metrics (except R² which can be negative)
                if metric_label not in ['R²'] and y_min < 0:
                    y_min = 0

                ax.set_ylim(y_min, y_max)

    plt.suptitle('Detailed Model Comparison for Sequence Length 3', fontsize=16, y=1.02)
    plt.tight_layout()

    # Save plot
    plot_filename = 'seq3_detailed_comparison.png'
    plots_dir = '../../results/analysis_plots/'
    plt.savefig(os.path.join(plots_dir, plot_filename), dpi=150, bbox_inches='tight')
    print(f"Saved plot: {plot_filename}")
    plt.show()

# Main execution
if __name__ == "__main__":
    print("Loading metrics from all models...")
    df = load_all_metrics()

    if df.empty:
        print("No data loaded. Check the directory structure and file paths.")
    else:
        print(f"Loaded data with shape: {df.shape}")
        print(f"Models found: {df['model_name'].unique().tolist()}")
        print(f"Sequence lengths: {df['sequence_length'].unique().tolist()}")
        print(f"Targets: {df['target_type'].unique().tolist()}")

        # Display sample data
        print("\nSample data:")
        print(df[['model_name', 'sequence_length', 'target_type', 'mse', 'mae', 'r2']].head())

        # Create analysis plots directory
        plots_dir = '../../results/analysis_plots/'
        os.makedirs(plots_dir, exist_ok=True)

        # Create all plots
        create_comparison_plots(df)
        create_seq3_comparison(df)
        create_best_model_comparison(df)
        # create_comprehensive_overview(df)
        # analyze_trends(df)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\nAll plots saved to: {plots_dir}")
