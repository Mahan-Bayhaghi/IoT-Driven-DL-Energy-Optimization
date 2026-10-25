"""
Main execution script for Energy Efficiency Prediction using CNN-BiLSTM
"""

import argparse
from src.data_preprocessing import DataPreprocessor
from src.training.train import ModelTrainer
from src.evaluation.metrics import ModelEvaluator
from src.utils.config import load_config


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Energy Efficiency Prediction')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['preprocess', 'train', 'evaluate', 'full'],
                        default='full', help='Execution mode')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    if args.mode in ['preprocess', 'full']:
        print("Step 1: Data Preprocessing...")
        preprocessor = DataPreprocessor(config)
        datasets = preprocessor.prepare_data()
        print("Data preprocessing completed!")

    if args.mode in ['train', 'full']:
        print("Step 2: Model Training...")
        trainer = ModelTrainer(config)
        models = trainer.train_models(datasets)
        print("Model training completed!")

    if args.mode in ['evaluate', 'full']:
        print("Step 3: Model Evaluation...")
        evaluator = ModelEvaluator(config)
        results = evaluator.evaluate_models(models, datasets)
        print("Model evaluation completed!")


if __name__ == "__main__":
    main()