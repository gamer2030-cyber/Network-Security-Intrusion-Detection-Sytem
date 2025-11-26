#!/usr/bin/env python3
"""
enhanced_model_trainer.py - Enhanced ML model training for cybersecurity datasets

This script trains multiple ML models on the processed cybersecurity datasets
with proper evaluation metrics and model persistence.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedModelTrainer:
    def __init__(self, processed_data_dir="./processed_datasets", models_dir="./models"):
        self.processed_data_dir = Path(processed_data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Model configurations
        self.models = {
            'random_forest': {
                'classifier': RandomForestClassifier,
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'random_state': [42]
                },
                'default_params': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
            },
            'gradient_boosting': {
                'classifier': GradientBoostingClassifier,
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.1, 0.2],
                    'max_depth': [6, 8],
                    'random_state': [42]
                },
                'default_params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'random_state': 42}
            },
            'svm': {
                'classifier': SVC,
                'params': {
                    'C': [1.0, 10.0],
                    'kernel': ['rbf', 'linear'],
                    'random_state': [42]
                },
                'default_params': {'C': 1.0, 'kernel': 'rbf', 'random_state': 42}
            },
            'neural_network': {
                'classifier': MLPClassifier,
                'params': {
                    'hidden_layer_sizes': [(100,), (100, 50), (100, 50, 25)],
                    'activation': ['relu', 'tanh'],
                    'max_iter': [500, 1000],
                    'random_state': [42]
                },
                'default_params': {'hidden_layer_sizes': (100, 50), 'activation': 'relu', 'max_iter': 500, 'random_state': 42}
            }
        }
        
        self.results = {}
    
    def load_processed_dataset(self, dataset_name):
        """Load processed dataset"""
        dataset_path = self.processed_data_dir / dataset_name
        
        logger.info(f"Loading processed {dataset_name} dataset...")
        
        # Load training data
        X_train = pd.read_csv(dataset_path / f"{dataset_name}_X_train.csv")
        y_train = pd.read_csv(dataset_path / f"{dataset_name}_y_train.csv")['label']
        
        # Load test data
        X_test = pd.read_csv(dataset_path / f"{dataset_name}_X_test.csv")
        y_test = pd.read_csv(dataset_path / f"{dataset_name}_y_test.csv")['label']
        
        # Load metadata
        with open(dataset_path / f"{dataset_name}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded {dataset_name}: Train {X_train.shape}, Test {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, metadata
    
    def train_model(self, model_name, X_train, y_train, X_test, y_test, dataset_name, use_grid_search=False):
        """Train a specific model"""
        logger.info(f"Training {model_name} on {dataset_name} dataset...")
        
        model_config = self.models[model_name]
        
        if use_grid_search:
            # Use GridSearchCV for hyperparameter tuning
            grid_search = GridSearchCV(
                model_config['classifier'](),
                model_config['params'],
                cv=3,
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            logger.info(f"Best parameters: {best_params}")
        else:
            # Use default parameters
            model = model_config['classifier'](**model_config['default_params'])
            model.fit(X_train, y_train)
            best_params = model_config['default_params']
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        result = {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'model': model,
            'best_params': best_params,
            'metrics': metrics,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return result
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Add ROC AUC if probabilities are available
        if y_pred_proba is not None and len(np.unique(y_true)) > 2:
            try:
                metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
            except:
                metrics['roc_auc_ovr'] = None
        
        return metrics
    
    def save_model(self, result, dataset_name):
        """Save trained model and results"""
        model_name = result['model_name']
        
        # Save model
        model_path = self.models_dir / f"{dataset_name}_{model_name}.pkl"
        joblib.dump(result['model'], model_path)
        
        # Save results
        results_path = self.models_dir / f"{dataset_name}_{model_name}_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        results_dict = {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'best_params': result['best_params'],
            'metrics': result['metrics'],
            'cv_mean': float(result['cv_mean']),
            'cv_std': float(result['cv_std'])
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Model saved: {model_path}")
        logger.info(f"Results saved: {results_path}")
    
    def train_all_models_on_dataset(self, dataset_name, use_grid_search=False):
        """Train all models on a specific dataset"""
        logger.info(f"Training all models on {dataset_name} dataset...")
        
        # Load dataset
        X_train, X_test, y_train, y_test, metadata = self.load_processed_dataset(dataset_name)
        
        dataset_results = {}
        
        # Train each model
        for model_name in self.models.keys():
            try:
                result = self.train_model(
                    model_name, X_train, y_train, X_test, y_test, 
                    dataset_name, use_grid_search
                )
                dataset_results[model_name] = result
                
                # Save model
                self.save_model(result, dataset_name)
                
            except Exception as e:
                logger.error(f"Error training {model_name} on {dataset_name}: {e}")
        
        self.results[dataset_name] = dataset_results
        return dataset_results
    
    def train_all_datasets(self, use_grid_search=False):
        """Train models on all available datasets"""
        logger.info("Training models on all datasets...")
        
        # Get available datasets
        available_datasets = [d.name for d in self.processed_data_dir.iterdir() if d.is_dir() and d.name != 'preprocessors']
        
        logger.info(f"Found datasets: {available_datasets}")
        
        for dataset_name in available_datasets:
            try:
                self.train_all_models_on_dataset(dataset_name, use_grid_search)
            except Exception as e:
                logger.error(f"Error processing dataset {dataset_name}: {e}")
        
        # Generate summary report
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        logger.info("Generating summary report...")
        
        report_path = self.models_dir / "training_summary_report.json"
        
        summary = {
            'datasets': {},
            'best_models': {},
            'overall_summary': {}
        }
        
        for dataset_name, dataset_results in self.results.items():
            summary['datasets'][dataset_name] = {}
            
            best_model = None
            best_accuracy = 0
            
            for model_name, result in dataset_results.items():
                metrics = result['metrics']
                summary['datasets'][dataset_name][model_name] = {
                    'accuracy': metrics['accuracy'],
                    'f1_macro': metrics['f1_macro'],
                    'cv_mean': result['cv_mean'],
                    'cv_std': result['cv_std']
                }
                
                if metrics['accuracy'] > best_accuracy:
                    best_accuracy = metrics['accuracy']
                    best_model = model_name
            
            summary['best_models'][dataset_name] = {
                'model': best_model,
                'accuracy': best_accuracy
            }
        
        # Overall summary
        all_accuracies = []
        for dataset_results in self.results.values():
            for result in dataset_results.values():
                all_accuracies.append(result['metrics']['accuracy'])
        
        summary['overall_summary'] = {
            'total_models_trained': len(all_accuracies),
            'average_accuracy': np.mean(all_accuracies),
            'best_accuracy': np.max(all_accuracies),
            'worst_accuracy': np.min(all_accuracies)
        }
        
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary report saved: {report_path}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("TRAINING SUMMARY REPORT")
        logger.info("="*60)
        
        for dataset_name, best_model_info in summary['best_models'].items():
            logger.info(f"\n{dataset_name.upper()}:")
            logger.info(f"  Best Model: {best_model_info['model']}")
            logger.info(f"  Accuracy: {best_model_info['accuracy']:.4f}")
        
        logger.info(f"\nOVERALL SUMMARY:")
        logger.info(f"  Total Models Trained: {summary['overall_summary']['total_models_trained']}")
        logger.info(f"  Average Accuracy: {summary['overall_summary']['average_accuracy']:.4f}")
        logger.info(f"  Best Accuracy: {summary['overall_summary']['best_accuracy']:.4f}")
        logger.info(f"  Worst Accuracy: {summary['overall_summary']['worst_accuracy']:.4f}")

def main():
    """Main function to train all models"""
    logger.info("Starting enhanced model training...")
    
    trainer = EnhancedModelTrainer()
    
    # Train models on all datasets (without grid search for speed)
    trainer.train_all_datasets(use_grid_search=False)
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETED!")
    logger.info("="*60)
    logger.info("Next steps:")
    logger.info("1. Review training results in models/ directory")
    logger.info("2. Test adversarial robustness")
    logger.info("3. Deploy best performing models")

if __name__ == "__main__":
    main()
