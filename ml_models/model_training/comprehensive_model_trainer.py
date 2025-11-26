#!/usr/bin/env python3
"""
comprehensive_model_trainer.py - Comprehensive ML model training and comparison

This script trains ALL major machine learning algorithms on cybersecurity datasets
and provides detailed comparisons between them.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

# Import ALL major ML algorithms
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    ExtraTreesClassifier, BaggingClassifier, VotingClassifier
)
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron
)
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveModelTrainer:
    """Comprehensive ML model trainer with ALL algorithms"""
    
    def __init__(self, processed_data_dir="./processed_datasets", models_dir="./models", results_dir="./results"):
        self.processed_data_dir = Path(processed_data_dir)
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Comprehensive model configurations
        self.models = {
            # Ensemble Methods
            'random_forest': {
                'classifier': RandomForestClassifier,
                'params': {'n_estimators': [100, 200], 'max_depth': [10, 20], 'random_state': [42]},
                'default_params': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
            },
            'gradient_boosting': {
                'classifier': GradientBoostingClassifier,
                'params': {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.2], 'max_depth': [6, 8], 'random_state': [42]},
                'default_params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'random_state': 42}
            },
            'ada_boost': {
                'classifier': AdaBoostClassifier,
                'params': {'n_estimators': [50, 100], 'learning_rate': [0.5, 1.0], 'random_state': [42]},
                'default_params': {'n_estimators': 50, 'learning_rate': 1.0, 'random_state': 42}
            },
            'extra_trees': {
                'classifier': ExtraTreesClassifier,
                'params': {'n_estimators': [100, 200], 'max_depth': [10, 20], 'random_state': [42]},
                'default_params': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
            },
            'bagging': {
                'classifier': BaggingClassifier,
                'params': {'n_estimators': [10, 20], 'random_state': [42]},
                'default_params': {'n_estimators': 10, 'random_state': 42}
            },
            
            # Linear Models
            'logistic_regression': {
                'classifier': LogisticRegression,
                'params': {'C': [0.1, 1.0, 10.0], 'max_iter': [1000], 'random_state': [42]},
                'default_params': {'C': 1.0, 'max_iter': 1000, 'random_state': 42}
            },
            'ridge_classifier': {
                'classifier': RidgeClassifier,
                'params': {'alpha': [0.1, 1.0, 10.0], 'random_state': [42]},
                'default_params': {'alpha': 1.0, 'random_state': 42}
            },
            'sgd_classifier': {
                'classifier': SGDClassifier,
                'params': {'loss': ['hinge', 'log'], 'alpha': [0.0001, 0.001], 'random_state': [42]},
                'default_params': {'loss': 'hinge', 'alpha': 0.0001, 'random_state': 42}
            },
            'perceptron': {
                'classifier': Perceptron,
                'params': {'max_iter': [1000], 'random_state': [42]},
                'default_params': {'max_iter': 1000, 'random_state': 42}
            },
            
            # Support Vector Machines
            'svm_rbf': {
                'classifier': SVC,
                'params': {'C': [1.0, 10.0], 'kernel': ['rbf'], 'random_state': [42]},
                'default_params': {'C': 1.0, 'kernel': 'rbf', 'random_state': 42}
            },
            'svm_linear': {
                'classifier': SVC,
                'params': {'C': [1.0, 10.0], 'kernel': ['linear'], 'random_state': [42]},
                'default_params': {'C': 1.0, 'kernel': 'linear', 'random_state': 42}
            },
            'svm_poly': {
                'classifier': SVC,
                'params': {'C': [1.0, 10.0], 'kernel': ['poly'], 'degree': [2, 3], 'random_state': [42]},
                'default_params': {'C': 1.0, 'kernel': 'poly', 'degree': 2, 'random_state': 42}
            },
            'linear_svc': {
                'classifier': LinearSVC,
                'params': {'C': [1.0, 10.0], 'random_state': [42]},
                'default_params': {'C': 1.0, 'random_state': 42}
            },
            'nu_svc': {
                'classifier': NuSVC,
                'params': {'nu': [0.1, 0.5], 'kernel': ['rbf'], 'random_state': [42]},
                'default_params': {'nu': 0.5, 'kernel': 'rbf', 'random_state': 42}
            },
            
            # Neural Networks
            'mlp_classifier': {
                'classifier': MLPClassifier,
                'params': {'hidden_layer_sizes': [(100,), (100, 50)], 'activation': ['relu', 'tanh'], 'max_iter': [500], 'random_state': [42]},
                'default_params': {'hidden_layer_sizes': (100, 50), 'activation': 'relu', 'max_iter': 500, 'random_state': 42}
            },
            
            # Decision Trees
            'decision_tree': {
                'classifier': DecisionTreeClassifier,
                'params': {'max_depth': [10, 20], 'random_state': [42]},
                'default_params': {'max_depth': 10, 'random_state': 42}
            },
            
            # Naive Bayes
            'gaussian_nb': {
                'classifier': GaussianNB,
                'params': {},
                'default_params': {}
            },
            'multinomial_nb': {
                'classifier': MultinomialNB,
                'params': {'alpha': [0.1, 1.0]},
                'default_params': {'alpha': 1.0}
            },
            'bernoulli_nb': {
                'classifier': BernoulliNB,
                'params': {'alpha': [0.1, 1.0]},
                'default_params': {'alpha': 1.0}
            },
            
            # Nearest Neighbors
            'knn': {
                'classifier': KNeighborsClassifier,
                'params': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
                'default_params': {'n_neighbors': 5, 'weights': 'uniform'}
            },
            
            # Discriminant Analysis
            'lda': {
                'classifier': LinearDiscriminantAnalysis,
                'params': {},
                'default_params': {}
            },
            'qda': {
                'classifier': QuadraticDiscriminantAnalysis,
                'params': {},
                'default_params': {}
            }
        }
        
        self.results = {}
        self.comparison_data = []
    
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
        """Train a specific model with comprehensive evaluation"""
        logger.info(f"Training {model_name} on {dataset_name} dataset...")
        
        model_config = self.models[model_name]
        
        try:
            if use_grid_search and model_config['params']:
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
                try:
                    y_pred_proba = model.predict_proba(X_test)
                except:
                    y_pred_proba = None
            
            # Calculate comprehensive metrics
            metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Training time (simplified)
            import time
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            result = {
                'model_name': model_name,
                'dataset_name': dataset_name,
                'model': model,
                'best_params': best_params,
                'metrics': metrics,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time': training_time,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            return None
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, y_pred_proba=None):
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
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    # Multi-class classification
                    metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
                    metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovo', average='macro')
            except:
                metrics['roc_auc'] = None
                metrics['roc_auc_ovr'] = None
                metrics['roc_auc_ovo'] = None
        
        return metrics
    
    def train_all_models_on_dataset(self, dataset_name, use_grid_search=False):
        """Train ALL models on a specific dataset"""
        logger.info(f"Training ALL models on {dataset_name} dataset...")
        
        # Load dataset
        X_train, X_test, y_train, y_test, metadata = self.load_processed_dataset(dataset_name)
        
        dataset_results = {}
        successful_models = 0
        
        # Train each model
        for model_name in self.models.keys():
            try:
                result = self.train_model(
                    model_name, X_train, y_train, X_test, y_test, 
                    dataset_name, use_grid_search
                )
                
                if result:
                    dataset_results[model_name] = result
                    successful_models += 1
                    
                    # Add to comparison data
                    comparison_entry = {
                        'dataset': dataset_name,
                        'model': model_name,
                        'accuracy': result['metrics']['accuracy'],
                        'f1_macro': result['metrics']['f1_macro'],
                        'precision_macro': result['metrics']['precision_macro'],
                        'recall_macro': result['metrics']['recall_macro'],
                        'cv_mean': result['cv_mean'],
                        'cv_std': result['cv_std'],
                        'training_time': result['training_time']
                    }
                    
                    if result['metrics'].get('roc_auc'):
                        comparison_entry['roc_auc'] = result['metrics']['roc_auc']
                    elif result['metrics'].get('roc_auc_ovr'):
                        comparison_entry['roc_auc'] = result['metrics']['roc_auc_ovr']
                    
                    self.comparison_data.append(comparison_entry)
                    
                    # Save model
                    self.save_model(result, dataset_name)
                    
            except Exception as e:
                logger.error(f"Error training {model_name} on {dataset_name}: {e}")
        
        logger.info(f"Successfully trained {successful_models}/{len(self.models)} models on {dataset_name}")
        self.results[dataset_name] = dataset_results
        return dataset_results
    
    def save_model(self, result, dataset_name):
        """Save trained model and results"""
        model_name = result['model_name']
        
        # Save model
        model_path = self.models_dir / f"{dataset_name}_{model_name}_comprehensive.pkl"
        joblib.dump(result['model'], model_path)
        
        # Save results
        results_path = self.models_dir / f"{dataset_name}_{model_name}_comprehensive_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        results_dict = {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'best_params': result['best_params'],
            'metrics': result['metrics'],
            'cv_mean': float(result['cv_mean']),
            'cv_std': float(result['cv_std']),
            'training_time': float(result['training_time'])
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
    
    def train_all_datasets(self, use_grid_search=False):
        """Train ALL models on ALL available datasets"""
        logger.info("Training ALL models on ALL datasets...")
        
        # Get available datasets
        available_datasets = [d.name for d in self.processed_data_dir.iterdir() if d.is_dir() and d.name != 'preprocessors']
        
        logger.info(f"Found datasets: {available_datasets}")
        logger.info(f"Training {len(self.models)} models on {len(available_datasets)} datasets")
        
        for dataset_name in available_datasets:
            try:
                self.train_all_models_on_dataset(dataset_name, use_grid_search)
            except Exception as e:
                logger.error(f"Error processing dataset {dataset_name}: {e}")
        
        # Generate comprehensive comparison report
        self.generate_comprehensive_comparison_report()
    
    def generate_comprehensive_comparison_report(self):
        """Generate comprehensive comparison report"""
        logger.info("Generating comprehensive comparison report...")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(self.comparison_data)
        
        # Save detailed comparison data
        comparison_path = self.results_dir / "comprehensive_model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        # Generate summary statistics
        summary = {
            'total_models_trained': len(comparison_df),
            'total_datasets': comparison_df['dataset'].nunique(),
            'total_algorithms': comparison_df['model'].nunique(),
            'best_overall_accuracy': comparison_df['accuracy'].max(),
            'worst_overall_accuracy': comparison_df['accuracy'].min(),
            'average_accuracy': comparison_df['accuracy'].mean(),
            'best_model_by_dataset': {},
            'model_rankings': {}
        }
        
        # Best model per dataset
        for dataset in comparison_df['dataset'].unique():
            dataset_data = comparison_df[comparison_df['dataset'] == dataset]
            best_model = dataset_data.loc[dataset_data['accuracy'].idxmax()]
            summary['best_model_by_dataset'][dataset] = {
                'model': best_model['model'],
                'accuracy': best_model['accuracy'],
                'f1_macro': best_model['f1_macro']
            }
        
        # Overall model rankings
        model_avg_performance = comparison_df.groupby('model').agg({
            'accuracy': 'mean',
            'f1_macro': 'mean',
            'training_time': 'mean'
        }).sort_values('accuracy', ascending=False)
        
        summary['model_rankings'] = model_avg_performance.to_dict()
        
        # Save summary report
        summary_path = self.results_dir / "comprehensive_comparison_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Comprehensive comparison report saved: {comparison_path}")
        logger.info(f"Summary report saved: {summary_path}")
        
        # Print comprehensive summary
        self.print_comprehensive_summary(comparison_df, summary)
    
    def print_comprehensive_summary(self, comparison_df, summary):
        """Print comprehensive summary"""
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE ML MODEL COMPARISON SUMMARY")
        logger.info("="*80)
        
        logger.info(f"\nOVERALL STATISTICS:")
        logger.info(f"  Total Models Trained: {summary['total_models_trained']}")
        logger.info(f"  Total Datasets: {summary['total_datasets']}")
        logger.info(f"  Total Algorithms: {summary['total_algorithms']}")
        logger.info(f"  Best Overall Accuracy: {summary['best_overall_accuracy']:.4f}")
        logger.info(f"  Worst Overall Accuracy: {summary['worst_overall_accuracy']:.4f}")
        logger.info(f"  Average Accuracy: {summary['average_accuracy']:.4f}")
        
        logger.info(f"\nBEST MODEL PER DATASET:")
        for dataset, info in summary['best_model_by_dataset'].items():
            logger.info(f"  {dataset.upper()}: {info['model']} (Accuracy: {info['accuracy']:.4f}, F1: {info['f1_macro']:.4f})")
        
        logger.info(f"\nTOP 10 MODELS BY AVERAGE ACCURACY:")
        top_models = comparison_df.groupby('model')['accuracy'].mean().sort_values(ascending=False).head(10)
        for i, (model, accuracy) in enumerate(top_models.items(), 1):
            logger.info(f"  {i:2d}. {model:<20} {accuracy:.4f}")
        
        logger.info(f"\nFASTEST MODELS (by training time):")
        fastest_models = comparison_df.groupby('model')['training_time'].mean().sort_values().head(5)
        for i, (model, time) in enumerate(fastest_models.items(), 1):
            logger.info(f"  {i:2d}. {model:<20} {time:.4f}s")

def main():
    """Main function to train ALL models"""
    logger.info("Starting comprehensive ML model training...")
    
    trainer = ComprehensiveModelTrainer()
    
    # Train ALL models on ALL datasets
    trainer.train_all_datasets(use_grid_search=False)
    
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE ML TRAINING COMPLETED!")
    logger.info("="*80)
    logger.info("Next steps:")
    logger.info("1. Review comprehensive comparison results")
    logger.info("2. Analyze model performance patterns")
    logger.info("3. Select best models for production deployment")

if __name__ == "__main__":
    main()
