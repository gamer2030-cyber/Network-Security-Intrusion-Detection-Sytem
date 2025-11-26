#!/usr/bin/env python3
"""
model_comparison_dashboard.py - Comprehensive ML model comparison dashboard

This script creates detailed visualizations and comparisons of all trained ML models
across different cybersecurity datasets.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelComparisonDashboard:
    """Comprehensive model comparison dashboard"""
    
    def __init__(self, results_dir="./results", models_dir="./models", processed_data_dir="./processed_datasets"):
        self.results_dir = Path(results_dir)
        self.models_dir = Path(models_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.dashboard_dir = self.results_dir / "dashboard"
        self.dashboard_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_comparison_data(self):
        """Load comprehensive comparison data"""
        comparison_path = self.results_dir / "comprehensive_model_comparison.csv"
        
        if not comparison_path.exists():
            logger.error("Comprehensive comparison data not found. Please run comprehensive_model_trainer.py first.")
            return None
        
        comparison_df = pd.read_csv(comparison_path)
        logger.info(f"Loaded comparison data: {comparison_df.shape}")
        
        return comparison_df
    
    def create_accuracy_comparison_plot(self, comparison_df):
        """Create accuracy comparison visualization"""
        logger.info("Creating accuracy comparison plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('ML Model Accuracy Comparison Across Datasets', fontsize=20, fontweight='bold')
        
        # 1. Overall accuracy distribution
        ax1 = axes[0, 0]
        comparison_df.boxplot(column='accuracy', by='model', ax=ax1, rot=45)
        ax1.set_title('Accuracy Distribution by Model')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Accuracy')
        
        # 2. Accuracy by dataset
        ax2 = axes[0, 1]
        dataset_accuracy = comparison_df.pivot(index='model', columns='dataset', values='accuracy')
        sns.heatmap(dataset_accuracy, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax2)
        ax2.set_title('Accuracy Heatmap by Dataset')
        
        # 3. Top 10 models by average accuracy
        ax3 = axes[1, 0]
        top_models = comparison_df.groupby('model')['accuracy'].mean().sort_values(ascending=True).tail(10)
        top_models.plot(kind='barh', ax=ax3, color='skyblue')
        ax3.set_title('Top 10 Models by Average Accuracy')
        ax3.set_xlabel('Average Accuracy')
        
        # 4. Accuracy vs Training Time
        ax4 = axes[1, 1]
        scatter = ax4.scatter(comparison_df['training_time'], comparison_df['accuracy'], 
                            c=comparison_df['f1_macro'], cmap='viridis', alpha=0.7, s=100)
        ax4.set_xlabel('Training Time (seconds)')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Accuracy vs Training Time (colored by F1-score)')
        plt.colorbar(scatter, ax=ax4, label='F1 Macro Score')
        
        plt.tight_layout()
        plt.savefig(self.dashboard_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Accuracy comparison plot saved")
    
    def create_performance_metrics_plot(self, comparison_df):
        """Create comprehensive performance metrics visualization"""
        logger.info("Creating performance metrics plot...")
        
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        fig.suptitle('Comprehensive Performance Metrics Comparison', fontsize=20, fontweight='bold')
        
        metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'cv_mean']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//3, i%3]
            
            # Create box plot for each metric
            comparison_df.boxplot(column=metric, by='model', ax=ax, rot=45)
            ax.set_title(f'{metric.replace("_", " ").title()} Distribution')
            ax.set_xlabel('Model')
            ax.set_ylabel(metric.replace("_", " ").title())
        
        # 6. Model complexity vs performance
        ax6 = axes[1, 2]
        model_complexity = comparison_df.groupby('model').agg({
            'training_time': 'mean',
            'accuracy': 'mean',
            'f1_macro': 'mean'
        })
        
        scatter = ax6.scatter(model_complexity['training_time'], model_complexity['accuracy'],
                            c=model_complexity['f1_macro'], cmap='plasma', s=200, alpha=0.7)
        
        # Add model labels
        for model in model_complexity.index:
            ax6.annotate(model, (model_complexity.loc[model, 'training_time'], 
                               model_complexity.loc[model, 'accuracy']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax6.set_xlabel('Average Training Time (seconds)')
        ax6.set_ylabel('Average Accuracy')
        ax6.set_title('Model Complexity vs Performance')
        plt.colorbar(scatter, ax=ax6, label='Average F1 Macro')
        
        plt.tight_layout()
        plt.savefig(self.dashboard_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Performance metrics plot saved")
    
    def create_dataset_specific_analysis(self, comparison_df):
        """Create dataset-specific analysis plots"""
        logger.info("Creating dataset-specific analysis...")
        
        datasets = comparison_df['dataset'].unique()
        n_datasets = len(datasets)
        
        fig, axes = plt.subplots(n_datasets, 2, figsize=(16, 6*n_datasets))
        if n_datasets == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Dataset-Specific Model Performance Analysis', fontsize=20, fontweight='bold')
        
        for i, dataset in enumerate(datasets):
            dataset_data = comparison_df[comparison_df['dataset'] == dataset]
            
            # Top models for this dataset
            ax1 = axes[i, 0]
            top_models_dataset = dataset_data.nlargest(10, 'accuracy')
            bars = ax1.barh(range(len(top_models_dataset)), top_models_dataset['accuracy'])
            ax1.set_yticks(range(len(top_models_dataset)))
            ax1.set_yticklabels(top_models_dataset['model'], fontsize=10)
            ax1.set_xlabel('Accuracy')
            ax1.set_title(f'Top 10 Models for {dataset.upper()}')
            
            # Color bars by F1 score
            colors = plt.cm.viridis(top_models_dataset['f1_macro'])
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # Performance correlation
            ax2 = axes[i, 1]
            scatter = ax2.scatter(dataset_data['precision_macro'], dataset_data['recall_macro'],
                                c=dataset_data['accuracy'], cmap='coolwarm', s=100, alpha=0.7)
            ax2.set_xlabel('Precision (Macro)')
            ax2.set_ylabel('Recall (Macro)')
            ax2.set_title(f'Precision vs Recall for {dataset.upper()}')
            plt.colorbar(scatter, ax=ax2, label='Accuracy')
            
            # Add model labels for top performers
            top_5 = dataset_data.nlargest(5, 'accuracy')
            for _, row in top_5.iterrows():
                ax2.annotate(row['model'], (row['precision_macro'], row['recall_macro']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.dashboard_dir / 'dataset_specific_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Dataset-specific analysis plot saved")
    
    def create_model_family_comparison(self, comparison_df):
        """Create comparison by model families"""
        logger.info("Creating model family comparison...")
        
        # Define model families
        model_families = {
            'Ensemble': ['random_forest', 'gradient_boosting', 'ada_boost', 'extra_trees', 'bagging'],
            'Linear': ['logistic_regression', 'ridge_classifier', 'sgd_classifier', 'perceptron'],
            'SVM': ['svm_rbf', 'svm_linear', 'svm_poly', 'linear_svc', 'nu_svc'],
            'Neural Network': ['mlp_classifier'],
            'Tree': ['decision_tree'],
            'Naive Bayes': ['gaussian_nb', 'multinomial_nb', 'bernoulli_nb'],
            'Neighbors': ['knn'],
            'Discriminant': ['lda', 'qda']
        }
        
        # Add family information to dataframe
        comparison_df['family'] = 'Other'
        for family, models in model_families.items():
            comparison_df.loc[comparison_df['model'].isin(models), 'family'] = family
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Model Family Performance Comparison', fontsize=20, fontweight='bold')
        
        # 1. Family accuracy comparison
        ax1 = axes[0, 0]
        family_accuracy = comparison_df.groupby('family')['accuracy'].agg(['mean', 'std']).sort_values('mean', ascending=True)
        family_accuracy['mean'].plot(kind='barh', ax=ax1, xerr=family_accuracy['std'], capsize=5)
        ax1.set_title('Average Accuracy by Model Family')
        ax1.set_xlabel('Accuracy')
        
        # 2. Family performance distribution
        ax2 = axes[0, 1]
        comparison_df.boxplot(column='accuracy', by='family', ax=ax2, rot=45)
        ax2.set_title('Accuracy Distribution by Family')
        ax2.set_xlabel('Model Family')
        
        # 3. Family training time comparison
        ax3 = axes[1, 0]
        family_time = comparison_df.groupby('family')['training_time'].mean().sort_values(ascending=True)
        family_time.plot(kind='bar', ax=ax3, color='orange')
        ax3.set_title('Average Training Time by Family')
        ax3.set_ylabel('Training Time (seconds)')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Family performance vs complexity
        ax4 = axes[1, 1]
        family_stats = comparison_df.groupby('family').agg({
            'accuracy': 'mean',
            'training_time': 'mean',
            'f1_macro': 'mean'
        })
        
        scatter = ax4.scatter(family_stats['training_time'], family_stats['accuracy'],
                            c=family_stats['f1_macro'], cmap='viridis', s=300, alpha=0.7)
        
        for family in family_stats.index:
            ax4.annotate(family, (family_stats.loc[family, 'training_time'], 
                                family_stats.loc[family, 'accuracy']),
                        xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
        
        ax4.set_xlabel('Average Training Time (seconds)')
        ax4.set_ylabel('Average Accuracy')
        ax4.set_title('Family Performance vs Complexity')
        plt.colorbar(scatter, ax=ax4, label='Average F1 Macro')
        
        plt.tight_layout()
        plt.savefig(self.dashboard_dir / 'model_family_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Model family comparison plot saved")
    
    def create_summary_table(self, comparison_df):
        """Create comprehensive summary table"""
        logger.info("Creating summary table...")
        
        # Calculate comprehensive statistics
        summary_stats = comparison_df.groupby('model').agg({
            'accuracy': ['mean', 'std', 'min', 'max'],
            'f1_macro': ['mean', 'std'],
            'precision_macro': ['mean', 'std'],
            'recall_macro': ['mean', 'std'],
            'cv_mean': ['mean', 'std'],
            'training_time': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
        
        # Sort by average accuracy
        summary_stats = summary_stats.sort_values('accuracy_mean', ascending=False)
        
        # Save to CSV
        summary_stats.to_csv(self.dashboard_dir / 'model_performance_summary.csv')
        
        # Create HTML table
        html_table = summary_stats.to_html(classes='table table-striped table-hover')
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML Model Performance Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .table {{ border-collapse: collapse; width: 100%; }}
                .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                .table th {{ background-color: #f2f2f2; font-weight: bold; }}
                .table tr:nth-child(even) {{ background-color: #f9f9f9; }}
                h1 {{ color: #333; text-align: center; }}
            </style>
        </head>
        <body>
            <h1>Comprehensive ML Model Performance Summary</h1>
            {html_table}
        </body>
        </html>
        """
        
        with open(self.dashboard_dir / 'model_performance_summary.html', 'w') as f:
            f.write(html_content)
        
        logger.info("Summary table saved")
    
    def generate_dashboard(self):
        """Generate complete comparison dashboard"""
        logger.info("Generating comprehensive model comparison dashboard...")
        
        # Load comparison data
        comparison_df = self.load_comparison_data()
        if comparison_df is None:
            return
        
        # Create all visualizations
        self.create_accuracy_comparison_plot(comparison_df)
        self.create_performance_metrics_plot(comparison_df)
        self.create_dataset_specific_analysis(comparison_df)
        self.create_model_family_comparison(comparison_df)
        self.create_summary_table(comparison_df)
        
        # Create main dashboard HTML
        self.create_main_dashboard_html(comparison_df)
        
        logger.info(f"Dashboard generated successfully in: {self.dashboard_dir}")
    
    def create_main_dashboard_html(self, comparison_df):
        """Create main dashboard HTML file"""
        logger.info("Creating main dashboard HTML...")
        
        # Get top performers
        top_models = comparison_df.groupby('model')['accuracy'].mean().sort_values(ascending=False).head(10)
        best_by_dataset = comparison_df.loc[comparison_df.groupby('dataset')['accuracy'].idxmax()]
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML-IDS-IPS Model Comparison Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; border-left: 4px solid #3498db; padding-left: 10px; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
                .stat-number {{ font-size: 2em; font-weight: bold; margin-bottom: 5px; }}
                .stat-label {{ font-size: 0.9em; opacity: 0.9; }}
                .top-models {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .model-item {{ display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #bdc3c7; }}
                .model-item:last-child {{ border-bottom: none; }}
                .model-name {{ font-weight: bold; color: #2c3e50; }}
                .model-score {{ color: #27ae60; font-weight: bold; }}
                .image-gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
                .image-card {{ text-align: center; }}
                .image-card img {{ max-width: 100%; height: auto; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .footer {{ text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 ML-IDS-IPS Comprehensive Model Comparison Dashboard</h1>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">{len(comparison_df)}</div>
                        <div class="stat-label">Models Trained</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{comparison_df['dataset'].nunique()}</div>
                        <div class="stat-label">Datasets</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{comparison_df['model'].nunique()}</div>
                        <div class="stat-label">Algorithms</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{comparison_df['accuracy'].max():.3f}</div>
                        <div class="stat-label">Best Accuracy</div>
                    </div>
                </div>
                
                <h2>🏆 Top 10 Models by Average Accuracy</h2>
                <div class="top-models">
        """
        
        for i, (model, accuracy) in enumerate(top_models.items(), 1):
            html_content += f"""
                    <div class="model-item">
                        <span class="model-name">{i}. {model.replace('_', ' ').title()}</span>
                        <span class="model-score">{accuracy:.4f}</span>
                    </div>
            """
        
        html_content += """
                </div>
                
                <h2>🎯 Best Model per Dataset</h2>
                <div class="top-models">
        """
        
        for _, row in best_by_dataset.iterrows():
            html_content += f"""
                    <div class="model-item">
                        <span class="model-name">{row['dataset'].upper()}: {row['model'].replace('_', ' ').title()}</span>
                        <span class="model-score">{row['accuracy']:.4f}</span>
                    </div>
            """
        
        html_content += """
                </div>
                
                <h2>📊 Performance Visualizations</h2>
                <div class="image-gallery">
                    <div class="image-card">
                        <h3>Accuracy Comparison</h3>
                        <img src="accuracy_comparison.png" alt="Accuracy Comparison">
                    </div>
                    <div class="image-card">
                        <h3>Performance Metrics</h3>
                        <img src="performance_metrics.png" alt="Performance Metrics">
                    </div>
                    <div class="image-card">
                        <h3>Dataset-Specific Analysis</h3>
                        <img src="dataset_specific_analysis.png" alt="Dataset Analysis">
                    </div>
                    <div class="image-card">
                        <h3>Model Family Comparison</h3>
                        <img src="model_family_comparison.png" alt="Family Comparison">
                    </div>
                </div>
                
                <h2>📋 Detailed Reports</h2>
                <ul>
                    <li><a href="model_performance_summary.html">Detailed Performance Summary Table</a></li>
                    <li><a href="model_performance_summary.csv">Performance Data (CSV)</a></li>
                </ul>
                
                <div class="footer">
                    <p>Generated by ML-IDS-IPS Comprehensive Model Comparison Dashboard</p>
                    <p>Total Models Analyzed: {len(comparison_df)} | Datasets: {comparison_df['dataset'].nunique()} | Algorithms: {comparison_df['model'].nunique()}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(self.dashboard_dir / 'index.html', 'w') as f:
            f.write(html_content)
        
        logger.info("Main dashboard HTML created")

def main():
    """Main function to generate dashboard"""
    logger.info("Starting comprehensive model comparison dashboard generation...")
    
    dashboard = ModelComparisonDashboard()
    dashboard.generate_dashboard()
    
    logger.info("\n" + "="*80)
    logger.info("DASHBOARD GENERATION COMPLETED!")
    logger.info("="*80)
    logger.info("Dashboard files saved in: results/dashboard/")
    logger.info("Open 'results/dashboard/index.html' in your browser to view the dashboard")

if __name__ == "__main__":
    main()
