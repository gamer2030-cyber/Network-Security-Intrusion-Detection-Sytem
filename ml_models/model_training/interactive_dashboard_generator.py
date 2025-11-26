#!/usr/bin/env python3
"""
interactive_dashboard_generator.py - Generate highly interactive ML model comparison dashboard

This script creates a fully interactive dashboard with dynamic filtering, 
interactive charts, and real-time model comparison capabilities.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InteractiveDashboardGenerator:
    """Generate highly interactive ML model comparison dashboard"""
    
    def __init__(self, results_dir="./results", models_dir="./models", processed_data_dir="./processed_datasets"):
        self.results_dir = Path(results_dir)
        self.models_dir = Path(models_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.dashboard_dir = self.results_dir / "interactive_dashboard"
        self.dashboard_dir.mkdir(exist_ok=True)
        
    def load_comparison_data(self):
        """Load comprehensive comparison data"""
        comparison_path = self.results_dir / "comprehensive_model_comparison.csv"
        
        if not comparison_path.exists():
            logger.error("Comprehensive comparison data not found. Please run comprehensive_model_trainer.py first.")
            return None
        
        comparison_df = pd.read_csv(comparison_path)
        logger.info(f"Loaded comparison data: {comparison_df.shape}")
        
        # Add model family information
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
        
        comparison_df['family'] = 'Other'
        for family, models in model_families.items():
            comparison_df.loc[comparison_df['model'].isin(models), 'family'] = family
        
        return comparison_df
    
    def create_interactive_accuracy_chart(self, comparison_df):
        """Create interactive accuracy comparison chart"""
        logger.info("Creating interactive accuracy chart...")
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy Distribution by Model', 'Accuracy Heatmap by Dataset',
                           'Top 10 Models by Average Accuracy', 'Accuracy vs Training Time'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Box plot for accuracy distribution
        for model in comparison_df['model'].unique():
            model_data = comparison_df[comparison_df['model'] == model]['accuracy']
            fig.add_trace(
                go.Box(y=model_data, name=model, showlegend=False),
                row=1, col=1
            )
        
        # 2. Heatmap
        pivot_data = comparison_df.pivot(index='model', columns='dataset', values='accuracy')
        fig.add_trace(
            go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale='RdYlBu_r',
                showscale=True,
                text=np.round(pivot_data.values, 3),
                texttemplate="%{text}",
                textfont={"size": 10}
            ),
            row=1, col=2
        )
        
        # 3. Top 10 models bar chart
        top_models = comparison_df.groupby('model')['accuracy'].mean().sort_values(ascending=True).tail(10)
        fig.add_trace(
            go.Bar(
                y=top_models.index,
                x=top_models.values,
                orientation='h',
                marker_color='lightblue',
                text=np.round(top_models.values, 3),
                textposition='auto'
            ),
            row=2, col=1
        )
        
        # 4. Scatter plot: Accuracy vs Training Time
        fig.add_trace(
            go.Scatter(
                x=comparison_df['training_time'],
                y=comparison_df['accuracy'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=comparison_df['f1_macro'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="F1 Macro Score")
                ),
                text=comparison_df['model'],
                hovertemplate='<b>%{text}</b><br>' +
                            'Accuracy: %{y:.3f}<br>' +
                            'Training Time: %{x:.3f}s<br>' +
                            'F1 Macro: %{marker.color:.3f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Interactive ML Model Performance Analysis",
            height=800,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Model", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_xaxes(title_text="Dataset", row=1, col=2)
        fig.update_yaxes(title_text="Model", row=1, col=2)
        fig.update_xaxes(title_text="Average Accuracy", row=2, col=1)
        fig.update_yaxes(title_text="Model", row=2, col=1)
        fig.update_xaxes(title_text="Training Time (seconds)", row=2, col=2)
        fig.update_yaxes(title_text="Accuracy", row=2, col=2)
        
        return fig
    
    def create_interactive_performance_metrics(self, comparison_df):
        """Create interactive performance metrics dashboard"""
        logger.info("Creating interactive performance metrics...")
        
        metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'cv_mean']
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f'{metric.replace("_", " ").title()} Distribution' for metric in metrics] + ['Model Complexity vs Performance'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Create box plots for each metric
        for i, metric in enumerate(metrics):
            row = i // 3 + 1
            col = i % 3 + 1
            
            for model in comparison_df['model'].unique():
                model_data = comparison_df[comparison_df['model'] == model][metric]
                fig.add_trace(
                    go.Box(
                        y=model_data, 
                        name=model, 
                        showlegend=False,
                        boxpoints='outliers'
                    ),
                    row=row, col=col
                )
        
        # Model complexity vs performance scatter plot
        model_stats = comparison_df.groupby('model').agg({
            'accuracy': 'mean',
            'training_time': 'mean',
            'f1_macro': 'mean'
        })
        
        fig.add_trace(
            go.Scatter(
                x=model_stats['training_time'],
                y=model_stats['accuracy'],
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=model_stats['f1_macro'],
                    colorscale='Plasma',
                    showscale=True,
                    colorbar=dict(title="Average F1 Macro")
                ),
                text=model_stats.index,
                textposition="top center",
                hovertemplate='<b>%{text}</b><br>' +
                            'Avg Accuracy: %{y:.3f}<br>' +
                            'Avg Training Time: %{x:.3f}s<br>' +
                            'Avg F1 Macro: %{marker.color:.3f}<extra></extra>'
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            title="Interactive Performance Metrics Dashboard",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_interactive_dataset_analysis(self, comparison_df):
        """Create interactive dataset-specific analysis"""
        logger.info("Creating interactive dataset analysis...")
        
        datasets = comparison_df['dataset'].unique()
        n_datasets = len(datasets)
        
        fig = make_subplots(
            rows=n_datasets, cols=2,
            subplot_titles=[f'Top Models for {dataset.upper()}' for dataset in datasets] + 
                          [f'Precision vs Recall for {dataset.upper()}' for dataset in datasets],
            specs=[[{"secondary_y": False}, {"secondary_y": False}] for _ in range(n_datasets)]
        )
        
        for i, dataset in enumerate(datasets):
            dataset_data = comparison_df[comparison_df['dataset'] == dataset]
            
            # Top models bar chart
            top_models = dataset_data.nlargest(10, 'accuracy')
            fig.add_trace(
                go.Bar(
                    y=list(range(len(top_models))),
                    x=top_models['accuracy'],
                    orientation='h',
                    marker=dict(
                        color=top_models['f1_macro'],
                        colorscale='Viridis',
                        showscale=True if i == 0 else False
                    ),
                    text=top_models['model'],
                    textposition='auto',
                    hovertemplate='<b>%{text}</b><br>' +
                                'Accuracy: %{x:.3f}<br>' +
                                'F1 Macro: %{marker.color:.3f}<extra></extra>'
                ),
                row=i+1, col=1
            )
            
            # Precision vs Recall scatter
            fig.add_trace(
                go.Scatter(
                    x=dataset_data['precision_macro'],
                    y=dataset_data['recall_macro'],
                    mode='markers+text',
                    marker=dict(
                        size=12,
                        color=dataset_data['accuracy'],
                        colorscale='RdBu',
                        showscale=True if i == 0 else False
                    ),
                    text=dataset_data['model'],
                    textposition="top center",
                    hovertemplate='<b>%{text}</b><br>' +
                                'Precision: %{x:.3f}<br>' +
                                'Recall: %{y:.3f}<br>' +
                                'Accuracy: %{marker.color:.3f}<extra></extra>'
                ),
                row=i+1, col=2
            )
        
        fig.update_layout(
            title="Interactive Dataset-Specific Analysis",
            height=400 * n_datasets,
            showlegend=False
        )
        
        return fig
    
    def create_interactive_model_family_comparison(self, comparison_df):
        """Create interactive model family comparison"""
        logger.info("Creating interactive model family comparison...")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Accuracy by Family', 'Accuracy Distribution by Family',
                           'Training Time by Family', 'Family Performance vs Complexity'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Average accuracy by family
        family_accuracy = comparison_df.groupby('family')['accuracy'].agg(['mean', 'std']).sort_values('mean')
        fig.add_trace(
            go.Bar(
                y=family_accuracy.index,
                x=family_accuracy['mean'],
                error_x=dict(type='data', array=family_accuracy['std']),
                orientation='h',
                marker_color='lightblue',
                text=np.round(family_accuracy['mean'], 3),
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # 2. Box plot by family
        for family in comparison_df['family'].unique():
            family_data = comparison_df[comparison_df['family'] == family]['accuracy']
            fig.add_trace(
                go.Box(
                    y=family_data,
                    name=family,
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. Training time by family
        family_time = comparison_df.groupby('family')['training_time'].mean().sort_values()
        fig.add_trace(
            go.Bar(
                x=family_time.index,
                y=family_time.values,
                marker_color='orange',
                text=np.round(family_time.values, 3),
                textposition='auto'
            ),
            row=2, col=1
        )
        
        # 4. Family performance vs complexity
        family_stats = comparison_df.groupby('family').agg({
            'accuracy': 'mean',
            'training_time': 'mean',
            'f1_macro': 'mean'
        })
        
        fig.add_trace(
            go.Scatter(
                x=family_stats['training_time'],
                y=family_stats['accuracy'],
                mode='markers+text',
                marker=dict(
                    size=20,
                    color=family_stats['f1_macro'],
                    colorscale='Viridis',
                    showscale=True
                ),
                text=family_stats.index,
                textposition="top center",
                hovertemplate='<b>%{text}</b><br>' +
                            'Avg Accuracy: %{y:.3f}<br>' +
                            'Avg Training Time: %{x:.3f}s<br>' +
                            'Avg F1 Macro: %{marker.color:.3f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Interactive Model Family Comparison",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_interactive_model_selector(self, comparison_df):
        """Create interactive model selector and comparison tool"""
        logger.info("Creating interactive model selector...")
        
        # Create a comprehensive comparison table
        model_summary = comparison_df.groupby('model').agg({
            'accuracy': ['mean', 'std', 'min', 'max'],
            'f1_macro': ['mean', 'std'],
            'precision_macro': ['mean', 'std'],
            'recall_macro': ['mean', 'std'],
            'cv_mean': ['mean', 'std'],
            'training_time': ['mean', 'std'],
            'family': 'first'
        }).round(4)
        
        # Flatten column names
        model_summary.columns = ['_'.join(col).strip() for col in model_summary.columns]
        model_summary = model_summary.sort_values('accuracy_mean', ascending=False)
        
        # Create interactive table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Model', 'Family', 'Avg Accuracy', 'Std Accuracy', 'Min Accuracy', 'Max Accuracy',
                       'Avg F1', 'Avg Precision', 'Avg Recall', 'Avg CV Score', 'Avg Training Time'],
                fill_color='paleturquoise',
                align='left',
                font=dict(size=12)
            ),
            cells=dict(
                values=[
                    model_summary.index,
                    model_summary['family_first'],
                    np.round(model_summary['accuracy_mean'], 3),
                    np.round(model_summary['accuracy_std'], 3),
                    np.round(model_summary['accuracy_min'], 3),
                    np.round(model_summary['accuracy_max'], 3),
                    np.round(model_summary['f1_macro_mean'], 3),
                    np.round(model_summary['precision_macro_mean'], 3),
                    np.round(model_summary['recall_macro_mean'], 3),
                    np.round(model_summary['cv_mean_mean'], 3),
                    np.round(model_summary['training_time_mean'], 3)
                ],
                fill_color='lavender',
                align='left',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(
            title="Interactive Model Performance Comparison Table",
            height=600
        )
        
        return fig
    
    def generate_interactive_dashboard(self):
        """Generate complete interactive dashboard"""
        logger.info("Generating interactive dashboard...")
        
        # Load comparison data
        comparison_df = self.load_comparison_data()
        if comparison_df is None:
            return
        
        # Create all interactive charts
        accuracy_chart = self.create_interactive_accuracy_chart(comparison_df)
        performance_chart = self.create_interactive_performance_metrics(comparison_df)
        dataset_chart = self.create_interactive_dataset_analysis(comparison_df)
        family_chart = self.create_interactive_model_family_comparison(comparison_df)
        selector_chart = self.create_interactive_model_selector(comparison_df)
        
        # Save individual charts
        accuracy_chart.write_html(self.dashboard_dir / 'interactive_accuracy_chart.html')
        performance_chart.write_html(self.dashboard_dir / 'interactive_performance_metrics.html')
        dataset_chart.write_html(self.dashboard_dir / 'interactive_dataset_analysis.html')
        family_chart.write_html(self.dashboard_dir / 'interactive_family_comparison.html')
        selector_chart.write_html(self.dashboard_dir / 'interactive_model_selector.html')
        
        # Create main interactive dashboard
        self.create_main_interactive_dashboard(comparison_df, accuracy_chart, performance_chart, 
                                             dataset_chart, family_chart, selector_chart)
        
        logger.info(f"Interactive dashboard generated successfully in: {self.dashboard_dir}")
    
    def create_main_interactive_dashboard(self, comparison_df, accuracy_chart, performance_chart, 
                                        dataset_chart, family_chart, selector_chart):
        """Create main interactive dashboard HTML"""
        logger.info("Creating main interactive dashboard HTML...")
        
        # Get top performers
        top_models = comparison_df.groupby('model')['accuracy'].mean().sort_values(ascending=False).head(10)
        best_by_dataset = comparison_df.loc[comparison_df.groupby('dataset')['accuracy'].idxmax()]
        
        # Create comprehensive statistics
        stats = {
            'total_models': len(comparison_df),
            'total_datasets': comparison_df['dataset'].nunique(),
            'total_algorithms': comparison_df['model'].nunique(),
            'best_accuracy': comparison_df['accuracy'].max(),
            'avg_accuracy': comparison_df['accuracy'].mean(),
            'fastest_model': comparison_df.loc[comparison_df['training_time'].idxmin(), 'model'],
            'fastest_time': comparison_df['training_time'].min()
        }
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>🤖 ML-IDS-IPS Interactive Model Comparison Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    color: #333;
                }}
                
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                
                .header {{
                    background: rgba(255, 255, 255, 0.95);
                    border-radius: 15px;
                    padding: 30px;
                    margin-bottom: 30px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                
                .header h1 {{
                    color: #2c3e50;
                    font-size: 2.5em;
                    margin-bottom: 10px;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
                }}
                
                .header p {{
                    color: #7f8c8d;
                    font-size: 1.2em;
                }}
                
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }}
                
                .stat-card {{
                    background: rgba(255, 255, 255, 0.95);
                    border-radius: 15px;
                    padding: 25px;
                    text-align: center;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                }}
                
                .stat-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 15px 30px rgba(0,0,0,0.2);
                }}
                
                .stat-number {{
                    font-size: 2.5em;
                    font-weight: bold;
                    margin-bottom: 10px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                }}
                
                .stat-label {{
                    font-size: 1em;
                    color: #7f8c8d;
                    font-weight: 500;
                }}
                
                .dashboard-section {{
                    background: rgba(255, 255, 255, 0.95);
                    border-radius: 15px;
                    padding: 30px;
                    margin: 30px 0;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                }}
                
                .section-title {{
                    color: #2c3e50;
                    font-size: 1.8em;
                    margin-bottom: 20px;
                    border-left: 5px solid #3498db;
                    padding-left: 15px;
                }}
                
                .chart-container {{
                    margin: 20px 0;
                    border-radius: 10px;
                    overflow: hidden;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }}
                
                .top-models {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                
                .model-card {{
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    transition: transform 0.3s ease;
                }}
                
                .model-card:hover {{
                    transform: scale(1.05);
                }}
                
                .model-rank {{
                    font-size: 2em;
                    font-weight: bold;
                    margin-bottom: 10px;
                }}
                
                .model-name {{
                    font-size: 1.2em;
                    margin-bottom: 10px;
                }}
                
                .model-score {{
                    font-size: 1.5em;
                    font-weight: bold;
                }}
                
                .nav-tabs {{
                    display: flex;
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 10px;
                    padding: 5px;
                    margin: 20px 0;
                }}
                
                .nav-tab {{
                    flex: 1;
                    padding: 15px;
                    text-align: center;
                    background: transparent;
                    border: none;
                    color: #2c3e50;
                    font-weight: bold;
                    border-radius: 8px;
                    cursor: pointer;
                    transition: all 0.3s ease;
                }}
                
                .nav-tab.active {{
                    background: white;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                
                .tab-content {{
                    display: none;
                }}
                
                .tab-content.active {{
                    display: block;
                }}
                
                .footer {{
                    text-align: center;
                    margin-top: 50px;
                    padding: 30px;
                    background: rgba(255, 255, 255, 0.95);
                    border-radius: 15px;
                    color: #7f8c8d;
                }}
                
                @media (max-width: 768px) {{
                    .container {{
                        padding: 10px;
                    }}
                    
                    .header h1 {{
                        font-size: 2em;
                    }}
                    
                    .stats-grid {{
                        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 ML-IDS-IPS Interactive Model Comparison Dashboard</h1>
                    <p>Comprehensive Analysis of {stats['total_algorithms']} Machine Learning Algorithms across {stats['total_datasets']} Cybersecurity Datasets</p>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">{stats['total_models']}</div>
                        <div class="stat-label">Models Trained</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{stats['total_datasets']}</div>
                        <div class="stat-label">Datasets</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{stats['total_algorithms']}</div>
                        <div class="stat-label">Algorithms</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{stats['best_accuracy']:.3f}</div>
                        <div class="stat-label">Best Accuracy</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{stats['avg_accuracy']:.3f}</div>
                        <div class="stat-label">Average Accuracy</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{stats['fastest_time']:.3f}s</div>
                        <div class="stat-label">Fastest Training</div>
                    </div>
                </div>
                
                <div class="dashboard-section">
                    <h2 class="section-title">🏆 Top 10 Models by Average Accuracy</h2>
                    <div class="top-models">
        """
        
        for i, (model, accuracy) in enumerate(top_models.items(), 1):
            html_content += f"""
                        <div class="model-card">
                            <div class="model-rank">#{i}</div>
                            <div class="model-name">{model.replace('_', ' ').title()}</div>
                            <div class="model-score">{accuracy:.4f}</div>
                        </div>
            """
        
        html_content += """
                    </div>
                </div>
                
                <div class="dashboard-section">
                    <h2 class="section-title">📊 Interactive Analysis Dashboard</h2>
                    <div class="nav-tabs">
                        <button class="nav-tab active" onclick="showTab('accuracy')">Accuracy Analysis</button>
                        <button class="nav-tab" onclick="showTab('performance')">Performance Metrics</button>
                        <button class="nav-tab" onclick="showTab('dataset')">Dataset Analysis</button>
                        <button class="nav-tab" onclick="showTab('family')">Model Families</button>
                        <button class="nav-tab" onclick="showTab('selector')">Model Selector</button>
                    </div>
                    
                    <div id="accuracy" class="tab-content active">
                        <div class="chart-container" id="accuracyChart"></div>
                    </div>
                    
                    <div id="performance" class="tab-content">
                        <div class="chart-container" id="performanceChart"></div>
                    </div>
                    
                    <div id="dataset" class="tab-content">
                        <div class="chart-container" id="datasetChart"></div>
                    </div>
                    
                    <div id="family" class="tab-content">
                        <div class="chart-container" id="familyChart"></div>
                    </div>
                    
                    <div id="selector" class="tab-content">
                        <div class="chart-container" id="selectorChart"></div>
                    </div>
                </div>
                
                <div class="dashboard-section">
                    <h2 class="section-title">🎯 Best Model per Dataset</h2>
                    <div class="top-models">
        """
        
        for _, row in best_by_dataset.iterrows():
            html_content += f"""
                        <div class="model-card">
                            <div class="model-name">{row['dataset'].upper()}</div>
                            <div class="model-score">{row['model'].replace('_', ' ').title()}</div>
                            <div style="margin-top: 10px; font-size: 1.2em;">{row['accuracy']:.4f}</div>
                        </div>
            """
        
        html_content += f"""
                    </div>
                </div>
                
                <div class="footer">
                    <h3>🚀 Interactive Features</h3>
                    <p>• Hover over charts for detailed information</p>
                    <p>• Click and drag to zoom into specific areas</p>
                    <p>• Use the navigation tabs to explore different analyses</p>
                    <p>• All charts are fully interactive and responsive</p>
                    <br>
                    <p><strong>Generated by ML-IDS-IPS Comprehensive Model Comparison Framework</strong></p>
                    <p>Total Models Analyzed: {stats['total_models']} | Datasets: {stats['total_datasets']} | Algorithms: {stats['total_algorithms']}</p>
                </div>
            </div>
            
            <script>
                // Tab switching functionality
                function showTab(tabName) {{
                    // Hide all tab contents
                    const tabContents = document.querySelectorAll('.tab-content');
                    tabContents.forEach(content => content.classList.remove('active'));
                    
                    // Remove active class from all tabs
                    const tabs = document.querySelectorAll('.nav-tab');
                    tabs.forEach(tab => tab.classList.remove('active'));
                    
                    // Show selected tab content
                    document.getElementById(tabName).classList.add('active');
                    
                    // Add active class to clicked tab
                    event.target.classList.add('active');
                }}
                
                // Load Plotly charts
                document.addEventListener('DOMContentLoaded', function() {{
                    // Load accuracy chart
                    fetch('interactive_accuracy_chart.html')
                        .then(response => response.text())
                        .then(html => {{
                            const parser = new DOMParser();
                            const doc = parser.parseFromString(html, 'text/html');
                            const plotDiv = doc.querySelector('#plotly-div');
                            if (plotDiv) {{
                                document.getElementById('accuracyChart').innerHTML = plotDiv.innerHTML;
                                Plotly.newPlot('accuracyChart', plotDiv.data, plotDiv.layout);
                            }}
                        }});
                    
                    // Load performance chart
                    fetch('interactive_performance_metrics.html')
                        .then(response => response.text())
                        .then(html => {{
                            const parser = new DOMParser();
                            const doc = parser.parseFromString(html, 'text/html');
                            const plotDiv = doc.querySelector('#plotly-div');
                            if (plotDiv) {{
                                document.getElementById('performanceChart').innerHTML = plotDiv.innerHTML;
                                Plotly.newPlot('performanceChart', plotDiv.data, plotDiv.layout);
                            }}
                        }});
                    
                    // Load dataset chart
                    fetch('interactive_dataset_analysis.html')
                        .then(response => response.text())
                        .then(html => {{
                            const parser = new DOMParser();
                            const doc = parser.parseFromString(html, 'text/html');
                            const plotDiv = doc.querySelector('#plotly-div');
                            if (plotDiv) {{
                                document.getElementById('datasetChart').innerHTML = plotDiv.innerHTML;
                                Plotly.newPlot('datasetChart', plotDiv.data, plotDiv.layout);
                            }}
                        }});
                    
                    // Load family chart
                    fetch('interactive_family_comparison.html')
                        .then(response => response.text())
                        .then(html => {{
                            const parser = new DOMParser();
                            const doc = parser.parseFromString(html, 'text/html');
                            const plotDiv = doc.querySelector('#plotly-div');
                            if (plotDiv) {{
                                document.getElementById('familyChart').innerHTML = plotDiv.innerHTML;
                                Plotly.newPlot('familyChart', plotDiv.data, plotDiv.layout);
                            }}
                        }});
                    
                    // Load selector chart
                    fetch('interactive_model_selector.html')
                        .then(response => response.text())
                        .then(html => {{
                            const parser = new DOMParser();
                            const doc = parser.parseFromString(html, 'text/html');
                            const plotDiv = doc.querySelector('#plotly-div');
                            if (plotDiv) {{
                                document.getElementById('selectorChart').innerHTML = plotDiv.innerHTML;
                                Plotly.newPlot('selectorChart', plotDiv.data, plotDiv.layout);
                            }}
                        }});
                }});
            </script>
        </body>
        </html>
        """
        
        with open(self.dashboard_dir / 'index.html', 'w') as f:
            f.write(html_content)
        
        logger.info("Main interactive dashboard HTML created")

def main():
    """Main function to generate interactive dashboard"""
    logger.info("Starting interactive dashboard generation...")
    
    dashboard = InteractiveDashboardGenerator()
    dashboard.generate_interactive_dashboard()
    
    logger.info("\n" + "="*80)
    logger.info("INTERACTIVE DASHBOARD GENERATION COMPLETED!")
    logger.info("="*80)
    logger.info("Interactive dashboard files saved in: results/interactive_dashboard/")
    logger.info("Open 'results/interactive_dashboard/index.html' in your browser to view the interactive dashboard")

if __name__ == "__main__":
    main()
