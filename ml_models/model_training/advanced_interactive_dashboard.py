#!/usr/bin/env python3
"""
advanced_interactive_dashboard.py - Advanced interactive dashboard with filtering and real-time features

This script creates an ultra-interactive dashboard with advanced filtering, 
real-time model comparison, and dynamic data exploration capabilities.
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
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedInteractiveDashboard:
    """Generate ultra-interactive dashboard with advanced features"""
    
    def __init__(self, results_dir="./results", models_dir="./models", processed_data_dir="./processed_datasets"):
        self.results_dir = Path(results_dir)
        self.models_dir = Path(models_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.dashboard_dir = self.results_dir / "advanced_interactive_dashboard"
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
    
    def create_advanced_interactive_dashboard(self):
        """Create ultra-advanced interactive dashboard"""
        logger.info("Creating advanced interactive dashboard...")
        
        # Load comparison data
        comparison_df = self.load_comparison_data()
        if comparison_df is None:
            return
        
        # Create comprehensive statistics
        stats = {
            'total_models': len(comparison_df),
            'total_datasets': comparison_df['dataset'].nunique(),
            'total_algorithms': comparison_df['model'].nunique(),
            'best_accuracy': comparison_df['accuracy'].max(),
            'avg_accuracy': comparison_df['accuracy'].mean(),
            'fastest_model': comparison_df.loc[comparison_df['training_time'].idxmin(), 'model'],
            'fastest_time': comparison_df['training_time'].min(),
            'families': comparison_df['family'].unique().tolist(),
            'datasets': comparison_df['dataset'].unique().tolist()
        }
        
        # Convert DataFrame to JSON for JavaScript
        comparison_json = comparison_df.to_json(orient='records')
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>🚀 ML-IDS-IPS Advanced Interactive Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
                    max-width: 1600px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                
                .header {{
                    background: rgba(255, 255, 255, 0.95);
                    border-radius: 20px;
                    padding: 40px;
                    margin-bottom: 30px;
                    box-shadow: 0 15px 35px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                
                .header h1 {{
                    color: #2c3e50;
                    font-size: 3em;
                    margin-bottom: 15px;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
                }}
                
                .header p {{
                    color: #7f8c8d;
                    font-size: 1.3em;
                }}
                
                .controls-panel {{
                    background: rgba(255, 255, 255, 0.95);
                    border-radius: 15px;
                    padding: 25px;
                    margin: 20px 0;
                    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                }}
                
                .controls-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                
                .control-group {{
                    display: flex;
                    flex-direction: column;
                }}
                
                .control-label {{
                    font-weight: bold;
                    margin-bottom: 8px;
                    color: #2c3e50;
                }}
                
                .control-select {{
                    padding: 10px;
                    border: 2px solid #ddd;
                    border-radius: 8px;
                    font-size: 14px;
                    background: white;
                    transition: border-color 0.3s ease;
                }}
                
                .control-select:focus {{
                    outline: none;
                    border-color: #3498db;
                }}
                
                .control-range {{
                    width: 100%;
                    margin: 10px 0;
                }}
                
                .range-value {{
                    text-align: center;
                    font-weight: bold;
                    color: #3498db;
                }}
                
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }}
                
                .stat-card {{
                    background: rgba(255, 255, 255, 0.95);
                    border-radius: 15px;
                    padding: 25px;
                    text-align: center;
                    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                    cursor: pointer;
                }}
                
                .stat-card:hover {{
                    transform: translateY(-8px);
                    box-shadow: 0 20px 40px rgba(0,0,0,0.2);
                }}
                
                .stat-number {{
                    font-size: 2.8em;
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
                    border-radius: 20px;
                    padding: 35px;
                    margin: 30px 0;
                    box-shadow: 0 15px 35px rgba(0,0,0,0.1);
                }}
                
                .section-title {{
                    color: #2c3e50;
                    font-size: 2em;
                    margin-bottom: 25px;
                    border-left: 6px solid #3498db;
                    padding-left: 20px;
                }}
                
                .chart-container {{
                    margin: 25px 0;
                    border-radius: 15px;
                    overflow: hidden;
                    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
                    background: white;
                    padding: 20px;
                }}
                
                .nav-tabs {{
                    display: flex;
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 15px;
                    padding: 8px;
                    margin: 25px 0;
                    flex-wrap: wrap;
                }}
                
                .nav-tab {{
                    flex: 1;
                    min-width: 150px;
                    padding: 15px 20px;
                    text-align: center;
                    background: transparent;
                    border: none;
                    color: #2c3e50;
                    font-weight: bold;
                    border-radius: 12px;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    margin: 2px;
                }}
                
                .nav-tab.active {{
                    background: white;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    transform: scale(1.05);
                }}
                
                .tab-content {{
                    display: none;
                }}
                
                .tab-content.active {{
                    display: block;
                }}
                
                .model-comparison {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin: 25px 0;
                }}
                
                .model-card {{
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    color: white;
                    padding: 25px;
                    border-radius: 15px;
                    text-align: center;
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                    cursor: pointer;
                }}
                
                .model-card:hover {{
                    transform: scale(1.08);
                    box-shadow: 0 15px 30px rgba(0,0,0,0.3);
                }}
                
                .model-card.selected {{
                    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                    transform: scale(1.05);
                }}
                
                .model-rank {{
                    font-size: 2.5em;
                    font-weight: bold;
                    margin-bottom: 15px;
                }}
                
                .model-name {{
                    font-size: 1.3em;
                    margin-bottom: 15px;
                    font-weight: bold;
                }}
                
                .model-score {{
                    font-size: 1.8em;
                    font-weight: bold;
                }}
                
                .comparison-panel {{
                    background: rgba(255, 255, 255, 0.95);
                    border-radius: 15px;
                    padding: 25px;
                    margin: 20px 0;
                    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                }}
                
                .comparison-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }}
                
                .comparison-item {{
                    text-align: center;
                    padding: 15px;
                    background: #f8f9fa;
                    border-radius: 10px;
                    border: 2px solid transparent;
                    transition: all 0.3s ease;
                }}
                
                .comparison-item:hover {{
                    border-color: #3498db;
                    background: #e3f2fd;
                }}
                
                .comparison-label {{
                    font-weight: bold;
                    color: #2c3e50;
                    margin-bottom: 5px;
                }}
                
                .comparison-value {{
                    font-size: 1.5em;
                    font-weight: bold;
                    color: #3498db;
                }}
                
                .footer {{
                    text-align: center;
                    margin-top: 50px;
                    padding: 40px;
                    background: rgba(255, 255, 255, 0.95);
                    border-radius: 20px;
                    color: #7f8c8d;
                }}
                
                .loading {{
                    text-align: center;
                    padding: 50px;
                    font-size: 1.2em;
                    color: #3498db;
                }}
                
                @media (max-width: 768px) {{
                    .container {{
                        padding: 10px;
                    }}
                    
                    .header h1 {{
                        font-size: 2.2em;
                    }}
                    
                    .stats-grid {{
                        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    }}
                    
                    .nav-tabs {{
                        flex-direction: column;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 ML-IDS-IPS Advanced Interactive Dashboard</h1>
                    <p>Ultra-Interactive Analysis of {stats['total_algorithms']} ML Algorithms across {stats['total_datasets']} Cybersecurity Datasets</p>
                </div>
                
                <div class="controls-panel">
                    <h2 class="section-title">🎛️ Interactive Controls</h2>
                    <div class="controls-grid">
                        <div class="control-group">
                            <label class="control-label">Filter by Dataset:</label>
                            <select class="control-select" id="datasetFilter" onchange="filterData()">
                                <option value="all">All Datasets</option>
                                {''.join([f'<option value="{dataset}">{dataset.upper()}</option>' for dataset in stats['datasets']])}
                            </select>
                        </div>
                        
                        <div class="control-group">
                            <label class="control-label">Filter by Model Family:</label>
                            <select class="control-select" id="familyFilter" onchange="filterData()">
                                <option value="all">All Families</option>
                                {''.join([f'<option value="{family}">{family}</option>' for family in stats['families']])}
                            </select>
                        </div>
                        
                        <div class="control-group">
                            <label class="control-label">Min Accuracy:</label>
                            <input type="range" class="control-range" id="accuracyRange" min="0" max="1" step="0.01" value="0" oninput="updateAccuracyValue()">
                            <div class="range-value" id="accuracyValue">0.00</div>
                        </div>
                        
                        <div class="control-group">
                            <label class="control-label">Max Training Time (s):</label>
                            <input type="range" class="control-range" id="timeRange" min="0" max="100" step="1" value="100" oninput="updateTimeValue()">
                            <div class="range-value" id="timeValue">100.00</div>
                        </div>
                        
                        <div class="control-group">
                            <label class="control-label">Sort by:</label>
                            <select class="control-select" id="sortBy" onchange="sortData()">
                                <option value="accuracy">Accuracy</option>
                                <option value="f1_macro">F1 Score</option>
                                <option value="training_time">Training Time</option>
                                <option value="cv_mean">CV Score</option>
                            </select>
                        </div>
                        
                        <div class="control-group">
                            <label class="control-label">Sort Order:</label>
                            <select class="control-select" id="sortOrder" onchange="sortData()">
                                <option value="desc">Descending</option>
                                <option value="asc">Ascending</option>
                            </select>
                        </div>
                    </div>
                </div>
                
                <div class="stats-grid" id="statsGrid">
                    <div class="stat-card" onclick="showModelDetails('best')">
                        <div class="stat-number">{stats['best_accuracy']:.3f}</div>
                        <div class="stat-label">Best Accuracy</div>
                    </div>
                    <div class="stat-card" onclick="showModelDetails('fastest')">
                        <div class="stat-number">{stats['fastest_time']:.3f}s</div>
                        <div class="stat-label">Fastest Training</div>
                    </div>
                    <div class="stat-card" onclick="showModelDetails('average')">
                        <div class="stat-number">{stats['avg_accuracy']:.3f}</div>
                        <div class="stat-label">Average Accuracy</div>
                    </div>
                    <div class="stat-card" onclick="showModelDetails('total')">
                        <div class="stat-number">{stats['total_models']}</div>
                        <div class="stat-label">Total Models</div>
                    </div>
                </div>
                
                <div class="dashboard-section">
                    <h2 class="section-title">📊 Interactive Analysis Dashboard</h2>
                    <div class="nav-tabs">
                        <button class="nav-tab active" onclick="showTab('overview')">Overview</button>
                        <button class="nav-tab" onclick="showTab('performance')">Performance</button>
                        <button class="nav-tab" onclick="showTab('comparison')">Model Comparison</button>
                        <button class="nav-tab" onclick="showTab('family')">Family Analysis</button>
                        <button class="nav-tab" onclick="showTab('dataset')">Dataset Analysis</button>
                        <button class="nav-tab" onclick="showTab('insights')">AI Insights</button>
                    </div>
                    
                    <div id="overview" class="tab-content active">
                        <div class="chart-container" id="overviewChart">
                            <div class="loading">Loading interactive overview chart...</div>
                        </div>
                    </div>
                    
                    <div id="performance" class="tab-content">
                        <div class="chart-container" id="performanceChart">
                            <div class="loading">Loading performance metrics...</div>
                        </div>
                    </div>
                    
                    <div id="comparison" class="tab-content">
                        <div class="comparison-panel">
                            <h3>Select Models to Compare:</h3>
                            <div class="model-comparison" id="modelComparison">
                                <div class="loading">Loading model comparison...</div>
                            </div>
                        </div>
                    </div>
                    
                    <div id="family" class="tab-content">
                        <div class="chart-container" id="familyChart">
                            <div class="loading">Loading family analysis...</div>
                        </div>
                    </div>
                    
                    <div id="dataset" class="tab-content">
                        <div class="chart-container" id="datasetChart">
                            <div class="loading">Loading dataset analysis...</div>
                        </div>
                    </div>
                    
                    <div id="insights" class="tab-content">
                        <div class="comparison-panel">
                            <h3>🤖 AI-Generated Insights</h3>
                            <div id="aiInsights">
                                <div class="loading">Generating AI insights...</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="footer">
                    <h3>🚀 Advanced Interactive Features</h3>
                    <p>• Real-time filtering and sorting</p>
                    <p>• Interactive model selection and comparison</p>
                    <p>• Dynamic charts with zoom and pan</p>
                    <p>• AI-powered insights and recommendations</p>
                    <p>• Responsive design for all devices</p>
                    <br>
                    <p><strong>Generated by ML-IDS-IPS Advanced Interactive Dashboard</strong></p>
                    <p>Total Models: {stats['total_models']} | Datasets: {stats['total_datasets']} | Algorithms: {stats['total_algorithms']}</p>
                </div>
            </div>
            
            <script>
                // Global data
                let allData = {comparison_json};
                let filteredData = [...allData];
                let selectedModels = [];
                
                // Initialize dashboard
                document.addEventListener('DOMContentLoaded', function() {{
                    initializeDashboard();
                    loadCharts();
                    generateAIInsights();
                }});
                
                // Tab switching
                function showTab(tabName) {{
                    const tabContents = document.querySelectorAll('.tab-content');
                    tabContents.forEach(content => content.classList.remove('active'));
                    
                    const tabs = document.querySelectorAll('.nav-tab');
                    tabs.forEach(tab => tab.classList.remove('active'));
                    
                    document.getElementById(tabName).classList.add('active');
                    event.target.classList.add('active');
                    
                    // Load chart for active tab
                    loadChartForTab(tabName);
                }}
                
                // Filter data based on controls
                function filterData() {{
                    const datasetFilter = document.getElementById('datasetFilter').value;
                    const familyFilter = document.getElementById('familyFilter').value;
                    const minAccuracy = parseFloat(document.getElementById('accuracyRange').value);
                    const maxTime = parseFloat(document.getElementById('timeRange').value);
                    
                    filteredData = allData.filter(item => {{
                        return (datasetFilter === 'all' || item.dataset === datasetFilter) &&
                               (familyFilter === 'all' || item.family === familyFilter) &&
                               item.accuracy >= minAccuracy &&
                               item.training_time <= maxTime;
                    }});
                    
                    updateCharts();
                    updateModelComparison();
                }}
                
                // Sort data
                function sortData() {{
                    const sortBy = document.getElementById('sortBy').value;
                    const sortOrder = document.getElementById('sortOrder').value;
                    
                    filteredData.sort((a, b) => {{
                        const aVal = a[sortBy];
                        const bVal = b[sortBy];
                        return sortOrder === 'desc' ? bVal - aVal : aVal - bVal;
                    }});
                    
                    updateCharts();
                    updateModelComparison();
                }}
                
                // Update range value displays
                function updateAccuracyValue() {{
                    document.getElementById('accuracyValue').textContent = 
                        parseFloat(document.getElementById('accuracyRange').value).toFixed(2);
                    filterData();
                }}
                
                function updateTimeValue() {{
                    document.getElementById('timeValue').textContent = 
                        parseFloat(document.getElementById('timeRange').value).toFixed(2);
                    filterData();
                }}
                
                // Initialize dashboard
                function initializeDashboard() {{
                    // Set initial range values
                    updateAccuracyValue();
                    updateTimeValue();
                    
                    // Load initial data
                    updateModelComparison();
                }}
                
                // Load charts
                function loadCharts() {{
                    loadOverviewChart();
                    loadPerformanceChart();
                    loadFamilyChart();
                    loadDatasetChart();
                }}
                
                // Load chart for specific tab
                function loadChartForTab(tabName) {{
                    switch(tabName) {{
                        case 'overview':
                            loadOverviewChart();
                            break;
                        case 'performance':
                            loadPerformanceChart();
                            break;
                        case 'family':
                            loadFamilyChart();
                            break;
                        case 'dataset':
                            loadDatasetChart();
                            break;
                    }}
                }}
                
                // Load overview chart
                function loadOverviewChart() {{
                    const trace1 = {{
                        x: filteredData.map(d => d.model),
                        y: filteredData.map(d => d.accuracy),
                        type: 'bar',
                        name: 'Accuracy',
                        marker: {{color: 'lightblue'}}
                    }};
                    
                    const trace2 = {{
                        x: filteredData.map(d => d.model),
                        y: filteredData.map(d => d.f1_macro),
                        type: 'bar',
                        name: 'F1 Score',
                        marker: {{color: 'lightgreen'}}
                    }};
                    
                    const layout = {{
                        title: 'Model Performance Overview',
                        xaxis: {{title: 'Model'}},
                        yaxis: {{title: 'Score'}},
                        barmode: 'group'
                    }};
                    
                    Plotly.newPlot('overviewChart', [trace1, trace2], layout);
                }}
                
                // Load performance chart
                function loadPerformanceChart() {{
                    const trace = {{
                        x: filteredData.map(d => d.training_time),
                        y: filteredData.map(d => d.accuracy),
                        mode: 'markers',
                        type: 'scatter',
                        text: filteredData.map(d => d.model),
                        marker: {{
                            size: 12,
                            color: filteredData.map(d => d.f1_macro),
                            colorscale: 'Viridis',
                            showscale: true,
                            colorbar: {{title: 'F1 Score'}}
                        }},
                        hovertemplate: '<b>%{{text}}</b><br>' +
                                    'Accuracy: %{{y:.3f}}<br>' +
                                    'Training Time: %{{x:.3f}}s<br>' +
                                    'F1 Score: %{{marker.color:.3f}}<extra></extra>'
                    }};
                    
                    const layout = {{
                        title: 'Performance vs Training Time',
                        xaxis: {{title: 'Training Time (seconds)'}},
                        yaxis: {{title: 'Accuracy'}}
                    }};
                    
                    Plotly.newPlot('performanceChart', [trace], layout);
                }}
                
                // Load family chart
                function loadFamilyChart() {{
                    const familyStats = {{}};
                    filteredData.forEach(item => {{
                        if (!familyStats[item.family]) {{
                            familyStats[item.family] = {{accuracy: [], count: 0}};
                        }}
                        familyStats[item.family].accuracy.push(item.accuracy);
                        familyStats[item.family].count++;
                    }});
                    
                    const families = Object.keys(familyStats);
                    const avgAccuracy = families.map(family => 
                        familyStats[family].accuracy.reduce((a, b) => a + b, 0) / familyStats[family].count
                    );
                    
                    const trace = {{
                        x: families,
                        y: avgAccuracy,
                        type: 'bar',
                        marker: {{color: 'lightcoral'}},
                        text: avgAccuracy.map(acc => acc.toFixed(3)),
                        textposition: 'auto'
                    }};
                    
                    const layout = {{
                        title: 'Average Accuracy by Model Family',
                        xaxis: {{title: 'Model Family'}},
                        yaxis: {{title: 'Average Accuracy'}}
                    }};
                    
                    Plotly.newPlot('familyChart', [trace], layout);
                }}
                
                // Load dataset chart
                function loadDatasetChart() {{
                    const datasets = [...new Set(filteredData.map(d => d.dataset))];
                    const models = [...new Set(filteredData.map(d => d.model))];
                    
                    const traces = datasets.map(dataset => {{
                        const datasetData = filteredData.filter(d => d.dataset === dataset);
                        return {{
                            x: datasetData.map(d => d.model),
                            y: datasetData.map(d => d.accuracy),
                            type: 'bar',
                            name: dataset.toUpperCase(),
                            hovertemplate: '<b>%{{x}}</b><br>' +
                                        'Dataset: ' + dataset + '<br>' +
                                        'Accuracy: %{{y:.3f}}<extra></extra>'
                        }};
                    }});
                    
                    const layout = {{
                        title: 'Model Performance by Dataset',
                        xaxis: {{title: 'Model'}},
                        yaxis: {{title: 'Accuracy'}},
                        barmode: 'group'
                    }};
                    
                    Plotly.newPlot('datasetChart', traces, layout);
                }}
                
                // Update charts when data changes
                function updateCharts() {{
                    loadOverviewChart();
                    loadPerformanceChart();
                    loadFamilyChart();
                    loadDatasetChart();
                }}
                
                // Update model comparison
                function updateModelComparison() {{
                    const topModels = filteredData
                        .sort((a, b) => b.accuracy - a.accuracy)
                        .slice(0, 12);
                    
                    const html = topModels.map((model, index) => `
                        <div class="model-card ${{selectedModels.includes(model.model) ? 'selected' : ''}}" 
                             onclick="toggleModelSelection('${{model.model}}')">
                            <div class="model-rank">#${{index + 1}}</div>
                            <div class="model-name">${{model.model.replace('_', ' ').toUpperCase()}}</div>
                            <div class="model-score">${{model.accuracy.toFixed(3)}}</div>
                        </div>
                    `).join('');
                    
                    document.getElementById('modelComparison').innerHTML = html;
                }}
                
                // Toggle model selection
                function toggleModelSelection(modelName) {{
                    const index = selectedModels.indexOf(modelName);
                    if (index > -1) {{
                        selectedModels.splice(index, 1);
                    }} else {{
                        selectedModels.push(modelName);
                    }}
                    updateModelComparison();
                    updateComparisonPanel();
                }}
                
                // Update comparison panel
                function updateComparisonPanel() {{
                    if (selectedModels.length === 0) return;
                    
                    const comparisonData = filteredData.filter(d => selectedModels.includes(d.model));
                    
                    const html = `
                        <h3>Selected Models Comparison</h3>
                        <div class="comparison-grid">
                            ${{comparisonData.map(model => `
                                <div class="comparison-item">
                                    <div class="comparison-label">${{model.model.replace('_', ' ').toUpperCase()}}</div>
                                    <div class="comparison-value">${{model.accuracy.toFixed(3)}}</div>
                                    <div style="font-size: 0.9em; color: #666;">
                                        F1: ${{model.f1_macro.toFixed(3)}}<br>
                                        Time: ${{model.training_time.toFixed(2)}}s
                                    </div>
                                </div>
                            `).join('')}}
                        </div>
                    `;
                    
                    // Add comparison panel if it doesn't exist
                    let panel = document.querySelector('.comparison-panel .comparison-grid');
                    if (!panel) {{
                        panel = document.createElement('div');
                        panel.className = 'comparison-grid';
                        document.querySelector('.comparison-panel').appendChild(panel);
                    }}
                    panel.innerHTML = html;
                }}
                
                // Generate AI insights
                function generateAIInsights() {{
                    const insights = [
                        `🏆 <strong>Best Performing Model:</strong> ${{filteredData.reduce((best, current) => 
                            current.accuracy > best.accuracy ? current : best).model}} with ${{filteredData.reduce((best, current) => 
                            current.accuracy > best.accuracy ? current : best).accuracy.toFixed(3)}} accuracy`,
                        `⚡ <strong>Fastest Model:</strong> ${{filteredData.reduce((fastest, current) => 
                            current.training_time < fastest.training_time ? current : fastest).model}} with ${{filteredData.reduce((fastest, current) => 
                            current.training_time < fastest.training_time ? current : fastest).training_time.toFixed(3)}}s training time`,
                        `📊 <strong>Most Consistent Model:</strong> ${{filteredData.reduce((most, current) => 
                            current.cv_std < most.cv_std ? current : most).model}} with ${{filteredData.reduce((most, current) => 
                            current.cv_std < most.cv_std ? current : most).cv_std.toFixed(3)}} CV standard deviation`,
                        `🎯 <strong>Recommended for Production:</strong> Based on accuracy and training time balance, consider ${{filteredData
                            .sort((a, b) => (b.accuracy * 0.7 + (100 - a.training_time) * 0.3) - (a.accuracy * 0.7 + (100 - b.training_time) * 0.3))
                            [0].model}} for optimal performance`
                    ];
                    
                    document.getElementById('aiInsights').innerHTML = insights.map(insight => 
                        `<div style="margin: 15px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #3498db;">${{insight}}</div>`
                    ).join('');
                }}
                
                // Show model details
                function showModelDetails(type) {{
                    let model;
                    switch(type) {{
                        case 'best':
                            model = filteredData.reduce((best, current) => 
                                current.accuracy > best.accuracy ? current : best);
                            break;
                        case 'fastest':
                            model = filteredData.reduce((fastest, current) => 
                                current.training_time < fastest.training_time ? current : fastest);
                            break;
                        case 'average':
                            // Show model closest to average
                            const avg = filteredData.reduce((sum, item) => sum + item.accuracy, 0) / filteredData.length;
                            model = filteredData.reduce((closest, current) => 
                                Math.abs(current.accuracy - avg) < Math.abs(closest.accuracy - avg) ? current : closest);
                            break;
                        default:
                            return;
                    }}
                    
                    alert(`Model: ${{model.model}}\\nAccuracy: ${{model.accuracy.toFixed(3)}}\\nF1 Score: ${{model.f1_macro.toFixed(3)}}\\nTraining Time: ${{model.training_time.toFixed(2)}}s\\nFamily: ${{model.family}}`);
                }}
            </script>
        </body>
        </html>
        """
        
        with open(self.dashboard_dir / 'index.html', 'w') as f:
            f.write(html_content)
        
        logger.info("Advanced interactive dashboard HTML created")

def main():
    """Main function to generate advanced interactive dashboard"""
    logger.info("Starting advanced interactive dashboard generation...")
    
    dashboard = AdvancedInteractiveDashboard()
    dashboard.create_advanced_interactive_dashboard()
    
    logger.info("\n" + "="*80)
    logger.info("ADVANCED INTERACTIVE DASHBOARD GENERATION COMPLETED!")
    logger.info("="*80)
    logger.info("Advanced interactive dashboard saved in: results/advanced_interactive_dashboard/")
    logger.info("Open 'results/advanced_interactive_dashboard/index.html' in your browser to view the ultra-interactive dashboard")

if __name__ == "__main__":
    main()
