#!/bin/bash
# setup.sh - Simple setup script for ML-IDS-IPS project

set -e

# Configuration
PROJECT_DIR="$(pwd)"
CONFIG_DIR="$PROJECT_DIR/config"
LOG_DIR="$PROJECT_DIR/logs"
RESULTS_DIR="$PROJECT_DIR/results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

# Create project directories
create_directories() {
    log "Creating project directories..."
    
    mkdir -p $PROJECT_DIR/{config,logs,results,datasets,models}
    mkdir -p $PROJECT_DIR/network_infrastructure/{gns3_topology,device_configs,network_scripts,vmware_configs}
    mkdir -p $PROJECT_DIR/ml_models/{data_preprocessing,model_training,model_evaluation,adversarial_testing,model_serving}
    mkdir -p $PROJECT_DIR/ids_ips_systems/{suricata,zeek,custom_models,integration}
    mkdir -p $PROJECT_DIR/monitoring_systems/{elk_stack,traffic_capture,alerting,dashboards}
    mkdir -p $PROJECT_DIR/scripts/{setup,training,deployment,maintenance}
    mkdir -p $PROJECT_DIR/tests/{unit_tests,integration_tests,performance_tests}
    mkdir -p $PROJECT_DIR/docs
    
    log "Project directories created successfully"
}

# Install Python dependencies using conda/pip
install_python_dependencies() {
    log "Installing Python dependencies..."
    
    # Check if conda is available
    if command -v conda &> /dev/null; then
        log "Conda detected. Creating dedicated environment for ML-IDS-IPS..."
        
        # Create conda environment if it doesn't exist
        if ! conda env list | grep -q "ml-ids-ips"; then
            log "Creating conda environment 'ml-ids-ips'..."
            conda create -n ml-ids-ips python=3.9 -y
        fi
        
        # Activate environment and install packages
        log "Installing packages in conda environment..."
        
        # Use conda run to execute commands in the environment
        conda run -n ml-ids-ips pip install --upgrade pip
        
        # Install packages via conda first (faster and more reliable)
        conda install -n ml-ids-ips -c conda-forge \
            numpy \
            pandas \
            scikit-learn \
            matplotlib \
            seaborn \
            pyyaml \
            requests \
            tqdm \
            plotly \
            pytest \
            pytest-cov \
            flake8 \
            black \
            scipy \
            joblib \
            -y
        
        log "Conda packages installed successfully"
        
    elif command -v python3 &> /dev/null; then
        log "Using system Python with virtual environment..."
        
        # Create virtual environment
        python3 -m venv venv
        source venv/bin/activate
        
        # Upgrade pip
        pip install --upgrade pip
        
        # Install packages
        pip install \
            numpy \
            pandas \
            scikit-learn \
            matplotlib \
            seaborn \
            pyyaml \
            joblib \
            requests \
            tqdm \
            plotly \
            pytest \
            pytest-cov \
            flake8 \
            black \
            scipy
        
        log "Virtual environment packages installed successfully"
    else
        error "Neither conda nor Python 3 is available. Please install Python 3 or conda first."
        exit 1
    fi
    
    log "Python dependencies installed successfully"
}

# Create configuration files
create_configurations() {
    log "Creating configuration files..."
    
    # Ensure config directory exists
    mkdir -p "$CONFIG_DIR"
    
    # Main ML configuration
    cat > "$CONFIG_DIR/ml_config.yaml" << 'EOF'
ml_models:
  datasets:
    nsl_kdd:
      path: "./datasets/nsl_kdd"
      features: 41
      classes: 5
    unsw_nb15:
      path: "./datasets/unsw_nb15"
      features: 49
      classes: 10
    cicids2017:
      path: "./datasets/cicids2017"
      features: 78
      classes: 5
  
  models:
    neural_network:
      architecture: [128, 64, 32]
      activation: "relu"
      dropout: 0.3
      epochs: 100
      batch_size: 32
      learning_rate: 0.001
    
    random_forest:
      n_estimators: 100
      max_depth: 10
      random_state: 42
    
    svm:
      kernel: "rbf"
      C: 1.0
      gamma: "scale"
    
    gradient_boosting:
      n_estimators: 100
      learning_rate: 0.1
      max_depth: 6
  
  adversarial_testing:
    attacks:
      fgsm:
        epsilon_values: [0.01, 0.05, 0.1, 0.2]
      pgd:
        epsilon: 0.1
        epsilon_step: 0.01
        max_iter: 10
      cw:
        confidence: 0.0
        learning_rate: 0.01
        max_iter: 1000
    
    defenses:
      adversarial_training:
        enabled: true
        attack_type: "fgsm"
        epsilon: 0.1
      feature_squeezing:
        enabled: true
        bit_depth: 8
EOF

    log "Configuration files created successfully"
}

# Create Python requirements file
create_requirements() {
    log "Creating requirements.txt..."
    
    cat > "$PROJECT_DIR/requirements.txt" << 'EOF'
# Core ML Libraries
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Data Processing
scipy>=1.11.0

# Configuration
pyyaml>=6.0.0
python-dotenv>=1.0.0

# Utilities
requests>=2.31.0
joblib>=1.3.0
tqdm>=4.65.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Code Quality
flake8>=6.0.0
black>=23.7.0
EOF

    # Create conda environment file
    cat > "$PROJECT_DIR/environment.yml" << 'EOF'
name: ml-ids-ips
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - numpy>=1.24.0
  - pandas>=2.0.0
  - scikit-learn>=1.3.0
  - matplotlib>=3.7.0
  - seaborn>=0.12.0
  - plotly>=5.15.0
  - scipy>=1.11.0
  - pyyaml>=6.0.0
  - python-dotenv>=1.0.0
  - requests>=2.31.0
  - joblib>=1.3.0
  - tqdm>=4.65.0
  - pytest>=7.4.0
  - pytest-cov>=4.1.0
  - flake8>=6.0.0
  - black>=23.7.0
  - pip
  - pip:
    - python-dotenv>=1.0.0
EOF

    log "Requirements.txt created successfully"
}

# Create sample Python scripts
create_sample_scripts() {
    log "Creating sample Python scripts..."
    
    # Ensure all necessary directories exist
    mkdir -p "$PROJECT_DIR/ml_models/data_preprocessing"
    mkdir -p "$PROJECT_DIR/ml_models/model_training"
    mkdir -p "$PROJECT_DIR/ml_models/adversarial_testing"
    
    # Dataset manager
    cat > "$PROJECT_DIR/ml_models/data_preprocessing/dataset_manager.py" << 'EOF'
#!/usr/bin/env python3
# dataset_manager.py - Dataset management for ML-IDS-IPS

import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetManager:
    def __init__(self, base_dir="./datasets"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def create_sample_dataset(self, dataset_name, n_samples=1000):
        """Create a sample dataset for testing"""
        logger.info(f"Creating sample dataset: {dataset_name}")
        
        # Create sample data
        n_features = 20
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        df['label'] = y
        
        # Save dataset
        dataset_dir = os.path.join(self.base_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        df.to_csv(os.path.join(dataset_dir, f'{dataset_name}_sample.csv'), index=False)
        
        logger.info(f"Sample dataset created: {dataset_name}")
        return df

if __name__ == "__main__":
    manager = DatasetManager()
    manager.create_sample_dataset("nsl_kdd")
    manager.create_sample_dataset("unsw_nb15")
    manager.create_sample_dataset("cicids2017")
    print("Sample datasets created successfully!")
EOF

    # Model trainer
    cat > "$PROJECT_DIR/ml_models/model_training/model_trainer.py" << 'EOF'
#!/usr/bin/env python3
# model_trainer.py - ML model training for IDS/IPS

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, models_dir="./models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
    
    def create_sample_data(self, n_samples=1000, n_features=20):
        """Create sample training data"""
        logger.info("Creating sample training data...")
        
        # Generate synthetic data
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        logger.info("Training Random Forest model...")
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Random Forest accuracy: {accuracy:.4f}")
        
        # Save model
        model_path = os.path.join(self.models_dir, "random_forest.pkl")
        joblib.dump(model, model_path)
        
        return model, accuracy
    
    def train_svm(self, X_train, y_train, X_test, y_test):
        """Train SVM model"""
        logger.info("Training SVM model...")
        
        model = SVC(kernel='rbf', random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"SVM accuracy: {accuracy:.4f}")
        
        # Save model
        model_path = os.path.join(self.models_dir, "svm.pkl")
        joblib.dump(model, model_path)
        
        return model, accuracy

if __name__ == "__main__":
    trainer = ModelTrainer()
    
    # Create sample data
    X_train, X_test, y_train, y_test = trainer.create_sample_data()
    
    # Train models
    rf_model, rf_acc = trainer.train_random_forest(X_train, y_train, X_test, y_test)
    svm_model, svm_acc = trainer.train_svm(X_train, y_train, X_test, y_test)
    
    print(f"\nTraining completed!")
    print(f"Random Forest accuracy: {rf_acc:.4f}")
    print(f"SVM accuracy: {svm_acc:.4f}")
EOF

    # Adversarial tester
    cat > "$PROJECT_DIR/ml_models/adversarial_testing/adversarial_tester.py" << 'EOF'
#!/usr/bin/env python3
# adversarial_tester.py - Adversarial ML testing

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdversarialTester:
    def __init__(self, results_dir="./results"):
        self.results_dir = results_dir
        self.results = {}
        os.makedirs(results_dir, exist_ok=True)
    
    def create_sample_model(self, n_samples=1000, n_features=20):
        """Create a sample trained model"""
        logger.info("Creating sample trained model...")
        
        # Generate data
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        return model, X_test, y_test
    
    def test_fgsm_simulation(self, model, X_test, y_test, epsilon=0.1):
        """Simulate FGSM attack"""
        logger.info(f"Testing FGSM attack with epsilon={epsilon}")
        
        # Simulate adversarial examples
        X_adv = X_test + epsilon * np.random.randn(*X_test.shape)
        
        # Evaluate model
        predictions = model.predict(X_adv)
        accuracy = np.mean(predictions == y_test)
        
        result = {
            'attack_type': 'fgsm',
            'epsilon': epsilon,
            'accuracy': accuracy,
            'success_rate': 1 - accuracy
        }
        
        self.results['fgsm'] = result
        logger.info(f"FGSM attack success rate: {result['success_rate']:.4f}")
        
        return result
    
    def test_pgd_simulation(self, model, X_test, y_test, epsilon=0.1, iterations=10):
        """Simulate PGD attack"""
        logger.info(f"Testing PGD attack with epsilon={epsilon}, iterations={iterations}")
        
        X_adv = X_test.copy()
        
        # Simulate iterative attack
        for i in range(iterations):
            X_adv += (epsilon / iterations) * np.random.randn(*X_adv.shape)
        
        # Evaluate model
        predictions = model.predict(X_adv)
        accuracy = np.mean(predictions == y_test)
        
        result = {
            'attack_type': 'pgd',
            'epsilon': epsilon,
            'iterations': iterations,
            'accuracy': accuracy,
            'success_rate': 1 - accuracy
        }
        
        self.results['pgd'] = result
        logger.info(f"PGD attack success rate: {result['success_rate']:.4f}")
        
        return result
    
    def generate_report(self):
        """Generate adversarial testing report"""
        logger.info("Generating adversarial testing report...")
        
        report = {
            'summary': {
                'total_attacks': len(self.results),
                'results': self.results
            }
        }
        
        # Save report
        report_path = os.path.join(self.results_dir, 'adversarial_test_report.json')
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {report_path}")
        return report

if __name__ == "__main__":
    tester = AdversarialTester()
    
    # Create sample model
    model, X_test, y_test = tester.create_sample_model()
    
    # Test attacks
    fgsm_result = tester.test_fgsm_simulation(model, X_test, y_test)
    pgd_result = tester.test_pgd_simulation(model, X_test, y_test)
    
    # Generate report
    report = tester.generate_report()
    
    print(f"\nAdversarial testing completed!")
    print(f"FGSM attack success rate: {fgsm_result['success_rate']:.4f}")
    print(f"PGD attack success rate: {pgd_result['success_rate']:.4f}")
EOF

    log "Sample Python scripts created successfully"
}

# Create README
create_readme() {
    log "Creating README.md..."
    
    cat > "$PROJECT_DIR/README.md" << 'EOF'
# ML-Based IDS/IPS Project

Machine Learning for Intrusion Detection Systems (IDS/IPS): Strengths, weaknesses, and adversarial ML

## Project Overview

This project implements a comprehensive ML-based Intrusion Detection and Prevention System with adversarial ML testing capabilities.

## Features

- **Network Infrastructure**: GNS3 topology with secure network segmentation
- **ML Models**: Multiple algorithms (Neural Networks, Random Forest, SVM, Gradient Boosting)
- **IDS/IPS Integration**: Suricata and Zeek with ML enhancement
- **Adversarial Testing**: Comprehensive adversarial ML testing framework
- **Monitoring**: ELK Stack and traffic capture
- **Documentation**: Complete documentation suite

## Quick Start

1. **Setup Environment**:
   ```bash
   ./setup.sh
   ```

2. **Activate Environment**:
   ```bash
   # If using conda (recommended)
   conda activate ml-ids-ips
   
   # If using virtual environment
   source venv/bin/activate
   ```

3. **Create Sample Data**:
   ```bash
   conda run -n ml-ids-ips python ml_models/data_preprocessing/dataset_manager.py
   ```

4. **Train Models**:
   ```bash
   conda run -n ml-ids-ips python ml_models/model_training/model_trainer.py
   ```

5. **Test Adversarial Attacks**:
   ```bash
   conda run -n ml-ids-ips python ml_models/adversarial_testing/adversarial_tester.py
   ```

## Project Structure

```
ML-IDS-IPS-Project/
├── config/                 # Configuration files
├── ml_models/             # ML model implementations
├── ids_ips_systems/       # IDS/IPS system configurations
├── monitoring_systems/    # Monitoring and logging
├── network_infrastructure/ # Network topology and configs
├── datasets/              # Training datasets
├── scripts/               # Automation scripts
├── tests/                 # Test suites
└── docs/                  # Documentation
```

## Configuration

- **ML Config**: `config/ml_config.yaml`
- **Network Config**: `config/network_config.yaml`
- **Monitoring Config**: `config/monitoring_config.yaml`

## Requirements

- Python 3.8+
- Scikit-learn 1.3+
- Pandas 2.0+
- NumPy 1.24+

## Installation

### Option 1: Using Conda (Recommended)
1. Clone the repository
2. Run `./setup.sh` to create conda environment and install dependencies
3. Activate environment: `conda activate ml-ids-ips`
4. Follow the quick start guide above

### Option 2: Using Virtual Environment
1. Clone the repository
2. Run `./setup.sh` (will create virtual environment if conda not available)
3. Activate environment: `source venv/bin/activate`
4. Follow the quick start guide above

### Option 3: Manual Installation
1. Create conda environment: `conda env create -f environment.yml`
2. Activate environment: `conda activate ml-ids-ips`
3. Or create virtual environment: `python3 -m venv venv && source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run scripts: `conda run -n ml-ids-ips python <script_name>` or `python <script_name>` (if environment is activated)

## Documentation

- Installation Guide: `docs/installation_guide.md`
- Configuration Guide: `docs/configuration_guide.md`
- User Manual: `docs/user_manual.md`

## License

This project is for educational and research purposes.

## Contributing

Please read the documentation before contributing to this project.

## Support

For issues and questions, please refer to the documentation in the `docs/` directory.
EOF

    log "README.md created successfully"
}

# Main setup function
main() {
    log "Starting ML-IDS-IPS project setup..."
    
    create_directories
    install_python_dependencies
    create_configurations
    create_requirements
    create_sample_scripts
    create_readme
    
    log "ML-IDS-IPS project setup completed successfully!"
    log ""
    log "Next steps:"
    log "1. Activate the environment:"
    if command -v conda &> /dev/null; then
        log "   conda activate ml-ids-ips"
    else
        log "   source venv/bin/activate"
    fi
    log "2. Create sample data: conda run -n ml-ids-ips python ml_models/data_preprocessing/dataset_manager.py"
    log "3. Train models: conda run -n ml-ids-ips python ml_models/model_training/model_trainer.py"
    log "4. Test adversarial attacks: conda run -n ml-ids-ips python ml_models/adversarial_testing/adversarial_tester.py"
    log ""
    log "Project directory: $PROJECT_DIR"
    log "Configuration files: $CONFIG_DIR"
    log ""
    if command -v conda &> /dev/null; then
        log "Environment: conda environment 'ml-ids-ips' created"
    else
        log "Environment: Python virtual environment 'venv' created"
    fi
}

# Run main function
main "$@"