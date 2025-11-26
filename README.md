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
   python3 ml_models/data_preprocessing/dataset_manager.py
   ```

4. **Train Models**:
   ```bash
   python3 ml_models/model_training/model_trainer.py
   ```

5. **Test Adversarial Attacks**:
   ```bash
   python3 ml_models/adversarial_testing/adversarial_tester.py
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

## System Requirements

### Software Requirements
- Python 3.8+
- Scikit-learn 1.3+
- Pandas 2.0+
- NumPy 1.24+
- Flask 2.0+
- Redis
- Apache Kafka

### Hardware Requirements
- CPU: Multi-core processor (4+ cores recommended)
- RAM: 8GB+ (16GB+ for production)
- Storage: 10GB+ for datasets and models
- Network: Network interface capable of packet capture

For complete requirements specification, see `REQUIREMENTS.md`.

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

## Documentation

- **Requirements Document**: `REQUIREMENTS.md` - Complete project requirements and specifications
- **Project Scope**: `Project_Scope_and_Objectives.md` - Project scope and objectives
- **Implementation Summary**: `Project_Implementation_Summary.md` - Implementation overview
- **User Guide**: `docs/USER_GUIDE.md` - End-user documentation
- **Administrator Guide**: `docs/ADMINISTRATOR_GUIDE.md` - System administration guide
- **Production Guide**: `PRODUCTION_README.md` - Production deployment guide

## License

This project is for educational and research purposes.

## Contributing

Please read the documentation before contributing to this project.

## Support

For issues and questions, please refer to the documentation in the `docs/` directory.
