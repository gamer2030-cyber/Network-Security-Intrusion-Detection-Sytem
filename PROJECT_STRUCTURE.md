# Project Structure - Organized Layout

## 📁 Folder Organization

### **`src/`** - Core Application Files
Main Python application files (moved from root):
- `live_data_streaming_system.py` - Main packet capture and ML prediction system
- `production_dashboard.py` - Web dashboard interface  
- `url_threat_detector.py` - URL/domain threat detection
- `honeypot_system.py` - Honeypot for catching attackers
- `security_alerts.py` - Alert management

### **`scripts/startup/`** - Startup Scripts
Scripts to start system components (moved from root):
- `start_monitoring.sh` - Start main monitoring system
- `start_infrastructure.sh` - Start Docker services (Kafka, Redis, Zookeeper)
- `start_honeypot.sh` - Start honeypot system
- `stop_infrastructure.sh` - Stop Docker services
- `create_kafka_topics.sh` - Create Kafka topics

### **`scripts/setup/`** - Setup/Installation Scripts
Installation and setup utilities (moved from root):
- `install_all.sh` - Complete installation script
- `setup.sh` - Initial setup script
- `detect_network_interface.py` - Detect network interface
- `download_datasets.py` - Download training datasets
- `preprocess_datasets.py` - Preprocess datasets

### **`config/`** - Configuration Files
- `live_config.yaml` - Live system configuration
- `production_config.yaml` - Production configuration
- `ml_config.yaml` - ML model configuration
- `email_config.yaml` - Email alert configuration
- `threat_intelligence.json` - Threat intelligence data

### **`templates/`** - HTML Templates
- `production_dashboard.html` - Main dashboard template
- `login.html` - Login page template
- `simplified_live_dashboard.html` - Simplified dashboard

### **`models/`** - Trained ML Models
Pre-trained machine learning models (.pkl files):
- Random Forest models
- Gradient Boosting models
- SVM models
- Bagging models

### **`datasets/`** - Training Datasets
- `nsl_kdd/` - NSL-KDD dataset
- `unsw_nb15/` - UNSW-NB15 dataset
- `cicids2017/` - CICIDS2017 dataset

### **`processed_datasets/`** - Preprocessed Data
Preprocessed training data and preprocessors

### **`docs/`** - Documentation
- `project/` - Project documentation (HOW_THE_PROJECT_WORKS.md, etc.)
- `USER_GUIDE.md` - User guide
- `ADMINISTRATOR_GUIDE.md` - Administrator guide

### **`results/`** - Results and Reports
Model training results, reports, and dashboards

### **`ml_models/`** - ML Model Training Code
- `model_training/` - Model training scripts
- `data_preprocessing/` - Data preprocessing
- `adversarial_testing/` - Adversarial testing

### **`logs/`** - Log Files
System log files (generated at runtime)

### **`certs/`** - SSL Certificates
SSL/TLS certificates for HTTPS (optional, for production)

---

## 🚀 Quick Start Commands

### Using Convenience Scripts (Root Directory)
```bash
# Start infrastructure (Docker services)
./start_infrastructure.sh

# Start monitoring system
./start_monitoring.sh

# Start honeypot
./start_honeypot.sh

# Start dashboard
python src/production_dashboard.py
```

### Using Organized Scripts (Direct Path)
```bash
# Start infrastructure
./scripts/startup/start_infrastructure.sh

# Start monitoring
./scripts/startup/start_monitoring.sh

# Start honeypot
./scripts/startup/start_honeypot.sh
```

---

## 📊 File Organization Summary

**Before:** All files in root directory (messy, hard to navigate)

**After:** Organized into logical folders:
- ✅ Core application files → `src/`
- ✅ Startup scripts → `scripts/startup/`
- ✅ Setup scripts → `scripts/setup/`
- ✅ Project docs → `docs/project/`
- ✅ Convenience scripts in root for easy access

---

## 📝 Notes

- All core Python files are now in `src/`
- All startup scripts are in `scripts/startup/`
- All setup scripts are in `scripts/setup/`
- Configuration files remain in `config/`
- Root directory has convenience scripts that redirect to organized locations
- All script paths have been updated to work with new structure
