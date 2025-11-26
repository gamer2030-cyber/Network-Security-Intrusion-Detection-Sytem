#!/bin/bash
# install_all.sh - Complete installation script for ML-IDS-IPS Project
# This script installs all requirements for the project

set -e

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$PROJECT_DIR/config"
LOG_DIR="$PROJECT_DIR/logs"
RESULTS_DIR="$PROJECT_DIR/results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    local missing_deps=()
    
    # Check Python
    if ! command_exists python3; then
        missing_deps+=("python3")
    else
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        info "Python version: $PYTHON_VERSION"
    fi
    
    # Check pip
    if ! command_exists pip3 && ! command_exists pip; then
        missing_deps+=("pip")
    fi
    
    # Check Docker (optional but recommended)
    if ! command_exists docker; then
        warning "Docker not found. Docker is required for Kafka, Redis, and other services."
        warning "Install Docker from: https://docs.docker.com/get-docker/"
    else
        DOCKER_VERSION=$(docker --version 2>&1)
        info "Docker: $DOCKER_VERSION"
    fi
    
    # Check Docker Compose (optional but recommended)
    if ! command_exists docker-compose && ! docker compose version >/dev/null 2>&1; then
        warning "Docker Compose not found. Required for running infrastructure services."
    else
        info "Docker Compose: Available"
    fi
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        error "Missing required dependencies: ${missing_deps[*]}"
        error "Please install the missing dependencies and run this script again."
        exit 1
    fi
    
    log "Prerequisites check completed"
}

# Create project directories
create_directories() {
    log "Creating project directories..."
    
    mkdir -p "$PROJECT_DIR"/{config,logs,results,datasets,models}
    mkdir -p "$PROJECT_DIR/network_infrastructure"/{gns3_topology,device_configs,network_scripts,vmware_configs}
    mkdir -p "$PROJECT_DIR/ml_models"/{data_preprocessing,model_training,model_evaluation,adversarial_testing,model_serving}
    mkdir -p "$PROJECT_DIR/ids_ips_systems"/{suricata,zeek,custom_models,integration}
    mkdir -p "$PROJECT_DIR/monitoring_systems"/{elk_stack,traffic_capture,alerting,dashboards}
    mkdir -p "$PROJECT_DIR/scripts"/{setup,training,deployment,maintenance}
    mkdir -p "$PROJECT_DIR/tests"/{unit_tests,integration_tests,performance_tests}
    mkdir -p "$PROJECT_DIR/docs"
    mkdir -p "$PROJECT_DIR/processed_datasets"
    mkdir -p "$PROJECT_DIR/templates"
    
    log "Project directories created successfully"
}

# Update requirements.txt with all dependencies
update_requirements() {
    log "Updating requirements.txt with all project dependencies..."
    
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

# Web Framework and Real-time Communication
flask>=2.3.0
flask-socketio>=5.3.0
bcrypt>=4.0.0

# Real-time Streaming (Kafka, Redis)
redis>=4.5.0
kafka-python>=2.0.2

# Network Monitoring
psutil>=5.9.0
scapy>=2.5.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Code Quality
flake8>=6.0.0
black>=23.7.0
EOF

    log "requirements.txt updated successfully"
}

# Install Python dependencies
install_python_dependencies() {
    log "Installing Python dependencies..."
    
    # Determine pip command
    if command_exists pip3; then
        PIP_CMD="pip3"
    elif command_exists pip; then
        PIP_CMD="pip"
    else
        error "pip not found. Please install pip first."
        exit 1
    fi
    
    # Check if conda is available
    if command_exists conda; then
        info "Conda detected. Creating/updating conda environment..."
        
        # Create conda environment if it doesn't exist
        if ! conda env list | grep -q "ml-ids-ips"; then
            log "Creating conda environment 'ml-ids-ips'..."
            conda create -n ml-ids-ips python=3.9 -y
        else
            log "Conda environment 'ml-ids-ips' already exists. Updating..."
        fi
        
        # Install packages via conda
        log "Installing packages in conda environment..."
        conda run -n ml-ids-ips $PIP_CMD install --upgrade pip
        
        # Install from requirements.txt
        conda run -n ml-ids-ips $PIP_CMD install -r "$PROJECT_DIR/requirements.txt"
        
        log "Conda environment setup completed"
        ENV_TYPE="conda"
        ENV_NAME="ml-ids-ips"
        
    elif [ -d "$PROJECT_DIR/venv" ]; then
        info "Virtual environment found. Using existing venv..."
        source "$PROJECT_DIR/venv/bin/activate"
        $PIP_CMD install --upgrade pip
        $PIP_CMD install -r "$PROJECT_DIR/requirements.txt"
        ENV_TYPE="venv"
        
    else
        info "Creating Python virtual environment..."
        python3 -m venv "$PROJECT_DIR/venv"
        source "$PROJECT_DIR/venv/bin/activate"
        $PIP_CMD install --upgrade pip
        $PIP_CMD install -r "$PROJECT_DIR/requirements.txt"
        ENV_TYPE="venv"
    fi
    
    log "Python dependencies installed successfully"
}

# Setup Docker services
setup_docker_services() {
    log "Setting up Docker services..."
    
    if ! command_exists docker; then
        warning "Docker not found. Skipping Docker services setup."
        warning "To use Kafka, Redis, and other services, please install Docker."
        return 0
    fi
    
    # Check if docker-compose.yml exists
    if [ ! -f "$PROJECT_DIR/docker-compose.yml" ]; then
        warning "docker-compose.yml not found. Skipping Docker services setup."
        return 0
    fi
    
    # Check if Docker daemon is running
    if ! docker info >/dev/null 2>&1; then
        warning "Docker daemon is not running. Please start Docker and run:"
        warning "  docker-compose up -d"
        return 0
    fi
    
    info "Starting Docker services (Kafka, Zookeeper, Redis, Kafka-UI)..."
    
    # Determine docker-compose command
    if docker compose version >/dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    elif command_exists docker-compose; then
        COMPOSE_CMD="docker-compose"
    else
        warning "Docker Compose not found. Please install Docker Compose."
        return 0
    fi
    
    cd "$PROJECT_DIR"
    $COMPOSE_CMD up -d
    
    log "Waiting for services to start..."
    sleep 10
    
    # Check service status
    if $COMPOSE_CMD ps | grep -q "Up"; then
        log "Docker services started successfully"
        info "Services available at:"
        info "  - Kafka: localhost:9092"
        info "  - Redis: localhost:6379"
        info "  - Kafka UI: http://localhost:8080"
    else
        warning "Some Docker services may not have started. Check with: docker-compose ps"
    fi
    
    cd "$PROJECT_DIR"
}

# Create installation instructions file
create_installation_instructions() {
    log "Creating installation instructions..."
    
    cat > "$PROJECT_DIR/INSTALLATION_INSTRUCTIONS.md" << 'EOF'
# Installation Instructions for ML-IDS-IPS Project

## Quick Installation

Run the installation script:

```bash
chmod +x install_all.sh
./install_all.sh
```

This script will:
1. Check prerequisites (Python, Docker, etc.)
2. Create all necessary directories
3. Install all Python dependencies
4. Set up Docker services (Kafka, Redis, etc.)

## Manual Installation

### Step 1: Prerequisites

#### Required:
- **Python 3.8+** (Python 3.9 recommended)
- **pip** (Python package manager)

#### Recommended:
- **Docker** and **Docker Compose** (for Kafka, Redis, and other services)
- **Conda** (optional, for environment management)

### Step 2: Install Python Dependencies

#### Option A: Using Conda (Recommended)

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate ml-ids-ips

# Install additional packages
pip install -r requirements.txt
```

#### Option B: Using Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Set Up Docker Services

If you have Docker installed:

```bash
# Start all services (Kafka, Zookeeper, Redis, Kafka-UI)
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

Services will be available at:
- **Kafka**: `localhost:9092`
- **Redis**: `localhost:6379`
- **Kafka UI**: `http://localhost:8080`

### Step 4: Verify Installation

```bash
# Activate your environment first
conda activate ml-ids-ips
# OR
source venv/bin/activate

# Test Python imports
python3 -c "import numpy, pandas, sklearn, flask, redis, kafka; print('All imports successful!')"

# Check Docker services
docker-compose ps
```

## Project Structure

After installation, your project structure should look like:

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
├── docs/                  # Documentation
├── logs/                  # Log files
├── results/               # Results and outputs
├── models/                # Trained models
├── requirements.txt       # Python dependencies
├── environment.yml        # Conda environment file
├── docker-compose.yml     # Docker services configuration
└── install_all.sh         # Installation script
```

## Running the Project

### 1. Start Infrastructure Services

```bash
# Start Docker services
docker-compose up -d
```

### 2. Activate Environment

```bash
# If using conda
conda activate ml-ids-ips

# If using venv
source venv/bin/activate
```

### 3. Run Components

```bash
# Start live monitoring dashboard
python3 production_dashboard.py

# Start live data streaming system
python3 live_data_streaming_system.py

# Start honeypot system
python3 honeypot_system.py
```

## Troubleshooting

### Python Import Errors

If you get import errors:
1. Make sure your environment is activated
2. Reinstall requirements: `pip install -r requirements.txt`

### Docker Issues

If Docker services don't start:
1. Check Docker is running: `docker info`
2. Check ports are available: `netstat -an | grep -E '9092|6379|8080'`
3. View logs: `docker-compose logs`

### Permission Issues

On Linux/macOS, you may need to run some commands with sudo:
- Docker commands may require sudo or Docker group membership
- Network monitoring (scapy) may require root privileges

## Additional Resources

- **README.md**: Project overview and quick start
- **docs/USER_GUIDE.md**: User documentation
- **docs/ADMINISTRATOR_GUIDE.md**: Administrator guide
- **PRODUCTION_README.md**: Production deployment guide

## Support

For issues and questions, refer to the documentation in the `docs/` directory.
EOF

    log "Installation instructions created"
}

# Main installation function
main() {
    echo ""
    echo "=========================================="
    echo "  ML-IDS-IPS Project Installation"
    echo "=========================================="
    echo ""
    
    log "Starting installation process..."
    log "Project directory: $PROJECT_DIR"
    echo ""
    
    # Run installation steps
    check_prerequisites
    echo ""
    
    create_directories
    echo ""
    
    update_requirements
    echo ""
    
    install_python_dependencies
    echo ""
    
    setup_docker_services
    echo ""
    
    create_installation_instructions
    echo ""
    
    # Summary
    log "=========================================="
    log "Installation completed successfully!"
    log "=========================================="
    echo ""
    
    info "Next steps:"
    echo ""
    
    if [ "$ENV_TYPE" = "conda" ]; then
        info "1. Activate conda environment:"
        echo "   conda activate $ENV_NAME"
    else
        info "1. Activate virtual environment:"
        echo "   source venv/bin/activate"
    fi
    
    echo ""
    info "2. Verify Docker services are running:"
    echo "   docker-compose ps"
    echo ""
    info "3. Start the system:"
    echo "   python3 production_dashboard.py"
    echo ""
    info "4. Access the dashboard:"
    echo "   http://localhost:5000 (or check the script output)"
    echo ""
    info "For detailed instructions, see: INSTALLATION_INSTRUCTIONS.md"
    echo ""
    log "Installation complete!"
}

# Run main function
main "$@"

