#!/bin/bash

# Start Network Monitoring System
# This script starts the live data streaming system with proper permissions

echo "🚀 Starting Network Monitoring System..."
echo "========================================"

# Check if we're in an interactive terminal
if [ ! -t 0 ]; then
    echo "⚠️  Warning: Not running in an interactive terminal"
    echo "   Sudo password prompt may not work"
    echo "   Please run this script in a terminal window"
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "⚠️  Docker daemon is not running!"
    echo "   Please start Docker Desktop or Docker daemon first"
    echo "   On macOS: Open Docker Desktop application"
    echo "   On Linux: sudo systemctl start docker"
    exit 1
fi

# Check if native Kafka/Zookeeper services are running (Homebrew)
if command -v brew > /dev/null 2>&1; then
    if brew services list 2>/dev/null | grep -q "zookeeper.*started\|kafka.*started"; then
        echo "⚠️  Native Kafka/Zookeeper services are running (Homebrew)"
        echo "   Stopping them to avoid port conflicts..."
        brew services stop zookeeper kafka 2>/dev/null || true
        sleep 2
    fi
fi

# Check if infrastructure containers are running
KAFKA_RUNNING=$(docker ps --filter "name=ml-ids-ips-kafka" --format "{{.Names}}" 2>/dev/null | grep -c "ml-ids-ips-kafka" || echo "0")
REDIS_RUNNING=$(docker ps --filter "name=ml-ids-ips-redis" --format "{{.Names}}" 2>/dev/null | grep -c "ml-ids-ips-redis" || echo "0")
ZOOKEEPER_RUNNING=$(docker ps --filter "name=ml-ids-ips-zookeeper" --format "{{.Names}}" 2>/dev/null | grep -c "ml-ids-ips-zookeeper" || echo "0")

if [ "$KAFKA_RUNNING" -eq 0 ] || [ "$REDIS_RUNNING" -eq 0 ] || [ "$ZOOKEEPER_RUNNING" -eq 0 ]; then
    echo "⚠️  Infrastructure containers are not running. Starting infrastructure..."
    if [ -f "./start_infrastructure.sh" ]; then
        ./start_infrastructure.sh
        echo "⏳ Waiting for services to be ready..."
        sleep 15
        
        # Verify services are up
        MAX_RETRIES=10
        RETRY_COUNT=0
        while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
            KAFKA_RUNNING=$(docker ps --filter "name=ml-ids-ips-kafka" --format "{{.Names}}" 2>/dev/null | grep -c "ml-ids-ips-kafka" || echo "0")
            REDIS_RUNNING=$(docker ps --filter "name=ml-ids-ips-redis" --format "{{.Names}}" 2>/dev/null | grep -c "ml-ids-ips-redis" || echo "0")
            
            if [ "$KAFKA_RUNNING" -eq 1 ] && [ "$REDIS_RUNNING" -eq 1 ]; then
                echo "✅ Infrastructure is ready!"
                break
            fi
            
            RETRY_COUNT=$((RETRY_COUNT + 1))
            echo "   Waiting for services... ($RETRY_COUNT/$MAX_RETRIES)"
            sleep 3
        done
        
        if [ "$KAFKA_RUNNING" -eq 0 ] || [ "$REDIS_RUNNING" -eq 0 ]; then
            echo "❌ Error: Infrastructure failed to start"
            echo "   Please check Docker logs: docker-compose logs"
            exit 1
        fi
    else
        echo "❌ Error: start_infrastructure.sh not found"
        echo "   Please start infrastructure manually: docker-compose up -d"
        exit 1
    fi
else
    echo "✅ Infrastructure containers are running"
fi

# Test connections to services
echo ""
echo "🔍 Testing service connections..."
echo "   Testing Redis connection (localhost:6379)..."
if command -v nc > /dev/null 2>&1; then
    if nc -z localhost 6379 2>/dev/null; then
        echo "   ✅ Redis is accessible"
    else
        echo "   ⚠️  Warning: Cannot connect to Redis on localhost:6379"
        echo "      Container may still be starting up..."
        sleep 5
    fi
else
    echo "   ⚠️  'nc' command not available, skipping connection test"
fi

echo "   Testing Kafka connection (localhost:9092)..."
if command -v nc > /dev/null 2>&1; then
    if nc -z localhost 9092 2>/dev/null; then
        echo "   ✅ Kafka is accessible"
    else
        echo "   ⚠️  Warning: Cannot connect to Kafka on localhost:9092"
        echo "      Container may still be starting up..."
        echo "      If this persists, check: docker-compose logs kafka"
        sleep 5
    fi
else
    echo "   ⚠️  'nc' command not available, skipping connection test"
fi
echo ""

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "❌ Error: Virtual environment not found"
    echo "   Please create a virtual environment first:"
    echo "   python3 -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python not found in virtual environment"
    exit 1
fi

# Stop any existing monitoring processes
echo "🛑 Stopping any existing monitoring processes..."
pkill -f "live_data_streaming_system.py" 2>/dev/null
pkill -f "url_threat_detector.py" 2>/dev/null
sleep 3

# Force kill if still running
if pgrep -f "live_data_streaming_system.py" > /dev/null; then
    echo "⚠️  Force stopping remaining live_data_streaming_system processes..."
    pkill -9 -f "live_data_streaming_system.py" 2>/dev/null
    sleep 2
fi

if pgrep -f "url_threat_detector.py" > /dev/null; then
    echo "⚠️  Force stopping remaining url_threat_detector processes..."
    pkill -9 -f "url_threat_detector.py" 2>/dev/null
    sleep 2
fi

# Verify all stopped
if pgrep -f "live_data_streaming_system.py" > /dev/null; then
    echo "❌ Error: Could not stop existing live_data_streaming_system processes"
    echo "   Please manually kill them: pkill -9 -f live_data_streaming_system.py"
    exit 1
fi

if pgrep -f "url_threat_detector.py" > /dev/null; then
    echo "❌ Error: Could not stop existing url_threat_detector processes"
    echo "   Please manually kill them: pkill -9 -f url_threat_detector.py"
    exit 1
fi

echo "✅ All existing processes stopped"

# Start the live data streaming system with sudo
echo ""
echo "📡 Starting live data streaming system..."
echo "⚠️  This requires sudo permissions for packet capture on macOS"
echo ""

# Check if sudo password is cached
if sudo -n true 2>/dev/null; then
    # Sudo password is cached
    echo "✅ Sudo password is cached, starting system..."
    sudo -E python live_data_streaming_system.py > live_system.log 2>&1 &
else
    # Need password - validate it first interactively
    echo "🔐 Please enter your password when prompted..."
    echo "   (This is required for packet capture on macOS)"
    echo ""
    
    # Validate password first (this will prompt interactively)
    if sudo -v; then
        # Password validated, now start the process
        echo "✅ Password validated, starting system..."
        sudo -E python src/live_data_streaming_system.py > live_system.log 2>&1 &
    else
        echo "❌ Password validation failed"
        exit 1
    fi
fi

# Wait for it to start
sleep 3

# Check if live data streaming system is running
if pgrep -f "live_data_streaming_system.py" > /dev/null; then
    echo ""
    echo "✅ Live Data Streaming System is running!"
    echo ""
    echo "📊 Process Information:"
    ps aux | grep -E "live_data_streaming_system.py" | grep -v grep | head -1
    echo ""
else
    echo ""
    echo "❌ Failed to start live data streaming system"
    echo ""
    echo "📝 Checking logs..."
    if [ -f "live_system.log" ]; then
        echo "   Last 20 lines of live_system.log:"
        tail -20 live_system.log
    else
        echo "   No log file found"
    fi
    echo ""
    echo "🔍 Troubleshooting:"
    echo "1. Check if you have sudo permissions"
    echo "2. Check if Kafka and Redis are running:"
    echo "   - Kafka: pgrep -f kafka"
    echo "   - Redis: pgrep -f redis"
    echo "3. Check if virtual environment is activated"
    echo "4. Try running manually: sudo -E python src/live_data_streaming_system.py"
    echo ""
    exit 1
fi

# Start URL Threat Detector
echo ""
echo "🌐 Starting URL Threat Detector..."
echo "⚠️  This requires sudo permissions for packet capture"
echo ""

# Check if url_threat_detector.py exists
if [ ! -f "src/url_threat_detector.py" ]; then
    echo "⚠️  Warning: url_threat_detector.py not found"
    echo "   Skipping URL threat detection..."
else
    # Check if sudo password is cached
    if sudo -n true 2>/dev/null; then
        # Sudo password is cached
        echo "✅ Sudo password is cached, starting URL threat detector..."
        sudo -E python src/url_threat_detector.py > url_threat_detector.log 2>&1 &
    else
        # Need password - validate it first interactively
        echo "🔐 Please enter your password when prompted (if not already cached)..."
        echo "   (This is required for packet capture on macOS)"
        echo ""
        
        # Validate password first (this will prompt interactively)
        if sudo -v; then
            # Password validated, now start the process
            echo "✅ Password validated, starting URL threat detector..."
            sudo -E python src/url_threat_detector.py > url_threat_detector.log 2>&1 &
        else
            echo "❌ Password validation failed"
            echo "⚠️  Continuing without URL threat detector..."
        fi
    fi
    
    # Wait for it to start
    sleep 3
    
    # Check if it's running
    if pgrep -f "url_threat_detector.py" > /dev/null; then
        echo "✅ URL Threat Detector is running!"
        echo ""
        echo "📊 Process Information:"
        ps aux | grep -E "url_threat_detector.py" | grep -v grep | head -1
        echo ""
        echo "📝 Logs are being written to: url_threat_detector.log"
        echo "   View logs: tail -f url_threat_detector.log"
        echo ""
    else
        echo "⚠️  URL Threat Detector failed to start (check logs: url_threat_detector.log)"
        echo "   Continuing with main monitoring system..."
        echo ""
    fi
fi

# Final summary
echo "========================================"
echo "✅ Network Monitoring System Started!"
echo "========================================"
echo ""
echo "📊 Running Services:"
if pgrep -f "live_data_streaming_system.py" > /dev/null; then
    echo "   ✅ Live Data Streaming System"
else
    echo "   ❌ Live Data Streaming System (not running)"
fi

if pgrep -f "url_threat_detector.py" > /dev/null; then
    echo "   ✅ URL Threat Detector (malicious website detection)"
else
    echo "   ⚠️  URL Threat Detector (not running)"
fi
echo ""
echo "📊 Next steps:"
echo "1. Open the dashboard: http://localhost:5050 (or port shown in dashboard output)"
echo "2. Login with credentials (default: admin / admin123)"
echo "3. Click 'Start Monitoring' button in the dashboard"
echo "4. Data should start appearing in the dashboard"
echo "5. URL threats will appear in 'Security Alerts' section"
echo ""
echo "📝 Log Files:"
echo "   - Main system: tail -f live_system.log"
if [ -f "src/url_threat_detector.py" ]; then
    echo "   - URL threats: tail -f url_threat_detector.log"
fi
echo ""
echo "⚠️  Keep this terminal open to keep monitoring running"
echo ""
echo "🛑 To stop all monitoring:"
echo "   pkill -f live_data_streaming_system.py"
echo "   pkill -f url_threat_detector.py"
echo ""

