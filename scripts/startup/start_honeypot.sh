#!/bin/bash
# Start Honeypot System
# This script starts the honeypot system for real-time threat detection

echo "🎣 Starting Honeypot System..."
echo "========================================"

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

# Stop any existing honeypot processes
echo "🛑 Stopping any existing honeypot processes..."
pkill -f "honeypot_system.py" 2>/dev/null
sleep 2

# Verify all stopped
if pgrep -f "honeypot_system.py" > /dev/null; then
    echo "⚠️  Force stopping remaining processes..."
    pkill -9 -f "honeypot_system.py" 2>/dev/null
    sleep 2
fi

echo "✅ All existing processes stopped"

# Start the honeypot system
echo ""
echo "🎣 Starting honeypot system..."
echo "   This will deploy fake services to attract attackers"
echo ""

# Start in background
python src/honeypot_system.py > honeypot.log 2>&1 &
HONEYPOT_PID=$!

# Wait for it to start
sleep 3

# Check if it's running
if pgrep -f "honeypot_system.py" > /dev/null; then
    echo ""
    echo "✅ Honeypot System is running!"
    echo ""
    echo "📊 Process Information:"
    ps aux | grep -E "honeypot_system.py" | grep -v grep | head -1
    echo ""
    echo "🎣 Active Honeypot Services:"
    echo "   - SSH: port 2222"
    echo "   - HTTP: port 8080"
    echo "   - FTP: port 2121"
    echo "   - Telnet: port 2323"
    echo "   - MySQL: port 3307"
    echo ""
    echo "📝 Logs are being written to: honeypot.log"
    echo "   View logs: tail -f honeypot.log"
    echo ""
    echo "⚠️  Keep this terminal open to keep honeypot running"
    echo "⚠️  To stop: pkill -f honeypot_system.py"
    echo ""
else
    echo ""
    echo "❌ Failed to start honeypot system"
    echo ""
    echo "📝 Checking logs..."
    if [ -f "honeypot.log" ]; then
        echo "   Last 20 lines of honeypot.log:"
        tail -20 honeypot.log
    else
        echo "   No log file found"
    fi
    echo ""
    exit 1
fi

