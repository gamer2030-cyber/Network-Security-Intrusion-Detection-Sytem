#!/bin/bash
# Start Production Dashboard
# This script starts the dashboard from the organized src/ directory

echo "🚀 Starting ML-IDS-IPS Production Dashboard..."
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

# Stop any existing dashboard processes
echo "🛑 Stopping any existing dashboard processes..."
pkill -f "production_dashboard.py" 2>/dev/null
sleep 2

# Verify all stopped
if pgrep -f "production_dashboard.py" > /dev/null; then
    echo "⚠️  Force stopping remaining processes..."
    pkill -9 -f "production_dashboard.py" 2>/dev/null
    sleep 2
fi

echo "✅ All existing processes stopped"

# Start the dashboard
echo ""
echo "🌐 Starting production dashboard..."
echo ""

# Start in background
python src/production_dashboard.py > dashboard.log 2>&1 &
DASHBOARD_PID=$!

# Wait for it to start
sleep 5

# Check if it's running
if pgrep -f "production_dashboard.py" > /dev/null; then
    echo ""
    echo "✅ Dashboard is running!"
    echo ""
    echo "📊 Process Information:"
    ps aux | grep -E "production_dashboard.py" | grep -v grep | head -1
    echo ""
    echo "🌐 Dashboard available at: http://localhost:5050"
    echo ""
    echo "📝 Logs are being written to: dashboard.log"
    echo "   View logs: tail -f dashboard.log"
    echo ""
    echo "⚠️  Keep this terminal open to keep dashboard running"
    echo "⚠️  To stop: pkill -f production_dashboard.py"
    echo ""
else
    echo ""
    echo "❌ Failed to start dashboard"
    echo ""
    echo "📝 Checking logs..."
    if [ -f "dashboard.log" ]; then
        echo "   Last 20 lines of dashboard.log:"
        tail -20 dashboard.log
    else
        echo "   No log file found"
    fi
    echo ""
    exit 1
fi

