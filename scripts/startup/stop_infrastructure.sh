#!/bin/bash
# Stop ML-IDS-IPS live infrastructure

echo "🛑 Stopping ML-IDS-IPS Live Infrastructure..."

# Stop Docker services
docker-compose down

echo "✅ Infrastructure stopped successfully!"
        