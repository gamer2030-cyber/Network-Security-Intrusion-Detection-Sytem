#!/bin/bash
# Start ML-IDS-IPS live infrastructure

echo "🚀 Starting ML-IDS-IPS Live Infrastructure..."

# Start Docker services
echo "Starting Docker services..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 30

# Create Kafka topics
echo "Creating Kafka topics..."
./scripts/startup/create_kafka_topics.sh

echo "✅ Infrastructure started successfully!"
echo ""
echo "Services available at:"
echo "  Kafka: localhost:9092"
echo "  Redis: localhost:6379"
echo "  Kafka UI: http://localhost:8080"
echo ""
echo "Next steps:"
echo "  1. Run: python src/live_data_streaming_system.py"
echo "  2. Run: python src/production_dashboard.py"
echo "  3. Open: http://localhost:5050"
        