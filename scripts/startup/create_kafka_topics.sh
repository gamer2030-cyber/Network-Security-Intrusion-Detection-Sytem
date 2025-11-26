#!/bin/bash
# Create Kafka topics for ML-IDS-IPS system

echo "Creating Kafka topics..."

# Wait for Kafka to be ready
sleep 30

# Create topics
docker exec ml-ids-ips-kafka kafka-topics --create --topic network_packets --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec ml-ids-ips-kafka kafka-topics --create --topic ml_predictions --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
docker exec ml-ids-ips-kafka kafka-topics --create --topic security_alerts --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

echo "Kafka topics created successfully!"
echo "Topics:"
docker exec ml-ids-ips-kafka kafka-topics --list --bootstrap-server localhost:9092
        