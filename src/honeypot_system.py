#!/usr/bin/env python3
"""
honeypot_system.py - Real-time Honeypot System for IDS/IPS

This module implements a comprehensive honeypot system that:
- Deploys fake services (SSH, HTTP, FTP, Telnet, MySQL)
- Detects connection attempts and suspicious activities
- Sends real-time alerts to Kafka
- Integrates with existing IDS/IPS dashboard
"""

import os
import sys
import time
import json
import logging
import threading
import socket
import ipaddress
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Real-time streaming libraries
try:
    import redis
    from kafka import KafkaProducer
    from kafka.errors import KafkaError
except ImportError:
    print("Installing required streaming libraries...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "redis", "kafka-python"])
    import redis
    from kafka import KafkaProducer
    from kafka.errors import KafkaError

try:
    import yaml
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
    import yaml


class HoneypotService:
    """Base class for honeypot services"""
    
    def __init__(self, name: str, port: int, alert_callback):
        self.name = name
        self.port = port
        self.alert_callback = alert_callback
        self.socket = None
        self.is_running = False
        self.connection_count = 0
        self.connections = []
        
    def start(self):
        """Start the honeypot service"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(('0.0.0.0', self.port))
            self.socket.listen(5)
            self.socket.settimeout(1.0)
            self.is_running = True
            logger.info(f"🎣 {self.name} honeypot started on port {self.port}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start {self.name} honeypot on port {self.port}: {e}")
            return False
    
    def stop(self):
        """Stop the honeypot service"""
        self.is_running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        logger.info(f"🛑 {self.name} honeypot stopped")
    
    def handle_connection(self, client_socket, address):
        """Handle incoming connection"""
        self.connection_count += 1
        ip_address = address[0]
        port = address[1]
        
        logger.warning(f"🚨 HONEYPOT ALERT: {ip_address} connected to {self.name} on port {self.port}")
        
        # Generate alert
        alert_data = {
            'alert_id': f"honeypot_{self.name}_{int(time.time())}_{self.connection_count}",
            'timestamp': datetime.now().isoformat(),
            'severity': 'HIGH',
            'threat_type': 'honeypot_connection',
            'honeypot_service': self.name,
            'honeypot_port': self.port,
            'src_ip': ip_address,
            'src_port': port,
            'dst_ip': '0.0.0.0',
            'dst_port': self.port,
            'protocol': 'TCP',
            'description': f"Honeypot connection detected: {ip_address} attempted to connect to {self.name} service on port {self.port}",
            'confidence': 1.0,  # Honeypot connections are 100% malicious
            'model': 'honeypot',
            'dataset': 'honeypot'
        }
        
        # Call alert callback
        if self.alert_callback:
            self.alert_callback(alert_data)
        
        # Log connection attempt
        try:
            # Send fake banner/response based on service
            if self.name == 'SSH':
                client_socket.send(b'SSH-2.0-OpenSSH_7.4\r\n')
            elif self.name == 'HTTP':
                response = b'HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<html><body>Welcome</body></html>\r\n'
                client_socket.send(response)
            elif self.name == 'FTP':
                client_socket.send(b'220 Welcome to FTP server\r\n')
            elif self.name == 'Telnet':
                client_socket.send(b'Welcome to Telnet server\r\n')
            elif self.name == 'MySQL':
                client_socket.send(b'\x0a5.7.28-0ubuntu0.18.04.4\x00')
            
            # Wait a bit to capture any data sent
            client_socket.settimeout(5.0)
            try:
                data = client_socket.recv(1024)
                if data:
                    logger.warning(f"📝 Data received from {ip_address}: {data[:100]}")
                    alert_data['data_received'] = data.decode('utf-8', errors='ignore')[:500]
                    alert_data['description'] += f" | Data: {data[:50].decode('utf-8', errors='ignore')}"
            except socket.timeout:
                pass
        except Exception as e:
            logger.debug(f"Error handling {self.name} connection: {e}")
        finally:
            try:
                client_socket.close()
            except:
                pass
    
    def run(self):
        """Main service loop"""
        while self.is_running:
            try:
                client_socket, address = self.socket.accept()
                # Handle connection in a separate thread
                thread = threading.Thread(
                    target=self.handle_connection,
                    args=(client_socket, address),
                    daemon=True
                )
                thread.start()
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running:
                    logger.error(f"❌ Error in {self.name} honeypot: {e}")
                break


class HoneypotSystem:
    """Main honeypot system that manages multiple honeypot services"""
    
    def __init__(self, config_path: str = "config/live_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        
        # Initialize connections
        self.redis_client = None
        self.kafka_producer = None
        
        # Honeypot services
        self.services: List[HoneypotService] = []
        self.is_running = False
        
        # Statistics
        self.stats = {
            'total_connections': 0,
            'connections_by_service': {},
            'connections_by_ip': {},
            'start_time': None
        }
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Add default honeypot config if not present
            if 'honeypot' not in config:
                config['honeypot'] = {
                    'enabled': True,
                    'services': {
                        'SSH': {'port': 2222, 'enabled': True},
                        'HTTP': {'port': 8080, 'enabled': True},
                        'FTP': {'port': 2121, 'enabled': True},
                        'Telnet': {'port': 2323, 'enabled': True},
                        'MySQL': {'port': 3307, 'enabled': True}
                    }
                }
            
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {
                'honeypot': {
                    'enabled': True,
                    'services': {
                        'SSH': {'port': 2222, 'enabled': True},
                        'HTTP': {'port': 8080, 'enabled': True},
                        'FTP': {'port': 2121, 'enabled': True},
                        'Telnet': {'port': 2323, 'enabled': True},
                        'MySQL': {'port': 3307, 'enabled': True}
                    }
                }
            }
    
    def initialize_redis(self) -> bool:
        """Initialize Redis connection"""
        try:
            redis_config = self.config.get('redis', {})
            self.redis_client = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0),
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("✅ Redis connection established for honeypot")
            return True
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            return False
    
    def initialize_kafka(self) -> bool:
        """Initialize Kafka producer"""
        try:
            kafka_config = self.config.get('kafka', {})
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=kafka_config.get('bootstrap_servers', ['localhost:9092']),
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                retries=5,
                max_in_flight_requests_per_connection=1,
                acks='all',
                request_timeout_ms=30000,
                metadata_max_age_ms=300000,
                retry_backoff_ms=100
            )
            logger.info("✅ Kafka producer initialized for honeypot")
            return True
        except Exception as e:
            logger.error(f"❌ Kafka initialization failed: {e}")
            return False
    
    def send_alert(self, alert_data: Dict[str, Any]):
        """Send honeypot alert to Kafka and Redis"""
        try:
            # Send to Kafka
            if self.kafka_producer:
                try:
                    topic = self.config.get('kafka', {}).get('topic_alerts', 'security_alerts')
                    future = self.kafka_producer.send(
                        topic,
                        value=alert_data,
                        key=alert_data['alert_id']
                    )
                    future.add_errback(lambda e: logger.warning(f"⚠️ Kafka alert send error: {e}"))
                except Exception as e:
                    logger.warning(f"⚠️ Failed to send alert to Kafka: {e}")
            
            # Store in Redis
            if self.redis_client:
                try:
                    # Store alert
                    key = f"honeypot_alert:{alert_data['alert_id']}"
                    self.redis_client.setex(key, 3600, json.dumps(alert_data))  # Expire in 1 hour
                    
                    # Add to recent alerts list
                    self.redis_client.lpush("recent_honeypot_alerts", json.dumps(alert_data))
                    self.redis_client.ltrim("recent_honeypot_alerts", 0, 999)  # Keep last 1000
                    
                    # Update statistics
                    self.stats['total_connections'] += 1
                    service = alert_data.get('honeypot_service', 'unknown')
                    self.stats['connections_by_service'][service] = \
                        self.stats['connections_by_service'].get(service, 0) + 1
                    ip = alert_data.get('src_ip', 'unknown')
                    self.stats['connections_by_ip'][ip] = \
                        self.stats['connections_by_ip'].get(ip, 0) + 1
                except Exception as e:
                    logger.warning(f"⚠️ Failed to store alert in Redis: {e}")
            
            logger.warning(f"🚨 Honeypot alert sent: {alert_data['description']}")
            
        except Exception as e:
            logger.error(f"❌ Error sending honeypot alert: {e}")
    
    def start_services(self):
        """Start all enabled honeypot services"""
        honeypot_config = self.config.get('honeypot', {})
        if not honeypot_config.get('enabled', True):
            logger.info("⚠️ Honeypot system is disabled in config")
            return
        
        services_config = honeypot_config.get('services', {})
        
        for service_name, service_config in services_config.items():
            if service_config.get('enabled', True):
                port = service_config.get('port')
                if port:
                    service = HoneypotService(
                        name=service_name,
                        port=port,
                        alert_callback=self.send_alert
                    )
                    if service.start():
                        self.services.append(service)
                        # Start service in a separate thread
                        thread = threading.Thread(target=service.run, daemon=True)
                        thread.start()
                        logger.info(f"✅ {service_name} honeypot service started on port {port}")
                    else:
                        logger.warning(f"⚠️ Failed to start {service_name} honeypot on port {port}")
    
    def stop_services(self):
        """Stop all honeypot services"""
        for service in self.services:
            service.stop()
        self.services = []
        logger.info("🛑 All honeypot services stopped")
    
    def monitoring_thread(self):
        """Thread for monitoring and statistics"""
        logger.info("📊 Starting honeypot monitoring thread...")
        
        while self.is_running:
            try:
                uptime = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
                
                stats_data = {
                    'timestamp': datetime.now().isoformat(),
                    'uptime_seconds': uptime,
                    'total_connections': self.stats['total_connections'],
                    'connections_by_service': self.stats['connections_by_service'],
                    'active_services': len(self.services)
                }
                
                if self.redis_client:
                    self.redis_client.setex("honeypot_stats", 60, json.dumps(stats_data))
                
                logger.info(f"📊 Honeypot Stats - Total connections: {self.stats['total_connections']}, "
                          f"Active services: {len(self.services)}")
                
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Honeypot monitoring error: {e}")
                time.sleep(5)
    
    def start_system(self):
        """Start the honeypot system"""
        logger.info("🚀 Starting Honeypot System...")
        
        # Initialize connections
        if not self.initialize_redis():
            logger.warning("⚠️ Redis not available, continuing without Redis")
        
        if not self.initialize_kafka():
            logger.warning("⚠️ Kafka not available, continuing without Kafka")
        
        # Start system
        self.is_running = True
        self.stats['start_time'] = time.time()
        
        # Start honeypot services
        self.start_services()
        
        # Start monitoring thread
        monitoring_thread = threading.Thread(target=self.monitoring_thread, daemon=True)
        monitoring_thread.start()
        
        logger.info("✅ Honeypot System started successfully!")
        logger.info(f"🎣 Active honeypot services: {len(self.services)}")
        
        return True
    
    def stop_system(self):
        """Stop the honeypot system"""
        logger.info("🛑 Stopping Honeypot System...")
        
        self.is_running = False
        self.stop_services()
        
        # Close connections
        if self.kafka_producer:
            self.kafka_producer.close()
        if self.redis_client:
            self.redis_client.close()
        
        logger.info("✅ Honeypot System stopped")


def main():
    """Main function to run the honeypot system"""
    import signal
    
    honeypot = HoneypotSystem()
    
    def signal_handler(sig, frame):
        logger.info("\n🛑 Shutting down honeypot system...")
        honeypot.stop_system()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if honeypot.start_system():
        try:
            # Keep main thread alive
            while honeypot.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            signal_handler(None, None)
    else:
        logger.error("❌ Failed to start honeypot system")
        sys.exit(1)


if __name__ == "__main__":
    main()

