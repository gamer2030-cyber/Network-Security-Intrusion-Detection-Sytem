#!/usr/bin/env python3
"""
live_data_streaming_system.py - Real-time network traffic streaming and ML prediction system

This script implements a complete live data streaming system for real-time intrusion detection
using Apache Kafka, Redis, and live network traffic capture.
"""

import os
import sys
import time
import json
import logging
import threading
import queue
import subprocess
import random
import ipaddress
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Data processing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Real-time streaming libraries
try:
    import redis
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError
except ImportError:
    print("Installing required streaming libraries...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "redis", "kafka-python", "psutil", "scapy"])
    import redis
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError

# Network monitoring
try:
    import psutil
    from scapy.all import sniff, IP, TCP, UDP, ICMP, Raw
    from scapy.layers.inet import Ether
except ImportError:
    print("Installing network monitoring libraries...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil", "scapy"])
    import psutil
    from scapy.all import sniff, IP, TCP, UDP, ICMP, Raw
    from scapy.layers.inet import Ether

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LiveDataStreamingSystem:
    """Real-time network traffic streaming and ML prediction system"""
    
    def __init__(self, config_path: str = "config/live_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        
        # Initialize components
        self.redis_client = None
        self.kafka_producer = None
        self.kafka_consumer = None
        
        # ML models and preprocessors
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        
        # Data queues and buffers
        self.packet_queue = queue.Queue(maxsize=10000)
        self.prediction_queue = queue.Queue(maxsize=1000)
        
        # System state
        self.is_running = False
        self.stats = {
            'packets_processed': 0,
            'predictions_made': 0,
            'threats_detected': 0,
            'false_positives_filtered': 0,
            'start_time': None,
            'total_bytes_processed': 0,
            'last_bytes_time': None,
            'last_bytes_count': 0
        }
        
        # Session tracking for flow-based features
        self.sessions = {}  # Track connections by (src_ip, dst_ip, src_port, dst_port)
        self.session_timeout = 300  # 5 minutes
        
        # Whitelist and filtering
        self.whitelist = self._initialize_whitelist()
        
    def _initialize_whitelist(self) -> Dict[str, Any]:
        """Initialize whitelist for filtering false positives"""
        whitelist_config = self.config.get('whitelist', {})
        
        # Convert IP ranges to network objects
        safe_ip_ranges = []
        for ip_range in whitelist_config.get('safe_ip_ranges', []):
            try:
                safe_ip_ranges.append(ipaddress.ip_network(ip_range))
            except ValueError:
                logger.warning(f"Invalid IP range in whitelist: {ip_range}")
        
        return {
            'safe_ports': set(whitelist_config.get('safe_ports', [80, 443, 993, 995, 22, 21, 25, 53, 110, 143])),
            'safe_protocols': set(whitelist_config.get('safe_protocols', ['TCP', 'UDP'])),
            'safe_ip_ranges': safe_ip_ranges,
            'trusted_domains': set(whitelist_config.get('trusted_domains', []))
        }
    
    def _is_whitelisted(self, packet_info: Dict[str, Any]) -> bool:
        """Check if packet should be whitelisted (not flagged as threat)"""
        try:
            # Check if port is safe
            src_port = packet_info.get('src_port', 0)
            dst_port = packet_info.get('dst_port', 0)
            if src_port in self.whitelist['safe_ports'] or dst_port in self.whitelist['safe_ports']:
                return True
            
            # Check if protocol is safe
            protocol = packet_info.get('protocol', '')
            if protocol in self.whitelist['safe_protocols']:
                return True
            
            # Check if IP is in safe ranges
            src_ip = packet_info.get('src_ip', '')
            dst_ip = packet_info.get('dst_ip', '')
            
            for ip_range in self.whitelist['safe_ip_ranges']:
                try:
                    if ipaddress.ip_address(src_ip) in ip_range or ipaddress.ip_address(dst_ip) in ip_range:
                        return True
                except ValueError:
                    continue
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking whitelist: {e}")
            return False
    
    def _get_model_threshold(self, model_name: str) -> float:
        """Get model-specific threshold from config"""
        model_thresholds = self.config.get('ml', {}).get('model_thresholds', {})
        return model_thresholds.get(model_name, self.config.get('ml', {}).get('prediction_threshold', 0.9))
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        import yaml
        
        default_config = {
            'kafka': {
                'bootstrap_servers': ['localhost:9092'],
                'topic_packets': 'network_packets',
                'topic_predictions': 'ml_predictions',
                'topic_alerts': 'security_alerts'
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0
            },
            'network': {
                'interface': 'eth0',  # Change to your network interface
                'packet_count': 1000,  # Number of packets to capture per batch
                'timeout': 1.0  # Seconds to wait for packets
            },
            'ml': {
                'model_path': 'models',
                'prediction_threshold': 0.7,
                'batch_size': 100
            },
            'monitoring': {
                'update_interval': 5.0,  # Seconds between status updates
                'alert_cooldown': 30.0  # Seconds between duplicate alerts
            }
        }
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if subkey not in config[key]:
                                config[key][subkey] = subvalue
        else:
            config = default_config
            # Create config directory and file
            self.config_path.parent.mkdir(exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        
        return config
    
    def initialize_redis(self) -> bool:
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=self.config['redis']['host'],
                port=self.config['redis']['port'],
                db=self.config['redis']['db'],
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("✅ Redis connection established")
            return True
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            return False
    
    def initialize_kafka(self) -> bool:
        """Initialize Kafka producer and consumer"""
        try:
            # Initialize producer with retry and error handling
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=self.config['kafka']['bootstrap_servers'],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                retries=5,  # Retry failed sends
                max_in_flight_requests_per_connection=1,  # Ensure ordering
                acks='all',  # Wait for all replicas (but we only have 1)
                request_timeout_ms=30000,
                metadata_max_age_ms=300000,  # Refresh metadata every 5 minutes
                retry_backoff_ms=100,  # Wait 100ms between retries
                api_version=(0, 10, 1)  # Use compatible API version
            )
            
            # Initialize consumer
            self.kafka_consumer = KafkaConsumer(
                self.config['kafka']['topic_packets'],
                bootstrap_servers=self.config['kafka']['bootstrap_servers'],
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                consumer_timeout_ms=1000  # Don't block forever
            )
            
            logger.info("✅ Kafka connections established")
            return True
        except Exception as e:
            logger.error(f"❌ Kafka connection failed: {e}")
            return False
    
    def load_ml_models(self) -> bool:
        """Load trained ML models and preprocessors"""
        try:
            models_dir = Path(self.config['ml']['model_path'])
            
            if not models_dir.exists():
                logger.error(f"❌ Models directory not found: {models_dir}")
                return False
            
            # Load models for each dataset
            datasets = ['cicids2017', 'unsw_nb15', 'nsl_kdd']
            
            for dataset in datasets:
                self.models[dataset] = {}
                self.scalers[dataset] = {}
                self.label_encoders[dataset] = {}
                
                # Load models from the main models directory (they're .pkl files)
                model_files = list(models_dir.glob(f'{dataset}_*.pkl'))
                
                # Select only the best performing models for each dataset
                best_models = {
                    'cicids2017': [
                        'cicids2017_gradient_boosting_comprehensive.pkl',  # 100% accuracy
                        'cicids2017_bagging_comprehensive.pkl'            # 100% accuracy
                    ],
                    'unsw_nb15': [
                        'unsw_nb15_random_forest_comprehensive.pkl'       # 68% accuracy
                    ],
                    'nsl_kdd': [
                        'nsl_kdd_random_forest_comprehensive.pkl',        # 58.5% accuracy
                        'nsl_kdd_svm_rbf_comprehensive.pkl'               # 58.5% accuracy
                    ]
                }
                
                for model_filename in best_models.get(dataset, []):
                    model_file = models_dir / model_filename
                    if model_file.exists():
                        try:
                            # Try joblib first, then pickle
                            try:
                                model = joblib.load(model_file)
                            except:
                                import pickle
                                with open(model_file, 'rb') as f:
                                    model = pickle.load(f)
                            
                            model_name = model_file.stem.replace(f'{dataset}_', '')
                            self.models[dataset][model_name] = model
                            logger.info(f"✅ Loaded model: {dataset}/{model_name}")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to load model {model_file}: {e}")
                
                # Load scalers and encoders from processed_datasets
                scaler_file = Path('processed_datasets') / dataset / 'preprocessors' / f'{dataset}_scaler_scaler.pkl'
                encoder_file = Path('processed_datasets') / dataset / 'preprocessors' / f'{dataset}_label_encoder.pkl'
                
                if scaler_file.exists():
                    try:
                        import pickle
                        with open(scaler_file, 'rb') as f:
                            self.scalers[dataset] = pickle.load(f)
                        logger.info(f"✅ Loaded scaler for {dataset}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load scaler for {dataset}: {e}")
                        
                if encoder_file.exists():
                    try:
                        import pickle
                        with open(encoder_file, 'rb') as f:
                            self.label_encoders[dataset] = pickle.load(f)
                        logger.info(f"✅ Loaded label encoder for {dataset}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load label encoder for {dataset}: {e}")
            
            logger.info(f"✅ Loaded ML models for {len(self.models)} datasets")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load ML models: {e}")
            return False
    
    def extract_packet_features(self, packet) -> Dict[str, Any]:
        """Extract comprehensive features from network packet for real threat detection"""
        features = {
            'timestamp': datetime.now().isoformat(),
            'src_ip': None,
            'dst_ip': None,
            'src_port': None,
            'dst_port': None,
            'protocol': None,
            'packet_size': len(packet),
            'flags': None,
            'ttl': None,
            'window_size': None,
            'urgent_pointer': None,
            'options': None,
            'fragment_offset': None,
            'header_length': None,
            'total_length': None,
            'identification': None,
            'checksum': None,
            'service': None,
            'flag': None,
            'land': None,
            'wrong_fragment': None,
            'urgent': None,
            'hot': None,
            'num_failed_logins': None,
            'logged_in': None,
            'num_compromised': None,
            'root_shell': None,
            'su_attempted': None,
            'num_root': None,
            'num_file_creations': None,
            'num_shells': None,
            'num_access_files': None,
            'num_outbound_cmds': None,
            'is_host_login': None,
            'is_guest_login': None,
            'count': None,
            'srv_count': None,
            'serror_rate': None,
            'srv_serror_rate': None,
            'rerror_rate': None,
            'srv_rerror_rate': None,
            'same_srv_rate': None,
            'diff_srv_rate': None,
            'srv_diff_host_rate': None,
            'dst_host_count': None,
            'dst_host_srv_count': None,
            'dst_host_same_srv_rate': None,
            'dst_host_diff_srv_rate': None,
            'dst_host_same_src_port_rate': None,
            'dst_host_srv_diff_host_rate': None,
            'dst_host_serror_rate': None,
            'dst_host_srv_serror_rate': None,
            'dst_host_rerror_rate': None,
            'dst_host_srv_rerror_rate': None
        }
        
        try:
            # Extract IP layer information
            if packet.haslayer(IP):
                ip_layer = packet[IP]
                features['src_ip'] = str(ip_layer.src)
                features['dst_ip'] = str(ip_layer.dst)
                features['ttl'] = int(ip_layer.ttl)
                features['protocol'] = int(ip_layer.proto)
                features['header_length'] = int(ip_layer.ihl) * 4
                features['total_length'] = int(ip_layer.len)
                features['identification'] = int(ip_layer.id)
                features['fragment_offset'] = int(ip_layer.frag)
                features['checksum'] = int(ip_layer.chksum)
            
            # Extract TCP layer information
            if packet.haslayer(TCP):
                tcp_layer = packet[TCP]
                features['src_port'] = int(tcp_layer.sport)
                features['dst_port'] = int(tcp_layer.dport)
                features['flags'] = str(tcp_layer.flags)
                features['window_size'] = int(tcp_layer.window)
                features['urgent_pointer'] = int(tcp_layer.urgptr)
                
                # Map common services
                if features['dst_port'] in [80, 8080, 8000]:
                    features['service'] = 'http'
                elif features['dst_port'] in [443, 8443]:
                    features['service'] = 'https'
                
                # Extract HTTP URL/domain if available (for browser threat detection)
                if features['dst_port'] in [80, 8080, 8000] and packet.haslayer(Raw):
                    try:
                        raw_data = packet[Raw].load
                        http_data = raw_data.decode('utf-8', errors='ignore')
                        if http_data.startswith(('GET ', 'POST ', 'PUT ', 'DELETE ')):
                            lines = http_data.split('\r\n')
                            for line in lines:
                                if line.lower().startswith('host:'):
                                    host = line.split(':', 1)[1].strip().split(':')[0]
                                    features['http_host'] = host
                                    break
                    except:
                        pass
                elif features['dst_port'] == 22:
                    features['service'] = 'ssh'
                elif features['dst_port'] == 21:
                    features['service'] = 'ftp'
                elif features['dst_port'] == 25:
                    features['service'] = 'smtp'
                elif features['dst_port'] == 53:
                    features['service'] = 'dns'
                elif features['dst_port'] == 110:
                    features['service'] = 'pop3'
                elif features['dst_port'] == 143:
                    features['service'] = 'imap'
                else:
                    features['service'] = 'other'
                
                # TCP flag analysis
                flags = tcp_layer.flags
                features['urgent'] = 1 if flags & 0x20 else 0
                features['flag'] = 'SF' if flags & 0x02 and flags & 0x10 else 'S0' if flags & 0x02 else 'REJ' if flags & 0x04 else 'RSTR' if flags & 0x14 else 'RSTO' if flags & 0x04 else 'SH' if flags & 0x08 else 'S1' if flags & 0x10 else 'S2' if flags & 0x18 else 'S3' if flags & 0x1C else 'OTH'
            
            # Extract UDP layer information
            elif packet.haslayer(UDP):
                udp_layer = packet[UDP]
                features['src_port'] = int(udp_layer.sport)
                features['dst_port'] = int(udp_layer.dport)
                features['flag'] = 'SF'
                
                # Map UDP services
                if features['dst_port'] == 53:
                    features['service'] = 'dns'
                elif features['dst_port'] == 67 or features['dst_port'] == 68:
                    features['service'] = 'dhcp'
                elif features['dst_port'] == 123:
                    features['service'] = 'ntp'
                else:
                    features['service'] = 'other'
            
            # Extract ICMP layer information
            elif packet.haslayer(ICMP):
                icmp_layer = packet[ICMP]
                features['protocol'] = 1
                features['flag'] = 'SF'
                features['service'] = 'icmp'
            
            # Calculate derived features with None checks
            features['land'] = 1 if (features['src_ip'] == features['dst_ip'] and 
                                   features['src_port'] == features['dst_port'] and 
                                   features['src_ip'] is not None and features['dst_ip'] is not None and
                                   features['src_port'] is not None and features['dst_port'] is not None) else 0
            
            features['wrong_fragment'] = 1 if (features['fragment_offset'] is not None and 
                                            features['fragment_offset'] > 0) else 0
            
        except Exception as e:
            logger.warning(f"⚠️ Error extracting packet features: {e}")
        
        return features
    
    def update_session_features(self, packet_features: Dict[str, Any]) -> Dict[str, Any]:
        """Update session-based features for flow analysis"""
        try:
            # Create session key
            session_key = (
                packet_features.get('src_ip'),
                packet_features.get('dst_ip'), 
                packet_features.get('src_port'),
                packet_features.get('dst_port')
            )
            
            current_time = time.time()
            
            # Initialize or update session
            if session_key not in self.sessions:
                self.sessions[session_key] = {
                    'start_time': current_time,
                    'packet_count': 0,
                    'total_bytes': 0,
                    'flags_seen': set(),
                    'last_packet_time': current_time
                }
            
            session = self.sessions[session_key]
            session['packet_count'] += 1
            session['total_bytes'] += packet_features.get('packet_size', 0)
            session['last_packet_time'] = current_time
            
            # Track TCP flags
            if packet_features.get('flags'):
                session['flags_seen'].add(packet_features['flags'])
            
            # Calculate flow features
            duration = current_time - session['start_time']
            packet_features['duration'] = duration
            packet_features['count'] = session['packet_count']
            packet_features['src_bytes'] = session['total_bytes']
            packet_features['dst_bytes'] = 0  # Would need bidirectional tracking
            
            # Calculate rates
            if duration > 0:
                packet_features['packet_rate'] = session['packet_count'] / duration
                packet_features['byte_rate'] = session['total_bytes'] / duration
            else:
                packet_features['packet_rate'] = 0
                packet_features['byte_rate'] = 0
            
            # Clean up old sessions
            self.cleanup_old_sessions(current_time)
            
            return packet_features
            
        except Exception as e:
            logger.warning(f"⚠️ Session update error: {e}")
            return packet_features
    
    def cleanup_old_sessions(self, current_time: float):
        """Remove expired sessions"""
        try:
            expired_keys = [
                key for key, session in self.sessions.items()
                if current_time - session['last_packet_time'] > self.session_timeout
            ]
            for key in expired_keys:
                del self.sessions[key]
        except Exception as e:
            logger.warning(f"⚠️ Session cleanup error: {e}")
    
    def packet_capture_thread(self):
        """Thread for capturing network packets"""
        logger.info("🔍 Starting packet capture thread...")
        
        def packet_handler(packet):
            """Handle captured packet"""
            try:
                features = self.extract_packet_features(packet)
                # Update session-based features for flow analysis
                features = self.update_session_features(features)
                
                # Track bytes processed
                packet_size = features.get('packet_size', len(packet) if packet else 0)
                self.stats['total_bytes_processed'] += packet_size
                
                self.packet_queue.put(features, timeout=1)
                self.stats['packets_processed'] += 1
            except queue.Full:
                logger.warning("⚠️ Packet queue full, dropping packet")
            except Exception as e:
                logger.warning(f"⚠️ Error processing packet: {e}")
        
        try:
            # Start continuous packet capture (remove count limit)
            while self.is_running:
                try:
                    sniff(
                        iface=self.config['network']['interface'],
                        prn=packet_handler,
                        count=100,  # Process in smaller batches
                        timeout=self.config['network']['timeout']
                    )
                except Exception as sniff_error:
                    logger.warning(f"⚠️ Sniff error on {self.config['network']['interface']}: {sniff_error}")
                    # Try without promiscuous mode
                    try:
                        sniff(
                            iface=self.config['network']['interface'],
                            prn=packet_handler,
                            count=100,
                            timeout=self.config['network']['timeout'],
                            promisc=False
                        )
                    except Exception as e2:
                        logger.error(f"❌ Packet capture failed completely: {e2}")
                        time.sleep(5)  # Wait before retrying
        except Exception as e:
            logger.error(f"❌ Packet capture thread error: {e}")
    
    def packet_processing_thread(self):
        """Thread for processing captured packets"""
        logger.info("⚙️ Starting packet processing thread...")
        
        kafka_failures = 0
        max_kafka_failures = 5
        
        while self.is_running:
            try:
                # Collect packets in batches
                batch = []
                batch_size = self.config['ml']['batch_size']
                
                for _ in range(batch_size):
                    try:
                        packet = self.packet_queue.get(timeout=1)
                        batch.append(packet)
                    except queue.Empty:
                        break
                
                if batch:
                    # Try to send batch to Kafka
                    kafka_success = self.send_packet_batch(batch)
                    
                    # If Kafka is failing, process directly
                    if not kafka_success:
                        kafka_failures += 1
                        if kafka_failures >= max_kafka_failures:
                            logger.warning(f"⚠️ Kafka failing ({kafka_failures} times), processing packets directly...")
                            # Process batch directly when Kafka is down
                            batch_data = {
                                'batch_id': f"batch_{int(time.time())}",
                                'timestamp': datetime.now().isoformat(),
                                'packet_count': len(batch),
                                'packets': batch
                            }
                            self.process_packet_batch(batch_data)
                    else:
                        kafka_failures = 0  # Reset failure counter on success
                    
            except Exception as e:
                logger.error(f"❌ Packet processing error: {e}")
                time.sleep(1)
    
    def send_packet_batch(self, batch: List[Dict[str, Any]]) -> bool:
        """Send packet batch to Kafka. Returns True if successful, False otherwise."""
        try:
            batch_data = {
                'batch_id': f"batch_{int(time.time())}",
                'timestamp': datetime.now().isoformat(),
                'packet_count': len(batch),
                'packets': batch
            }
            
            if not self.kafka_producer:
                return False
                
            try:
                future = self.kafka_producer.send(
                    self.config['kafka']['topic_packets'],
                    value=batch_data,
                    key=batch_data['batch_id']
                )
                # Wait briefly to check if send succeeded
                try:
                    record_metadata = future.get(timeout=2)  # Wait up to 2 seconds
                    logger.debug(f"📤 Sent batch {batch_data['batch_id']} with {len(batch)} packets")
                    return True
                except Exception as get_error:
                    # Send failed
                    logger.debug(f"⚠️ Kafka send failed (will retry direct processing): {get_error}")
                    return False
            except Exception as send_error:
                logger.debug(f"⚠️ Failed to send packet batch: {send_error}")
                # Refresh metadata on error
                try:
                    self.kafka_producer.metadata_refresh()
                except:
                    pass
                return False
            
        except Exception as e:
            logger.debug(f"❌ Failed to send packet batch: {e}")
            return False
    
    def ml_prediction_thread(self):
        """Thread for ML predictions on packet data"""
        logger.info("🤖 Starting ML prediction thread...")
        
        while self.is_running:
            try:
                # Consume packets from Kafka
                message_batch = self.kafka_consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        batch_data = message.value
                        self.process_packet_batch(batch_data)
                        
            except Exception as e:
                logger.error(f"❌ ML prediction error: {e}")
                time.sleep(1)
    
    def process_packet_batch(self, batch_data: Dict[str, Any]):
        """Process packet batch for ML prediction"""
        try:
            packets = batch_data['packets']
            
            # Track bytes from batch (if not already tracked)
            for packet in packets:
                packet_size = packet.get('packet_size', 0)
                if packet_size > 0:
                    # Only add if not already counted (to avoid double counting)
                    pass  # Bytes are already tracked in capture thread
            
            # Convert packets to DataFrame
            df = pd.DataFrame(packets)
            
            # Process each dataset's models
            for dataset, models in self.models.items():
                if not models:
                    continue
                
                # Prepare features (simplified for demo)
                features = self.prepare_features_for_dataset(df, dataset)
                
                if features is None or len(features) == 0:
                    continue
                
                # Make predictions with each model
                for model_name, model in models.items():
                    try:
                        predictions = model.predict(features)
                        probabilities = model.predict_proba(features) if hasattr(model, 'predict_proba') else None
                        
                        # Process predictions
                        self.process_predictions(
                            dataset, model_name, predictions, probabilities, 
                            batch_data, features
                        )
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Prediction error for {dataset}/{model_name}: {e}")
            
        except Exception as e:
            logger.error(f"❌ Batch processing error: {e}")
    
    def prepare_features_for_dataset(self, df: pd.DataFrame, dataset: str) -> Optional[np.ndarray]:
        """Prepare real packet features for specific dataset model"""
        try:
            n_samples = len(df)
            
            # Handle NaN values in the dataframe
            df = df.fillna(0)
            
            # Define feature mappings for each dataset
            if dataset == 'nsl_kdd':
                # NSL-KDD has 41 features
                feature_matrix = np.zeros((n_samples, 41))
                
                for i, row in df.iterrows():
                    # Basic connection features
                    feature_matrix[i, 0] = row.get('duration', 0)  # duration
                    feature_matrix[i, 1] = row.get('protocol', 0)  # protocol_type
                    feature_matrix[i, 2] = row.get('service', 0) if isinstance(row.get('service'), (int, float)) else 0  # service
                    feature_matrix[i, 3] = row.get('flag', 0) if isinstance(row.get('flag'), (int, float)) else 0  # flag
                    feature_matrix[i, 4] = row.get('src_bytes', row.get('packet_size', 0))  # src_bytes
                    feature_matrix[i, 5] = row.get('dst_bytes', 0)  # dst_bytes
                    feature_matrix[i, 6] = row.get('land', 0)  # land
                    feature_matrix[i, 7] = row.get('wrong_fragment', 0)  # wrong_fragment
                    feature_matrix[i, 8] = row.get('urgent', 0)  # urgent
                    
                    # Content features (set to 0 for single packet analysis)
                    feature_matrix[i, 9:23] = 0  # hot, num_failed_logins, etc.
                    
                    # Traffic features (set to 0 for single packet analysis)
                    feature_matrix[i, 23:31] = 0  # count, srv_count, etc.
                    
                    # Host-based features (set to 0 for single packet analysis)
                    feature_matrix[i, 31:41] = 0  # dst_host_* features
                    
            elif dataset == 'unsw_nb15':
                # UNSW-NB15 has 49 features
                feature_matrix = np.zeros((n_samples, 49))
                
                for i, row in df.iterrows():
                    # Basic features
                    feature_matrix[i, 0] = row.get('srcip', 0) if isinstance(row.get('srcip'), (int, float)) else 0
                    feature_matrix[i, 1] = row.get('sport', row.get('src_port', 0))
                    feature_matrix[i, 2] = row.get('dstip', 0) if isinstance(row.get('dstip'), (int, float)) else 0
                    feature_matrix[i, 3] = row.get('dsport', row.get('dst_port', 0))
                    feature_matrix[i, 4] = row.get('proto', row.get('protocol', 0))
                    feature_matrix[i, 5] = row.get('state', 0) if isinstance(row.get('state'), (int, float)) else 0
                    feature_matrix[i, 6] = row.get('dur', 0)  # duration
                    feature_matrix[i, 7] = row.get('sbytes', row.get('packet_size', 0))
                    feature_matrix[i, 8] = row.get('dbytes', 0)
                    feature_matrix[i, 9] = row.get('sttl', row.get('ttl', 64))
                    feature_matrix[i, 10] = row.get('dttl', 64)
                    feature_matrix[i, 11] = row.get('sloss', 0)
                    feature_matrix[i, 12] = row.get('dloss', 0)
                    feature_matrix[i, 13] = row.get('service', 0) if isinstance(row.get('service'), (int, float)) else 0
                    feature_matrix[i, 14] = row.get('sload', 0)
                    feature_matrix[i, 15] = row.get('dload', 0)
                    feature_matrix[i, 16] = row.get('spkts', 1)
                    feature_matrix[i, 17] = row.get('dpkts', 0)
                    feature_matrix[i, 18] = row.get('swin', row.get('window_size', 0))
                    feature_matrix[i, 19] = row.get('dwin', 0)
                    feature_matrix[i, 20] = row.get('stcpb', 0)
                    feature_matrix[i, 21] = row.get('dtcpb', 0)
                    feature_matrix[i, 22] = row.get('smeansz', row.get('packet_size', 0))
                    feature_matrix[i, 23] = row.get('dmeansz', 0)
                    feature_matrix[i, 24] = row.get('trans_depth', 0)
                    feature_matrix[i, 25] = row.get('res_bdy_len', 0)
                    feature_matrix[i, 26] = row.get('sjit', 0)
                    feature_matrix[i, 27] = row.get('djit', 0)
                    feature_matrix[i, 28] = row.get('stime', 0)
                    feature_matrix[i, 29] = row.get('ltime', 0)
                    feature_matrix[i, 30] = row.get('sinpkt', 0)
                    feature_matrix[i, 31] = row.get('dinpkt', 0)
                    feature_matrix[i, 32] = row.get('tcprtt', 0)
                    feature_matrix[i, 33] = row.get('synack', 0)
                    feature_matrix[i, 34] = row.get('ackdat', 0)
                    feature_matrix[i, 35] = row.get('smean', 0)
                    feature_matrix[i, 36] = row.get('dmean', 0)
                    feature_matrix[i, 37] = row.get('response_body_len', 0)
                    feature_matrix[i, 38] = row.get('ct_srv_src', 0)
                    feature_matrix[i, 39] = row.get('ct_state_ttl', 0)
                    feature_matrix[i, 40] = row.get('ct_dst_ltm', 0)
                    feature_matrix[i, 41] = row.get('ct_src_dport_ltm', 0)
                    feature_matrix[i, 42] = row.get('ct_dst_sport_ltm', 0)
                    feature_matrix[i, 43] = row.get('ct_dst_src_ltm', 0)
                    feature_matrix[i, 44] = row.get('is_ftp_login', 0)
                    feature_matrix[i, 45] = row.get('ct_ftp_cmd', 0)
                    feature_matrix[i, 46] = row.get('ct_flw_http_mthd', 0)
                    feature_matrix[i, 47] = row.get('ct_src_ltm', 0)
                    feature_matrix[i, 48] = row.get('ct_srv_dst', 0)
                    
            elif dataset == 'cicids2017':
                # CICIDS2017 has 78 features
                feature_matrix = np.zeros((n_samples, 78))
                
                for i, row in df.iterrows():
                    # Basic flow features
                    feature_matrix[i, 0] = row.get('Destination Port', row.get('dst_port', 0))
                    feature_matrix[i, 1] = row.get('Flow Duration', 0)
                    feature_matrix[i, 2] = row.get('Total Fwd Packets', 1)
                    feature_matrix[i, 3] = row.get('Total Backward Packets', 0)
                    feature_matrix[i, 4] = row.get('Total Length of Fwd Packets', row.get('packet_size', 0))
                    feature_matrix[i, 5] = row.get('Total Length of Bwd Packets', 0)
                    feature_matrix[i, 6] = row.get('Fwd Packet Length Max', row.get('packet_size', 0))
                    feature_matrix[i, 7] = row.get('Fwd Packet Length Min', row.get('packet_size', 0))
                    feature_matrix[i, 8] = row.get('Fwd Packet Length Mean', row.get('packet_size', 0))
                    feature_matrix[i, 9] = row.get('Fwd Packet Length Std', 0)
                    feature_matrix[i, 10] = row.get('Bwd Packet Length Max', 0)
                    feature_matrix[i, 11] = row.get('Bwd Packet Length Min', 0)
                    feature_matrix[i, 12] = row.get('Bwd Packet Length Mean', 0)
                    feature_matrix[i, 13] = row.get('Bwd Packet Length Std', 0)
                    feature_matrix[i, 14] = row.get('Flow Bytes/s', 0)
                    feature_matrix[i, 15] = row.get('Flow Packets/s', 0)
                    feature_matrix[i, 16] = row.get('Flow IAT Mean', 0)
                    feature_matrix[i, 17] = row.get('Flow IAT Std', 0)
                    feature_matrix[i, 18] = row.get('Flow IAT Max', 0)
                    feature_matrix[i, 19] = row.get('Flow IAT Min', 0)
                    feature_matrix[i, 20] = row.get('Fwd IAT Total', 0)
                    feature_matrix[i, 21] = row.get('Fwd IAT Mean', 0)
                    feature_matrix[i, 22] = row.get('Fwd IAT Std', 0)
                    feature_matrix[i, 23] = row.get('Fwd IAT Max', 0)
                    feature_matrix[i, 24] = row.get('Fwd IAT Min', 0)
                    feature_matrix[i, 25] = row.get('Bwd IAT Total', 0)
                    feature_matrix[i, 26] = row.get('Bwd IAT Mean', 0)
                    feature_matrix[i, 27] = row.get('Bwd IAT Std', 0)
                    feature_matrix[i, 28] = row.get('Bwd IAT Max', 0)
                    feature_matrix[i, 29] = row.get('Bwd IAT Min', 0)
                    feature_matrix[i, 30] = row.get('Fwd PSH Flags', 0)
                    feature_matrix[i, 31] = row.get('Bwd PSH Flags', 0)
                    feature_matrix[i, 32] = row.get('Fwd URG Flags', row.get('urgent', 0))
                    feature_matrix[i, 33] = row.get('Bwd URG Flags', 0)
                    feature_matrix[i, 34] = row.get('Fwd Header Length', row.get('header_length', 20))
                    feature_matrix[i, 35] = row.get('Bwd Header Length', 0)
                    feature_matrix[i, 36] = row.get('Fwd Packets/s', 0)
                    feature_matrix[i, 37] = row.get('Bwd Packets/s', 0)
                    feature_matrix[i, 38] = row.get('Min Packet Length', row.get('packet_size', 0))
                    feature_matrix[i, 39] = row.get('Max Packet Length', row.get('packet_size', 0))
                    feature_matrix[i, 40] = row.get('Packet Length Mean', row.get('packet_size', 0))
                    feature_matrix[i, 41] = row.get('Packet Length Std', 0)
                    feature_matrix[i, 42] = row.get('Packet Length Variance', 0)
                    feature_matrix[i, 43] = row.get('FIN Flag Count', 1 if 'F' in str(row.get('flags', '')) else 0)
                    feature_matrix[i, 44] = row.get('SYN Flag Count', 1 if 'S' in str(row.get('flags', '')) else 0)
                    feature_matrix[i, 45] = row.get('RST Flag Count', 1 if 'R' in str(row.get('flags', '')) else 0)
                    feature_matrix[i, 46] = row.get('PSH Flag Count', 1 if 'P' in str(row.get('flags', '')) else 0)
                    feature_matrix[i, 47] = row.get('ACK Flag Count', 1 if 'A' in str(row.get('flags', '')) else 0)
                    feature_matrix[i, 48] = row.get('URG Flag Count', row.get('urgent', 0))
                    feature_matrix[i, 49] = row.get('CWE Flag Count', 0)
                    feature_matrix[i, 50] = row.get('ECE Flag Count', 0)
                    feature_matrix[i, 51] = row.get('Down/Up Ratio', 0)
                    feature_matrix[i, 52] = row.get('Average Packet Size', row.get('packet_size', 0))
                    feature_matrix[i, 53] = row.get('Avg Fwd Segment Size', row.get('packet_size', 0))
                    feature_matrix[i, 54] = row.get('Avg Bwd Segment Size', 0)
                    feature_matrix[i, 55] = row.get('Fwd Header Length.1', row.get('header_length', 20))
                    feature_matrix[i, 56] = row.get('Fwd Avg Bytes/Bulk', 0)
                    feature_matrix[i, 57] = row.get('Fwd Avg Packets/Bulk', 0)
                    feature_matrix[i, 58] = row.get('Fwd Avg Bulk Rate', 0)
                    feature_matrix[i, 59] = row.get('Bwd Avg Bytes/Bulk', 0)
                    feature_matrix[i, 60] = row.get('Bwd Avg Packets/Bulk', 0)
                    feature_matrix[i, 61] = row.get('Bwd Avg Bulk Rate', 0)
                    feature_matrix[i, 62] = row.get('Subflow Fwd Packets', 1)
                    feature_matrix[i, 63] = row.get('Subflow Fwd Bytes', row.get('packet_size', 0))
                    feature_matrix[i, 64] = row.get('Subflow Bwd Packets', 0)
                    feature_matrix[i, 65] = row.get('Subflow Bwd Bytes', 0)
                    feature_matrix[i, 66] = row.get('Init_Win_bytes_forward', row.get('window_size', 0))
                    feature_matrix[i, 67] = row.get('Init_Win_bytes_backward', 0)
                    feature_matrix[i, 68] = row.get('act_data_pkt_fwd', 0)
                    feature_matrix[i, 69] = row.get('min_seg_size_forward', 0)
                    feature_matrix[i, 70] = row.get('Active Mean', 0)
                    feature_matrix[i, 71] = row.get('Active Std', 0)
                    feature_matrix[i, 72] = row.get('Active Max', 0)
                    feature_matrix[i, 73] = row.get('Active Min', 0)
                    feature_matrix[i, 74] = row.get('Idle Mean', 0)
                    feature_matrix[i, 75] = row.get('Idle Std', 0)
                    feature_matrix[i, 76] = row.get('Idle Max', 0)
                    feature_matrix[i, 77] = row.get('Idle Min', 0)
            
            # Handle any remaining NaN values
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale features if scaler is available
            if dataset in self.scalers and self.scalers[dataset]:
                try:
                    feature_matrix = self.scalers[dataset].transform(feature_matrix)
                    # Handle NaN values after scaling
                    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
                except Exception as e:
                    logger.warning(f"⚠️ Scaling error for {dataset}: {e}")
            
            return feature_matrix
            
        except Exception as e:
            logger.warning(f"⚠️ Feature preparation error for {dataset}: {e}")
            return None
    
    def process_predictions(self, dataset: str, model_name: str, predictions: np.ndarray, 
                          probabilities: Optional[np.ndarray], batch_data: Dict[str, Any], 
                          features: np.ndarray):
        """Process ML predictions and generate alerts"""
        try:
            # Use model-specific threshold
            threshold = self._get_model_threshold(model_name)
            
            for i, prediction in enumerate(predictions):
                is_threat = bool(prediction != 0)  # Convert numpy bool_ to Python bool
                confidence = float(probabilities[i].max()) if probabilities is not None else 0.5
                
                # Get packet info for this prediction
                packet_info = {}
                if i < len(batch_data['packets']):
                    packet = batch_data['packets'][i]
                    protocol_num = packet.get('protocol', 0)
                    protocol_name = self._get_protocol_name(protocol_num)
                    
                    packet_info = {
                        'src_ip': packet.get('src_ip', 'unknown'),
                        'dst_ip': packet.get('dst_ip', 'unknown'),
                        'protocol': protocol_name,
                        'src_port': packet.get('src_port', 0),
                        'dst_port': packet.get('dst_port', 0)
                    }
                
                prediction_data = {
                    'timestamp': datetime.now().isoformat(),
                    'dataset': dataset,
                    'model': model_name,
                    'prediction': int(prediction),
                    'confidence': confidence,
                    'is_threat': is_threat,
                    'batch_id': batch_data['batch_id'],
                    'packet_index': i,
                    **packet_info  # Include packet info
                }
                
                # Store prediction in Redis
                self.store_prediction(prediction_data)
                
                # Send to prediction topic with error handling
                try:
                    future = self.kafka_producer.send(
                        self.config['kafka']['topic_predictions'],
                        value=prediction_data,
                        key=f"{dataset}_{model_name}"
                    )
                    # Don't wait for result to avoid blocking
                    future.add_errback(lambda e: logger.debug(f"⚠️ Kafka prediction send error (non-critical): {e}"))
                except Exception as send_error:
                    logger.debug(f"⚠️ Failed to send prediction (non-critical): {send_error}")
                    # Refresh metadata on error
                    try:
                        self.kafka_producer.metadata_refresh()
                    except:
                        pass
                
                # Count ALL threats detected (regardless of confidence or whitelist)
                # This matches what the dashboard displays
                if is_threat:
                    self.stats['threats_detected'] += 1
                
                # Generate alert only if threat detected AND passes confidence threshold
                # Whitelist filtering only affects alerts, not threat counting
                if is_threat and confidence > threshold:
                    # Check whitelist before generating alert
                    if self._is_whitelisted(packet_info):
                        self.stats['false_positives_filtered'] += 1
                        logger.debug(f"🔒 Whitelisted packet: {packet_info.get('src_ip')} -> {packet_info.get('dst_ip')} ({packet_info.get('protocol')})")
                    else:
                        self.generate_alert(prediction_data)
                
                self.stats['predictions_made'] += 1
            
        except Exception as e:
            logger.error(f"❌ Prediction processing error: {e}")
    
    def store_prediction(self, prediction_data: Dict[str, Any]):
        """Store prediction in Redis"""
        try:
            if self.redis_client:
                key = f"prediction:{prediction_data['timestamp']}"
                self.redis_client.setex(key, 3600, json.dumps(prediction_data))  # Expire in 1 hour
                
                # Add to recent predictions list
                self.redis_client.lpush("recent_predictions", json.dumps(prediction_data))
                self.redis_client.ltrim("recent_predictions", 0, 999)  # Keep last 1000
                
        except Exception as e:
            logger.warning(f"⚠️ Redis storage error: {e}")
    
    def _get_protocol_name(self, protocol_num: int) -> str:
        """Convert protocol number to name"""
        protocol_map = {
            1: 'ICMP',
            6: 'TCP', 
            17: 'UDP',
            41: 'IPv6',
            47: 'GRE',
            50: 'ESP',
            51: 'AH',
            89: 'OSPF'
        }
        return protocol_map.get(protocol_num, f'Protocol-{protocol_num}')
    
    def _get_threat_description(self, dataset: str, model: str, prediction: int, confidence: float) -> str:
        """Generate specific threat description based on dataset and prediction"""
        threat_descriptions = {
            'nsl_kdd': {
                0: 'Normal traffic',
                1: 'DoS attack detected',
                2: 'DDoS attack detected', 
                3: 'Network probe/reconnaissance',
                4: 'Remote-to-Local unauthorized access'
            },
            'unsw_nb15': {
                0: 'Normal traffic',
                1: 'Fuzzer attack detected',
                2: 'Analysis attack detected',
                3: 'Backdoor detected',
                4: 'DoS attack detected',
                5: 'Exploit detected',
                6: 'Generic attack detected',
                7: 'Reconnaissance detected',
                8: 'Shellcode detected',
                9: 'Worm detected'
            },
            'cicids2017': {
                0: 'Normal traffic',
                1: 'Brute Force attack detected',
                2: 'Heartbleed attack detected',
                3: 'Botnet communication detected',
                4: 'DDoS attack detected'
            }
        }
        
        dataset_threats = threat_descriptions.get(dataset, {})
        threat_name = dataset_threats.get(prediction, f'Unknown threat type {prediction}')
        
        return f"{threat_name} (Confidence: {confidence:.1%})"
    
    def generate_alert(self, prediction_data: Dict[str, Any]):
        """Generate security alert"""
        try:
            threat_description = self._get_threat_description(
                prediction_data['dataset'],
                prediction_data['model'],
                prediction_data['prediction'],
                prediction_data['confidence']
            )
            
            alert = {
                'alert_id': f"alert_{int(time.time())}",
                'timestamp': prediction_data['timestamp'],
                'severity': 'HIGH' if prediction_data['confidence'] > 0.9 else 'MEDIUM',
                'threat_type': f"{prediction_data['dataset']}_{prediction_data['model']}",
                'confidence': prediction_data['confidence'],
                'model': prediction_data['model'],
                'dataset': prediction_data['dataset'],
                'src_ip': prediction_data.get('src_ip', 'unknown'),
                'dst_ip': prediction_data.get('dst_ip', 'unknown'),
                'protocol': prediction_data.get('protocol', 'unknown'),
                'description': threat_description
            }
            
            # Send alert to Kafka with error handling
            try:
                future = self.kafka_producer.send(
                    self.config['kafka']['topic_alerts'],
                    value=alert,
                    key=alert['alert_id']
                )
                # Don't wait for result to avoid blocking
                future.add_errback(lambda e: logger.warning(f"⚠️ Kafka alert send error (non-critical): {e}"))
            except Exception as send_error:
                logger.warning(f"⚠️ Failed to send alert (non-critical): {send_error}")
                # Refresh metadata on error
                try:
                    self.kafka_producer.metadata_refresh()
                except:
                    pass
            
            # Store in Redis
            if self.redis_client:
                self.redis_client.setex(
                    f"alert:{alert['alert_id']}", 
                    86400,  # Expire in 24 hours
                    json.dumps(alert)
                )
            
            logger.warning(f"🚨 SECURITY ALERT: {alert['description']} (Confidence: {alert['confidence']:.3f})")
            
        except Exception as e:
            logger.error(f"❌ Alert generation error: {e}")
    
    def monitoring_thread(self):
        """Thread for system monitoring and statistics"""
        logger.info("📊 Starting monitoring thread...")
        
        # Track bandwidth calculation
        last_bandwidth_time = time.time()
        last_bandwidth_bytes = 0
        
        while self.is_running:
            try:
                # Update statistics
                uptime = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
                
                # Calculate active connections from sessions
                current_time = time.time()
                active_connections = 0
                total_session_bytes = 0
                
                # Clean up old sessions and count active ones
                active_sessions = {}
                for session_key, session_data in self.sessions.items():
                    # Check if session is still active (within timeout)
                    if current_time - session_data.get('last_seen', 0) < self.session_timeout:
                        active_sessions[session_key] = session_data
                        active_connections += 1
                        total_session_bytes += session_data.get('total_bytes', 0)
                
                # Update sessions dict (remove old ones)
                self.sessions = active_sessions
                
                # Calculate bandwidth (MB/s) from total bytes processed
                current_bytes = self.stats.get('total_bytes_processed', 0)
                time_delta = current_time - last_bandwidth_time
                
                if time_delta > 0 and last_bandwidth_time > 0:
                    bytes_delta = current_bytes - last_bandwidth_bytes
                    bandwidth_mbps = (bytes_delta / (1024 * 1024)) / time_delta  # MB/s
                else:
                    bandwidth_mbps = 0
                
                # If no bandwidth calculated yet, use session bytes as fallback
                if bandwidth_mbps == 0 and total_session_bytes > 0:
                    # Estimate from session bytes
                    bandwidth_mbps = (total_session_bytes / (1024 * 1024)) / max(uptime, 1)
                
                # Update bandwidth tracking
                last_bandwidth_bytes = current_bytes
                last_bandwidth_time = current_time
                
                stats_data = {
                    'timestamp': datetime.now().isoformat(),
                    'uptime_seconds': uptime,
                    'packets_processed': self.stats['packets_processed'],
                    'predictions_made': self.stats['predictions_made'],
                    'threats_detected': self.stats['threats_detected'],
                    'packet_rate': self.stats['packets_processed'] / max(uptime, 1),
                    'prediction_rate': self.stats['predictions_made'] / max(uptime, 1),
                    'threat_rate': self.stats['threats_detected'] / max(uptime, 1),
                    'active_connections': active_connections,
                    'bandwidth_mbps': max(0, bandwidth_mbps)  # Ensure non-negative
                }
                
                # Store in Redis
                if self.redis_client:
                    self.redis_client.setex("system_stats", 60, json.dumps(stats_data))
                
                # Log statistics
                logger.info(f"📊 Stats - Packets: {self.stats['packets_processed']}, "
                          f"Predictions: {self.stats['predictions_made']}, "
                          f"Threats: {self.stats['threats_detected']}, "
                          f"Connections: {active_connections}, "
                          f"Bandwidth: {bandwidth_mbps:.2f} MB/s")
                
                time.sleep(self.config['monitoring']['update_interval'])
                
            except Exception as e:
                logger.error(f"❌ Monitoring error: {e}")
                time.sleep(5)
    
    def start_system(self):
        """Start the live data streaming system"""
        logger.info("🚀 Starting Live Data Streaming System...")
        
        # Initialize components
        if not self.initialize_redis():
            logger.error("❌ Failed to initialize Redis")
            return False
        
        if not self.initialize_kafka():
            logger.error("❌ Failed to initialize Kafka")
            return False
        
        if not self.load_ml_models():
            logger.error("❌ Failed to load ML models")
            return False
        
        # Start system
        self.is_running = True
        self.stats['start_time'] = time.time()
        
        # Start threads
        threads = [
            threading.Thread(target=self.packet_capture_thread, daemon=True),
            threading.Thread(target=self.packet_processing_thread, daemon=True),
            threading.Thread(target=self.ml_prediction_thread, daemon=True),
            threading.Thread(target=self.monitoring_thread, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
        
        logger.info("✅ Live Data Streaming System started successfully!")
        logger.info("🔍 Monitoring network traffic for real-time intrusion detection...")
        
        return True
    
    def stop_system(self):
        """Stop the live data streaming system"""
        logger.info("🛑 Stopping Live Data Streaming System...")
        
        self.is_running = False
        
        # Close connections
        if self.kafka_producer:
            self.kafka_producer.close()
        if self.kafka_consumer:
            self.kafka_consumer.close()
        if self.redis_client:
            self.redis_client.close()
        
        logger.info("✅ Live Data Streaming System stopped")

def main():
    """Main function to run the live data streaming system"""
    logger.info("🚀 ML-IDS-IPS Live Data Streaming System")
    logger.info("=" * 60)
    
    # Create system
    system = LiveDataStreamingSystem()
    
    try:
        # Start system
        if system.start_system():
            logger.info("🎯 System running... Press Ctrl+C to stop")
            
            # Keep main thread alive
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ System error: {e}")
    finally:
        system.stop_system()
        logger.info("👋 Live Data Streaming System shutdown complete")

if __name__ == "__main__":
    main()
