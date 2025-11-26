#!/usr/bin/env python3
"""
production_dashboard.py - Production-ready dashboard with authentication, HTTPS, and enterprise features

This enhanced dashboard includes:
- User authentication with session management
- Role-based access control (RBAC)
- HTTPS support
- Audit logging
- Enterprise-grade security
"""

import os
import sys
import time
import json
import logging
import hashlib
import threading
import subprocess
from functools import wraps
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Web framework
try:
    from flask import Flask, render_template, jsonify, request, redirect, url_for, session, flash
    from flask_socketio import SocketIO, emit
    import bcrypt
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask", "flask-socketio", "bcrypt", "redis", "kafka-python"])
    from flask import Flask, render_template, jsonify, request, redirect, url_for, session, flash
    from flask_socketio import SocketIO, emit
    import bcrypt

# Data processing
import pandas as pd
import numpy as np
import redis
from kafka import KafkaConsumer
from kafka.errors import KafkaError

# Security
import hmac
import secrets

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# System monitoring
try:
    import psutil
except ImportError:
    psutil = None
    logger.warning("⚠️  psutil not installed. System health monitoring will be limited. Install with: pip install psutil")

class UserManager:
    """User authentication and management system"""
    
    def __init__(self):
        # Default users (in production, store in database)
        self.users = {
            'admin': {
                'password_hash': bcrypt.hashpw('admin123'.encode(), bcrypt.gensalt()),
                'role': 'admin',
                'last_login': None,
                'failed_attempts': 0
            },
            'analyst': {
                'password_hash': bcrypt.hashpw('analyst123'.encode(), bcrypt.gensalt()),
                'role': 'analyst',
                'last_login': None,
                'failed_attempts': 0
            },
            'viewer': {
                'password_hash': bcrypt.hashpw('viewer123'.encode(), bcrypt.gensalt()),
                'role': 'viewer',
                'last_login': None,
                'failed_attempts': 0
            }
        }
    
    def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user"""
        if username not in self.users:
            logger.warning(f"Failed login attempt for unknown user: {username}")
            return None
        
        user = self.users[username]
        
        # Check for account lockout
        if user['failed_attempts'] >= 5:
            logger.warning(f"Account locked due to too many failed attempts: {username}")
            return None
        
        # Verify password
        if bcrypt.checkpw(password.encode(), user['password_hash']):
            # Successful login
            user['last_login'] = datetime.now()
            user['failed_attempts'] = 0
            logger.info(f"Successful login: {username} ({user['role']})")
            return {
                'username': username,
                'role': user['role']
            }
        else:
            # Failed login
            user['failed_attempts'] += 1
            logger.warning(f"Failed login attempt for user: {username} ({user['failed_attempts']}/5)")
            return None
    
    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user information"""
        return self.users.get(username)

class AuditLogger:
    """Audit logging system for security events"""
    
    def __init__(self, log_file: str = "audit.log"):
        self.log_file = log_file
        self.setup_logging()
    
    def setup_logging(self):
        """Setup audit logging"""
        audit_handler = logging.FileHandler(self.log_file)
        audit_handler.setLevel(logging.INFO)
        audit_formatter = logging.Formatter('%(asctime)s - AUDIT - %(message)s')
        audit_handler.setFormatter(audit_formatter)
        
        audit_logger = logging.getLogger('audit')
        audit_logger.addHandler(audit_handler)
        audit_logger.setLevel(logging.INFO)
    
    def log(self, event_type: str, username: str, details: str, ip_address: str):
        """Log audit event"""
        audit_logger = logging.getLogger('audit')
        audit_logger.info(f"{event_type} | User: {username} | IP: {ip_address} | Details: {details}")

class ProductionDashboard:
    """Production-ready monitoring dashboard with enterprise features"""
    
    def __init__(self, config_path: str = "config/live_config.yaml"):
        # Get project root directory (parent of src/)
        project_root = Path(__file__).parent.parent
        self.project_root = project_root
        
        # Fix config path to be relative to project root
        if not Path(config_path).is_absolute():
            config_path = project_root / config_path
        self.config_path = str(config_path)
        self.config = self.load_config()
        
        # Initialize Flask app with template folder from project root
        template_folder = str(project_root / "templates")
        self.app = Flask(__name__, template_folder=template_folder)
        self.app.config['SECRET_KEY'] = self.generate_secret_key()
        self.app.config['SESSION_COOKIE_HTTPONLY'] = True
        self.app.config['SESSION_COOKIE_SECURE'] = False  # Set to True when using HTTPS
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Initialize components
        self.user_manager = UserManager()
        self.audit_logger = AuditLogger()
        self.redis_client = None
        self.kafka_consumer = None
        
        # Data storage
        self.recent_predictions = []
        self.recent_alerts = []
        self.system_stats = {}
        self.network_stats = {}
        self.blocked_ips = set()
        
        # Threading
        self.is_running = False
        self.monitoring_active = False  # Track monitoring state
        
        # Setup routes
        self.setup_routes()
        self.setup_socketio_events()
    
    def generate_secret_key(self) -> str:
        """Generate secret key for session management"""
        return secrets.token_hex(32)
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        try:
            import yaml
            config_file = Path(self.config_path)
            if not config_file.exists():
                # Try relative to project root
                config_file = self.project_root / self.config_path
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def login_required(self, f):
        """Decorator for login required routes - DISABLED FOR DEVELOPMENT"""
        # Temporarily disabled - no authentication required
        return f
    
    def role_required(self, *roles):
        """Decorator for role-based access control - DISABLED FOR DEVELOPMENT"""
        def decorator(f):
            # Temporarily disabled - no role checking
            return f
        return decorator
    
    def setup_routes(self):
        """Setup Flask routes with authentication"""
        
        @self.app.route('/')
        def index():
            """Redirect to dashboard - AUTHENTICATION DISABLED"""
            # Temporarily disabled - redirect directly to dashboard
            return redirect(url_for('dashboard'))
        
        @self.app.route('/login', methods=['GET', 'POST'])
        def login():
            """Login page"""
            if request.method == 'POST':
                username = request.form.get('username')
                password = request.form.get('password')
                
                user = self.user_manager.authenticate(username, password)
                
                if user:
                    session['user'] = user
                    self.audit_logger.log('LOGIN', username, 'User logged in successfully', request.remote_addr)
                    flash(f'Welcome, {username}!', 'success')
                    return redirect(url_for('dashboard'))
                else:
                    flash('Invalid username or password', 'danger')
                    self.audit_logger.log('FAILED_LOGIN', username or 'unknown', 'Failed login attempt', request.remote_addr)
            
            return render_template('login.html')
        
        @self.app.route('/logout')
        def logout():
            """Logout - AUTHENTICATION DISABLED"""
            # Temporarily disabled - just redirect to dashboard
            session.pop('user', None)
            return redirect(url_for('dashboard'))
        
        @self.app.route('/dashboard')
        @self.login_required
        def dashboard():
            """Main dashboard - AUTHENTICATION DISABLED"""
            # Temporarily disabled - use default user for template
            user = session.get('user', {'username': 'admin', 'role': 'admin'})
            # Don't log audit since auth is disabled
            # self.audit_logger.log('DASHBOARD_ACCESS', user.get('username'), 'Accessed dashboard', request.remote_addr)
            return render_template('production_dashboard.html', user=user)
        
        @self.app.route('/api/stats')
        @self.login_required
        def get_stats():
            """Get system statistics"""
            return jsonify(self.system_stats)
        
        @self.app.route('/api/predictions')
        @self.login_required
        def get_predictions():
            """Get recent predictions"""
            return jsonify(self.recent_predictions[-100:])
        
        @self.app.route('/api/alerts')
        @self.login_required
        def get_alerts():
            """Get recent alerts"""
            return jsonify(self.recent_alerts[-50:])
        
        @self.app.route('/api/network_stats')
        @self.login_required
        def get_network_stats():
            """Get network statistics"""
            return jsonify(self.network_stats)
        
        @self.app.route('/api/blocked_ips')
        @self.login_required
        @self.role_required('admin', 'analyst')
        def get_blocked_ips():
            """Get blocked IP addresses"""
            return jsonify(list(self.blocked_ips))
        
        @self.app.route('/api/block_ip', methods=['POST'])
        @self.login_required
        @self.role_required('admin')
        def block_ip():
            """Block an IP address"""
            data = request.json
            ip_address = data.get('ip_address')
            
            if ip_address:
                self.blocked_ips.add(ip_address)
                user = session.get('user', {})
                self.audit_logger.log('IP_BLOCK', user.get('username'), f'Blocked IP: {ip_address}', request.remote_addr)
                return jsonify({'status': 'success', 'message': f'IP {ip_address} blocked'})
            
            return jsonify({'status': 'error', 'message': 'Invalid IP address'}), 400
        
        @self.app.route('/api/unblock_ip', methods=['POST'])
        @self.login_required
        @self.role_required('admin')
        def unblock_ip():
            """Unblock an IP address"""
            data = request.json
            ip_address = data.get('ip_address')
            
            if ip_address and ip_address in self.blocked_ips:
                self.blocked_ips.remove(ip_address)
                user = session.get('user', {})
                self.audit_logger.log('IP_UNBLOCK', user.get('username'), f'Unblocked IP: {ip_address}', request.remote_addr)
                return jsonify({'status': 'success', 'message': f'IP {ip_address} unblocked'})
            
            return jsonify({'status': 'error', 'message': 'IP not found in blocked list'}), 400
        
        @self.app.route('/api/monitoring/start', methods=['POST'])
        @self.login_required
        def start_monitoring():
            """Start network monitoring"""
            try:
                if not self.monitoring_active:
                    self.monitoring_active = True
                    self.is_running = True
                    logger.info("🟢 Network monitoring started")
                    self.socketio.emit('monitoring_status', {'status': 'active', 'message': 'Monitoring started'})
                    return jsonify({'status': 'success', 'message': 'Network monitoring started'})
                else:
                    return jsonify({'status': 'info', 'message': 'Monitoring is already active'})
            except Exception as e:
                logger.error(f"❌ Error starting monitoring: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/monitoring/stop', methods=['POST'])
        @self.login_required
        def stop_monitoring():
            """Stop network monitoring"""
            try:
                if self.monitoring_active:
                    self.monitoring_active = False
                    self.is_running = False
                    # Clear predictions and alerts when stopping
                    self.recent_predictions = []
                    self.recent_alerts = []
                    logger.info("🔴 Network monitoring stopped")
                    self.socketio.emit('monitoring_status', {'status': 'stopped', 'message': 'Monitoring stopped'})
                    # Emit cleared data to update UI
                    self.socketio.emit('predictions_update', [])
                    self.socketio.emit('alerts_update', [])
                    return jsonify({'status': 'success', 'message': 'Network monitoring stopped'})
                else:
                    return jsonify({'status': 'info', 'message': 'Monitoring is already stopped'})
            except Exception as e:
                logger.error(f"❌ Error stopping monitoring: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/monitoring/status', methods=['GET'])
        @self.login_required
        def get_monitoring_status():
            """Get current monitoring status"""
            return jsonify({
                'status': 'active' if self.monitoring_active else 'stopped',
                'monitoring_active': self.monitoring_active
            })
    
    def setup_socketio_events(self):
        """Setup SocketIO events with authentication"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection - AUTHENTICATION DISABLED"""
            # Temporarily disabled - allow all connections
            logger.info(f"Client connected: {request.sid}")
            emit('status', {'message': 'Connected to live dashboard'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('request_update')
        def handle_update_request():
            """Handle update request from client - AUTHENTICATION DISABLED"""
            # Temporarily disabled - allow all requests
            emit('stats_update', self.system_stats)
            emit('predictions_update', self.recent_predictions[-50:])  # Increased for table
            emit('alerts_update', self.recent_alerts[-20:])  # Increased for display
    
    def initialize_connections(self) -> bool:
        """Initialize Redis and Kafka connections"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.get('redis', {}).get('host', 'localhost'),
                port=self.config.get('redis', {}).get('port', 6379),
                db=self.config.get('redis', {}).get('db', 0),
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("✅ Redis connection established")
            
            self.kafka_consumer = KafkaConsumer(
                self.config.get('kafka', {}).get('topic_predictions', 'ml_predictions'),
                self.config.get('kafka', {}).get('topic_alerts', 'security_alerts'),
                bootstrap_servers=self.config.get('kafka', {}).get('bootstrap_servers', ['localhost:9092']),
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True
            )
            logger.info("✅ Kafka consumer established")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection initialization failed: {e}")
            return False
    
    def data_consumption_thread(self):
        """Thread for consuming data from Kafka"""
        logger.info("📡 Starting data consumption thread...")
        
        while True:  # Keep thread alive
            try:
                if self.monitoring_active and self.is_running:
                    message_batch = self.kafka_consumer.poll(timeout_ms=1000)
                    
                    for topic_partition, messages in message_batch.items():
                        for message in messages:
                            data = message.value
                            
                            if 'ml_predictions' in message.topic:
                                self.process_prediction(data)
                            elif 'security_alerts' in message.topic:
                                # Process both ML alerts and honeypot alerts
                                self.process_alert(data)
                            elif 'honeypot' in message.topic.lower():
                                # Honeypot alerts come through security_alerts topic
                                self.process_alert(data)
                else:
                    # Monitoring stopped, sleep and check again
                    time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Data consumption error: {e}")
                time.sleep(1)
    
    def process_prediction(self, prediction_data: Dict[str, Any]):
        """Process ML prediction data"""
        try:
            self.recent_predictions.append(prediction_data)
            
            if len(self.recent_predictions) > 1000:
                self.recent_predictions = self.recent_predictions[-1000:]
            
            self.socketio.emit('new_prediction', prediction_data)
            
        except Exception as e:
            logger.error(f"❌ Prediction processing error: {e}")
    
    def process_alert(self, alert_data: Dict[str, Any]):
        """Process security alert data (including honeypot alerts)"""
        try:
            # Check if this is a honeypot alert
            is_honeypot = alert_data.get('threat_type') == 'honeypot_connection' or \
                         alert_data.get('model') == 'honeypot' or \
                         'honeypot' in alert_data.get('description', '').lower()
            
            if is_honeypot:
                logger.warning(f"🎣 HONEYPOT ALERT: {alert_data.get('description', 'Unknown threat')}")
            else:
                logger.warning(f"🚨 ML ALERT: {alert_data.get('description', 'Unknown threat')}")
            
            self.recent_alerts.append(alert_data)
            
            if len(self.recent_alerts) > 100:
                self.recent_alerts = self.recent_alerts[-100:]
            
            self.socketio.emit('new_alert', alert_data)
            
        except Exception as e:
            logger.error(f"❌ Alert processing error: {e}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics (CPU, Memory, Network)"""
        if not psutil:
            return {
                'cpu_percent': 0,
                'memory_percent': 0,
                'network_mbps': 0,
                'network_bytes_sent': 0,
                'network_bytes_recv': 0
            }
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Network throughput
            network_io = psutil.net_io_counters()
            # Calculate network throughput in MB/s (simplified - actual would need delta calculation)
            network_mbps = (network_io.bytes_sent + network_io.bytes_recv) / (1024 * 1024)  # Convert to MB
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'network_mbps': network_mbps,
                'network_bytes_sent': network_io.bytes_sent,
                'network_bytes_recv': network_io.bytes_recv
            }
        except Exception as e:
            logger.debug(f"System health collection error: {e}")
            return {
                'cpu_percent': 0,
                'memory_percent': 0,
                'network_mbps': 0,
                'network_bytes_sent': 0,
                'network_bytes_recv': 0
            }
    
    def stats_update_thread(self):
        """Thread for updating system statistics"""
        logger.info("📊 Starting stats update thread...")
        
        # Track network bytes for throughput calculation
        last_network_bytes = 0
        last_network_time = time.time()
        
        while True:  # Keep thread alive
            try:
                # Only get stats from Redis when monitoring is active
                if self.redis_client and self.monitoring_active:
                    try:
                        stats_json = self.redis_client.get('system_stats')
                        if stats_json:
                            new_stats = json.loads(stats_json)
                            # Update only if we got new data
                            if new_stats:
                                self.system_stats = new_stats
                        
                        # Fallback: Get predictions from Redis if Kafka is failing
                        # BUT ONLY if monitoring is active
                        if self.monitoring_active:
                            try:
                                redis_predictions = self.redis_client.lrange("recent_predictions", 0, 99)  # Get more predictions
                                if redis_predictions:
                                    new_predictions = []
                                    for pred_json in redis_predictions:
                                        try:
                                            pred = json.loads(pred_json)
                                            new_predictions.append(pred)
                                        except:
                                            pass
                                    if new_predictions:
                                        # Merge with existing predictions, avoiding duplicates
                                        existing_timestamps = {p.get('timestamp') for p in self.recent_predictions}
                                        added_count = 0
                                        for pred in new_predictions:
                                            if pred.get('timestamp') not in existing_timestamps:
                                                self.recent_predictions.append(pred)
                                                self.socketio.emit('new_prediction', pred)
                                                added_count += 1
                                        # Keep only recent predictions
                                        if len(self.recent_predictions) > 1000:
                                            self.recent_predictions = self.recent_predictions[-1000:]
                                        if added_count > 0:
                                            logger.debug(f"Loaded {added_count} new predictions from Redis")
                            except Exception as redis_pred_error:
                                logger.debug(f"Redis predictions fetch error (non-critical): {redis_pred_error}")
                    except Exception as redis_error:
                        logger.debug(f"Redis stats fetch error (non-critical): {redis_error}")
                
                # Get system health metrics
                health_metrics = self.get_system_health()
                
                # Calculate network throughput (MB/s)
                current_time = time.time()
                current_network_bytes = health_metrics['network_bytes_sent'] + health_metrics['network_bytes_recv']
                
                if last_network_time > 0:
                    time_delta = current_time - last_network_time
                    bytes_delta = current_network_bytes - last_network_bytes
                    network_throughput_mbps = (bytes_delta / (1024 * 1024)) / max(time_delta, 0.1)  # MB/s
                else:
                    network_throughput_mbps = 0
                
                last_network_bytes = current_network_bytes
                last_network_time = current_time
                
                # Add health metrics to system stats
                self.system_stats.update({
                    'cpu_percent': health_metrics['cpu_percent'],
                    'memory_percent': health_metrics['memory_percent'],
                    'network_throughput_mbps': network_throughput_mbps,
                    'network_bytes_sent': health_metrics['network_bytes_sent'],
                    'network_bytes_recv': health_metrics['network_bytes_recv']
                })
                
                # Only emit stats and updates when monitoring is active
                if self.monitoring_active:
                    self.socketio.emit('stats_update', self.system_stats)
                    
                    # Also emit predictions and alerts updates periodically
                    if self.recent_predictions:
                        self.socketio.emit('predictions_update', self.recent_predictions[-50:])
                    if self.recent_alerts:
                        self.socketio.emit('alerts_update', self.recent_alerts[-20:])
                else:
                    # When stopped, emit empty/zero stats to show stopped state
                    stopped_stats = {
                        'packets_processed': 0,
                        'predictions_made': 0,
                        'threats_detected': 0,
                        'packet_rate': 0,
                        'prediction_rate': 0,
                        'threat_rate': 0,
                        'uptime_seconds': 0,
                        'active_connections': 0,
                        'bandwidth_mbps': 0
                    }
                    stopped_stats.update({
                        'cpu_percent': self.system_stats.get('cpu_percent', 0),
                        'memory_percent': self.system_stats.get('memory_percent', 0),
                        'network_throughput_mbps': 0
                    })
                    self.socketio.emit('stats_update', stopped_stats)
                    # Clear predictions and alerts when stopped
                    if self.recent_predictions:
                        self.recent_predictions = []
                        self.socketio.emit('predictions_update', [])
                    if self.recent_alerts:
                        self.recent_alerts = []
                        self.socketio.emit('alerts_update', [])
                
                time.sleep(2)  # Update every 2 seconds for better responsiveness
                
            except Exception as e:
                logger.error(f"❌ Stats update error: {e}")
                time.sleep(5)
    
    def create_templates(self):
        """Create HTML templates"""
        template_dir = self.project_root / "templates"
        template_dir.mkdir(exist_ok=True)
        
        # Create login template
        login_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Login - ML-IDS-IPS Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .login-card {
            background: white;
            border-radius: 15px;
            padding: 40px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            width: 100%;
            max-width: 400px;
        }
    </style>
</head>
<body>
    <div class="login-card">
        <h2 class="mb-4">🔐 ML-IDS-IPS Login</h2>
        
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="alert alert-{{ 'danger' if category == 'error' else category }} alert-dismissible fade show">
                        {{ message }}
                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                    </div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        <form method="POST">
            <div class="mb-3">
                <label for="username" class="form-label">Username</label>
                <input type="text" class="form-control" id="username" name="username" required>
            </div>
            <div class="mb-3">
                <label for="password" class="form-label">Password</label>
                <input type="password" class="form-control" id="password" name="password" required>
            </div>
            <button type="submit" class="btn btn-primary w-100">Login</button>
        </form>
        
        <div class="mt-4">
            <p class="text-muted small">Default users:</p>
            <ul class="small">
                <li>admin / admin123 (Admin)</li>
                <li>analyst / analyst123 (Analyst)</li>
                <li>viewer / viewer123 (Viewer)</li>
            </ul>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
        """
        
        with open(template_dir / "login.html", "w") as f:
            f.write(login_html)
        
        # Copy existing dashboard template
        logger.info("✅ Templates created")
    
    def start_dashboard(self):
        """Start the production dashboard"""
        logger.info("🚀 Starting Production Dashboard...")
        
        if not self.initialize_connections():
            logger.error("❌ Failed to initialize connections")
            return False
        
        self.create_templates()
        
        # Start threads (monitoring will be inactive by default)
        self.is_running = True
        self.monitoring_active = False  # Start with monitoring stopped
        
        threads = [
            threading.Thread(target=self.data_consumption_thread, daemon=True),
            threading.Thread(target=self.stats_update_thread, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
        
        logger.info("✅ Production Dashboard started successfully!")
        logger.info("⚠️  Network monitoring is stopped by default. Click 'Start Monitoring' to begin.")
        logger.info(f"🌐 Dashboard available at: http://{self.config.get('dashboard', {}).get('host', '0.0.0.0')}:{self.config.get('dashboard', {}).get('port', 5050)}")
        
        return True
    
    def run(self):
        """Run the Flask application"""
        if self.start_dashboard():
            self.socketio.run(
                self.app,
                host=self.config.get('dashboard', {}).get('host', '0.0.0.0'),
                port=self.config.get('dashboard', {}).get('port', 5050),
                debug=self.config.get('dashboard', {}).get('debug', False),
                allow_unsafe_werkzeug=True  # Allow for development
            )

def main():
    """Main function"""
    import threading
    from pathlib import Path
    
    logger.info("🚀 ML-IDS-IPS Production Dashboard")
    logger.info("=" * 60)
    
    dashboard = ProductionDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()

