#!/usr/bin/env python3
"""
url_threat_detector.py - URL and Domain Threat Detection for Browser Traffic

This module enhances the IDS/IPS system to detect malicious websites and URLs
by analyzing HTTP/HTTPS traffic and DNS queries from browser activity.
"""

import os
import sys
import re
import json
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
from pathlib import Path
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

# Network monitoring
try:
    from scapy.all import sniff, IP, TCP, UDP, Raw, DNS, DNSQR
    from scapy.layers.http import HTTPRequest, HTTPResponse
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scapy"])
    from scapy.all import sniff, IP, TCP, UDP, Raw, DNS, DNSQR

# Real-time streaming
try:
    import redis
    from kafka import KafkaProducer
except ImportError:
    print("Installing streaming packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "redis", "kafka-python"])
    import redis
    from kafka import KafkaProducer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('url_threat_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class URLThreatDetector:
    """Detect malicious URLs and domains from browser traffic"""
    
    def __init__(self, config_path: str = "config/live_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        
        # Threat intelligence sources
        self.malicious_domains: Set[str] = set()
        self.malicious_urls: Set[str] = set()
        self.suspicious_keywords: List[str] = []
        
        # Tracked domains and URLs
        self.detected_domains: Dict[str, Dict] = {}
        self.detected_urls: Dict[str, Dict] = {}
        
        # Infrastructure
        self.redis_client = None
        self.kafka_producer = None
        
        # State
        self.is_running = False
        self.stats = {
            'domains_checked': 0,
            'urls_checked': 0,
            'threats_detected': 0,
            'dns_queries_analyzed': 0,
            'http_requests_analyzed': 0
        }
        
        # Initialize threat lists
        self.load_threat_intelligence()
        self.initialize_infrastructure()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        try:
            import yaml
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
        return {}
    
    def load_threat_intelligence(self):
        """Load threat intelligence data (malicious domains, URLs, etc.)"""
        logger.info("Loading threat intelligence data...")
        
        # Load from file if exists
        threat_file = Path("config/threat_intelligence.json")
        if threat_file.exists():
            try:
                with open(threat_file, 'r') as f:
                    data = json.load(f)
                    self.malicious_domains = set(data.get('malicious_domains', []))
                    self.malicious_urls = set(data.get('malicious_urls', []))
                    self.suspicious_keywords = data.get('suspicious_keywords', [])
            except Exception as e:
                logger.warning(f"Could not load threat intelligence file: {e}")
        
        # Add some example malicious domains for demonstration
        # In production, load from threat intelligence feeds
        example_threats = [
            'malware.com',
            'phishing-site.net',
            'suspicious-domain.org',
            'malicious-website.info'
        ]
        self.malicious_domains.update(example_threats)
        
        # Suspicious URL patterns
        self.suspicious_keywords.extend([
            'malware', 'phishing', 'trojan', 'virus', 'exploit',
            'hack', 'crack', 'keygen', 'warez', 'piracy'
        ])
        
        logger.info(f"Loaded {len(self.malicious_domains)} malicious domains")
        logger.info(f"Loaded {len(self.suspicious_keywords)} suspicious keywords")
    
    def initialize_infrastructure(self):
        """Initialize Redis and Kafka connections"""
        try:
            # Redis
            redis_config = self.config.get('redis', {})
            self.redis_client = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("✓ Redis connected")
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self.redis_client = None
        
        try:
            # Kafka
            kafka_config = self.config.get('kafka', {})
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=kafka_config.get('bootstrap_servers', 'localhost:9092'),
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            logger.info("✓ Kafka producer initialized")
        except Exception as e:
            logger.warning(f"Kafka not available: {e}")
            self.kafka_producer = None
    
    def extract_domain_from_dns(self, packet) -> Optional[str]:
        """Extract domain name from DNS query"""
        try:
            if packet.haslayer(DNS):
                dns_layer = packet[DNS]
                if dns_layer.qr == 0:  # DNS query (not response)
                    if dns_layer.haslayer(DNSQR):
                        query = dns_layer[DNSQR]
                        domain = query.qname.decode('utf-8').rstrip('.')
                        return domain.lower()
        except Exception as e:
            logger.debug(f"Error extracting DNS domain: {e}")
        return None
    
    def extract_url_from_http(self, packet) -> Optional[Dict[str, str]]:
        """Extract URL and domain from HTTP request"""
        try:
            if packet.haslayer(Raw):
                raw_data = packet[Raw].load
                
                # Try to parse as HTTP request
                try:
                    http_data = raw_data.decode('utf-8', errors='ignore')
                    
                    # Extract HTTP method and path
                    if http_data.startswith(('GET ', 'POST ', 'PUT ', 'DELETE ', 'HEAD ')):
                        lines = http_data.split('\r\n')
                        if lines:
                            request_line = lines[0]
                            parts = request_line.split(' ')
                            if len(parts) >= 2:
                                method = parts[0]
                                path = parts[1]
                                
                                # Extract Host header
                                host = None
                                for line in lines[1:]:
                                    if line.lower().startswith('host:'):
                                        host = line.split(':', 1)[1].strip()
                                        break
                                
                                if host:
                                    # Construct full URL
                                    url = f"http://{host}{path}"
                                    return {
                                        'url': url,
                                        'domain': host.split(':')[0],  # Remove port if present
                                        'path': path,
                                        'method': method
                                    }
                except:
                    pass
        except Exception as e:
            logger.debug(f"Error extracting HTTP URL: {e}")
        return None
    
    def check_domain_threat(self, domain: str) -> Dict[str, Any]:
        """Check if domain is malicious"""
        self.stats['domains_checked'] += 1
        
        threat_info = {
            'domain': domain,
            'is_threat': False,
            'threat_type': None,
            'confidence': 0.0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Check against malicious domain list
        if domain in self.malicious_domains:
            threat_info['is_threat'] = True
            threat_info['threat_type'] = 'malicious_domain'
            threat_info['confidence'] = 0.95
            return threat_info
        
        # Check for suspicious keywords in domain
        domain_lower = domain.lower()
        for keyword in self.suspicious_keywords:
            if keyword in domain_lower:
                threat_info['is_threat'] = True
                threat_info['threat_type'] = 'suspicious_keyword'
                threat_info['confidence'] = 0.70
                break
        
        # Check for suspicious TLDs
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.top', '.xyz']
        if any(domain.endswith(tld) for tld in suspicious_tlds):
            threat_info['is_threat'] = True
            threat_info['threat_type'] = 'suspicious_tld'
            threat_info['confidence'] = 0.60
        
        # Check domain length (very long domains can be suspicious)
        if len(domain) > 50:
            threat_info['is_threat'] = True
            threat_info['threat_type'] = 'suspicious_length'
            threat_info['confidence'] = 0.50
        
        return threat_info
    
    def check_url_threat(self, url: str, domain: str) -> Dict[str, Any]:
        """Check if URL is malicious"""
        self.stats['urls_checked'] += 1
        
        threat_info = {
            'url': url,
            'domain': domain,
            'is_threat': False,
            'threat_type': None,
            'confidence': 0.0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Check against malicious URL list
        if url in self.malicious_urls:
            threat_info['is_threat'] = True
            threat_info['threat_type'] = 'malicious_url'
            threat_info['confidence'] = 0.95
            return threat_info
        
        # Check domain first
        domain_check = self.check_domain_threat(domain)
        if domain_check['is_threat']:
            threat_info['is_threat'] = True
            threat_info['threat_type'] = domain_check['threat_type']
            threat_info['confidence'] = domain_check['confidence']
            return threat_info
        
        # Check URL for suspicious patterns
        url_lower = url.lower()
        for keyword in self.suspicious_keywords:
            if keyword in url_lower:
                threat_info['is_threat'] = True
                threat_info['threat_type'] = 'suspicious_url_pattern'
                threat_info['confidence'] = 0.75
                break
        
        return threat_info
    
    def process_packet(self, packet):
        """Process network packet to detect URL/domain threats"""
        try:
            # Extract domain from DNS query
            domain = self.extract_domain_from_dns(packet)
            if domain:
                self.stats['dns_queries_analyzed'] += 1
                threat_info = self.check_domain_threat(domain)
                
                if threat_info['is_threat']:
                    self.handle_threat_detection(threat_info, 'dns')
            
            # Extract URL from HTTP request
            if packet.haslayer(TCP) and packet.haslayer(Raw):
                tcp_layer = packet[TCP]
                if tcp_layer.dport in [80, 8080, 8000] or tcp_layer.sport in [80, 8080, 8000]:
                    url_info = self.extract_url_from_http(packet)
                    if url_info:
                        self.stats['http_requests_analyzed'] += 1
                        threat_info = self.check_url_threat(
                            url_info['url'],
                            url_info['domain']
                        )
                        
                        if threat_info['is_threat']:
                            self.handle_threat_detection(threat_info, 'http', url_info)
        
        except Exception as e:
            logger.debug(f"Error processing packet: {e}")
    
    def handle_threat_detection(self, threat_info: Dict, source: str, url_info: Optional[Dict] = None):
        """Handle detected threat"""
        self.stats['threats_detected'] += 1
        
        # Create alert
        alert = {
            'timestamp': datetime.now().isoformat(),
            'alert_type': 'url_threat',
            'severity': 'HIGH' if threat_info['confidence'] > 0.8 else 'MEDIUM',
            'source': source,
            'threat_info': threat_info,
            'description': f"Threat website detected: {threat_info.get('domain') or threat_info.get('url')}",
            'recommendation': 'Block access to this website immediately'
        }
        
        if url_info:
            alert['url_info'] = url_info
        
        logger.warning(f"🚨 THREAT DETECTED: {threat_info.get('domain') or threat_info.get('url')} "
                      f"({threat_info['threat_type']}, confidence: {threat_info['confidence']:.2f})")
        
        # Store in Redis
        if self.redis_client:
            try:
                alert_key = f"threat:url:{datetime.now().timestamp()}"
                self.redis_client.setex(alert_key, 3600, json.dumps(alert))
                self.redis_client.lpush('threats:url', alert_key)
                self.redis_client.ltrim('threats:url', 0, 999)  # Keep last 1000
            except Exception as e:
                logger.error(f"Error storing in Redis: {e}")
        
        # Send to Kafka (use same topic as main system)
        if self.kafka_producer:
            try:
                # Send to security_alerts topic (same as main IDS/IPS system)
                self.kafka_producer.send('security_alerts', alert)
                # Also send to threat-alerts topic for compatibility
                self.kafka_producer.send('threat-alerts', alert)
            except Exception as e:
                logger.error(f"Error sending to Kafka: {e}")
        
        # Store locally
        threat_id = f"{source}_{datetime.now().timestamp()}"
        if source == 'dns':
            self.detected_domains[threat_id] = threat_info
        else:
            self.detected_urls[threat_id] = threat_info
    
    def start_monitoring(self, interface: Optional[str] = None):
        """Start monitoring network traffic for URL threats"""
        if self.is_running:
            logger.warning("Monitoring already running")
            return
        
        self.is_running = True
        logger.info("Starting URL threat detection monitoring...")
        
        if not interface:
            interface = self.config.get('network', {}).get('interface', 'en0')
        
        logger.info(f"Monitoring interface: {interface}")
        logger.info("Monitoring browser traffic for malicious websites...")
        
        try:
            # Start packet capture
            sniff(
                iface=interface,
                prn=self.process_packet,
                store=False,
                stop_filter=lambda x: not self.is_running
            )
        except KeyboardInterrupt:
            logger.info("Stopping monitoring...")
            self.stop_monitoring()
        except Exception as e:
            logger.error(f"Error in packet capture: {e}")
            self.is_running = False
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_running = False
        logger.info("URL threat detection stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics"""
        return {
            **self.stats,
            'domains_tracked': len(self.detected_domains),
            'urls_tracked': len(self.detected_urls),
            'malicious_domains_loaded': len(self.malicious_domains)
        }


def main():
    """Main function to run URL threat detector"""
    import argparse
    
    parser = argparse.ArgumentParser(description='URL and Domain Threat Detector')
    parser.add_argument('--interface', '-i', type=str, help='Network interface to monitor')
    parser.add_argument('--config', '-c', type=str, default='config/live_config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    detector = URLThreatDetector(config_path=args.config)
    
    print("\n" + "="*60)
    print("  URL & Domain Threat Detector")
    print("="*60)
    print(f"Monitoring browser traffic for malicious websites...")
    print(f"Loaded {len(detector.malicious_domains)} malicious domains")
    print(f"Loaded {len(detector.suspicious_keywords)} suspicious keywords")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        detector.start_monitoring(interface=args.interface)
    except KeyboardInterrupt:
        print("\n\nStopping...")
        detector.stop_monitoring()
        
        # Print final stats
        stats = detector.get_stats()
        print("\n" + "="*60)
        print("  Detection Statistics")
        print("="*60)
        print(f"Domains checked: {stats['domains_checked']}")
        print(f"URLs checked: {stats['urls_checked']}")
        print(f"Threats detected: {stats['threats_detected']}")
        print(f"DNS queries analyzed: {stats['dns_queries_analyzed']}")
        print(f"HTTP requests analyzed: {stats['http_requests_analyzed']}")


if __name__ == "__main__":
    main()

