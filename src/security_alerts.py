#!/usr/bin/env python3
"""
security_alerts.py - Email alerts and auto-blocking system

This module provides:
- Email notification system
- SMS alerts (via email gateway)
- Automatic IP blocking via iptables
- Threat response automation
"""

import smtplib
import logging
import subprocess
import threading
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Optional, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class EmailAlertSystem:
    """Email alert system for security notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.smtp_server = config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = config.get('smtp_port', 587)
        self.from_email = config.get('from_email', 'ids@company.com')
        self.from_password = config.get('from_password', '')
        self.recipients = config.get('recipients', [])
        self.enabled = config.get('enabled', True)
    
    def send_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Send security alert via email"""
        if not self.enabled:
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = f"🚨 Security Alert: {alert_data.get('severity', 'HIGH')} - {alert_data.get('description', 'Unknown Threat')}"
            
            # Create email body
            body = f"""
            ⚠️ SECURITY ALERT ⚠️
            
            Alert Type: {alert_data.get('alert_type', 'Unknown')}
            Severity: {alert_data.get('severity', 'HIGH')}
            Description: {alert_data.get('description', 'No description')}
            Confidence: {alert_data.get('confidence', 0) * 100:.1f}%
            
            Source IP: {alert_data.get('source_ip', 'Unknown')}
            Destination IP: {alert_data.get('dest_ip', 'Unknown')}
            Protocol: {alert_data.get('protocol', 'Unknown')}
            
            Timestamp: {alert_data.get('timestamp', datetime.now().isoformat())}
            
            Recommendation: {'BLOCK IMMEDIATELY' if alert_data.get('severity') == 'HIGH' else 'Monitor closely'}
            
            ---
            ML-IDS-IPS System
            Automated Security Alert
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.from_email, self.from_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"✅ Email alert sent to {self.recipients}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send email alert: {e}")
            return False
    
    def send_summary(self, stats: Dict[str, Any]) -> bool:
        """Send daily summary email"""
        if not self.enabled:
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = f"📊 Daily Security Summary - {datetime.now().strftime('%Y-%m-%d')}"
            
            body = f"""
            Daily Security Summary
            
            Date: {datetime.now().strftime('%Y-%m-%d')}
            
            Statistics:
            - Packets Processed: {stats.get('packets_processed', 0)}
            - Predictions Made: {stats.get('predictions_made', 0)}
            - Threats Detected: {stats.get('threats_detected', 0)}
            - False Positives: {stats.get('false_positives', 0)}
            - Blocked IPs: {stats.get('blocked_ips', 0)}
            
            Top Threat Types:
            """
            
            # Add top threat types
            threat_types = stats.get('threat_types', {})
            for threat_type, count in sorted(threat_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                body += f"\n- {threat_type}: {count}"
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.from_email, self.from_password)
            server.send_message(msg)
            server.quit()
            
            logger.info("✅ Summary email sent")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send summary email: {e}")
            return False


class AutoBlockSystem:
    """Automatic IP blocking system using iptables"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enabled', True)
        self.block_duration = config.get('block_duration_minutes', 60)
        self.blocked_ips = set()
        self.blocked_ip_timestamps = {}
    
    def block_ip(self, ip_address: str, reason: str = "Threat detected") -> bool:
        """Block an IP address using iptables"""
        if not self.enabled:
            return False
        
        if ip_address in self.blocked_ips:
            logger.info(f"IP {ip_address} already blocked")
            return True
        
        try:
            # Check if running as root
            import os
            if os.geteuid() != 0:
                logger.warning("⚠️ Root privileges required for IP blocking. Running command without -j DROP")
                # Log to a file instead
                with open('blocked_ips.log', 'a') as f:
                    f.write(f"{datetime.now().isoformat()} | Blocked: {ip_address} | Reason: {reason}\n")
                self.blocked_ips.add(ip_address)
                return True
            
            # Add blocking rules
            subprocess.run(['iptables', '-A', 'INPUT', '-s', ip_address, '-j', 'DROP'], check=True)
            subprocess.run(['iptables', '-A', 'OUTPUT', '-d', ip_address, '-j', 'DROP'], check=True)
            
            self.blocked_ips.add(ip_address)
            self.blocked_ip_timestamps[ip_address] = datetime.now()
            
            logger.warning(f"🚫 IP {ip_address} blocked ({reason})")
            
            # Schedule unblock after duration
            threading.Thread(target=self._unblock_after_duration, args=(ip_address,), daemon=True).start()
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to block IP {ip_address}: {e}")
            return False
        except PermissionError:
            logger.warning("⚠️ Permission denied. Add user to sudoers or run with sudo.")
            # Log to file as fallback
            with open('blocked_ips.log', 'a') as f:
                f.write(f"{datetime.now().isoformat()} | Blocked: {ip_address} | Reason: {reason}\n")
            self.blocked_ips.add(ip_address)
            return True
    
    def unblock_ip(self, ip_address: str) -> bool:
        """Unblock an IP address"""
        if ip_address not in self.blocked_ips:
            return False
        
        try:
            import os
            if os.geteuid() != 0:
                logger.warning("⚠️ Root privileges required for IP unblocking")
                self.blocked_ips.discard(ip_address)
                if ip_address in self.blocked_ip_timestamps:
                    del self.blocked_ip_timestamps[ip_address]
                return True
            
            # Remove blocking rules
            try:
                subprocess.run(['iptables', '-D', 'INPUT', '-s', ip_address, '-j', 'DROP'], check=True)
            except:
                pass
            
            try:
                subprocess.run(['iptables', '-D', 'OUTPUT', '-d', ip_address, '-j', 'DROP'], check=True)
            except:
                pass
            
            self.blocked_ips.discard(ip_address)
            if ip_address in self.blocked_ip_timestamps:
                del self.blocked_ip_timestamps[ip_address]
            
            logger.info(f"✅ IP {ip_address} unblocked")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to unblock IP {ip_address}: {e}")
            return False
    
    def _unblock_after_duration(self, ip_address: str):
        """Unblock IP after specified duration"""
        time.sleep(self.block_duration * 60)
        self.unblock_ip(ip_address)
        logger.info(f"⏰ IP {ip_address} automatically unblocked after {self.block_duration} minutes")
    
    def is_blocked(self, ip_address: str) -> bool:
        """Check if IP is blocked"""
        return ip_address in self.blocked_ips
    
    def get_blocked_ips(self) -> List[str]:
        """Get list of blocked IPs"""
        return list(self.blocked_ips)


class ThreatResponseSystem:
    """Automated threat response system"""
    
    def __init__(self, email_system: EmailAlertSystem, block_system: AutoBlockSystem):
        self.email_system = email_system
        self.block_system = block_system
        self.threat_count = {}
        self.auto_block_threshold = 3  # Auto-block after 3 high-severity alerts
    
    def handle_threat(self, alert_data: Dict[str, Any]):
        """Handle detected threat with automated response"""
        severity = alert_data.get('severity', 'LOW')
        source_ip = alert_data.get('source_ip', '')
        
        # Send email alert
        if self.email_system:
            self.email_system.send_alert(alert_data)
        
        # Auto-block high severity threats
        if severity == 'HIGH' and source_ip:
            if source_ip not in self.threat_count:
                self.threat_count[source_ip] = 0
            
            self.threat_count[source_ip] += 1
            
            if self.threat_count[source_ip] >= self.auto_block_threshold:
                logger.warning(f"🚫 Auto-blocking IP {source_ip} after {self.threat_count[source_ip]} high-severity alerts")
                self.block_system.block_ip(source_ip, f"Multiple high-severity alerts: {self.threat_count[source_ip]}")
                self.threat_count[source_ip] = 0  # Reset counter
    
    def reset_threat_count(self, ip_address: str):
        """Reset threat count for IP"""
        if ip_address in self.threat_count:
            del self.threat_count[ip_address]


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'email': {
            'enabled': True,
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'from_email': 'your_email@gmail.com',
            'from_password': 'your_app_password',
            'recipients': ['security@company.com']
        },
        'auto_block': {
            'enabled': True,
            'block_duration_minutes': 60
        }
    }
    
    # Initialize systems
    email_system = EmailAlertSystem(config.get('email', {}))
    block_system = AutoBlockSystem(config.get('auto_block', {}))
    response_system = ThreatResponseSystem(email_system, block_system)
    
    # Test alert
    test_alert = {
        'alert_type': 'Malware Detection',
        'severity': 'HIGH',
        'description': 'Malicious network activity detected',
        'confidence': 0.95,
        'source_ip': '192.168.1.100',
        'dest_ip': '10.0.0.1',
        'protocol': 'TCP',
        'timestamp': datetime.now().isoformat()
    }
    
    # Handle threat
    response_system.handle_threat(test_alert)
    
    print("✅ Threat response system test completed")

