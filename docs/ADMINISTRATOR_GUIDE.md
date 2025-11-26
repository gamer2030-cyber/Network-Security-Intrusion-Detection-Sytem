# ML-IDS-IPS ADMINISTRATOR GUIDE

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [User Management](#user-management)
4. [System Monitoring](#system-monitoring)
5. [Maintenance](#maintenance)
6. [Security Best Practices](#security-best-practices)
7. [Troubleshooting](#troubleshooting)

---

## 1. Installation

### Pre-Installation Checklist

- [ ] Server with 8GB+ RAM
- [ ] 100GB+ free disk space
- [ ] Network interface access
- [ ] Admin/Sudo privileges
- [ ] Python 3.8+ installed
- [ ] Docker and Docker Compose (optional but recommended)

### Installation Steps

#### Option 1: Automated Installation (Recommended)

```bash
# Clone repository
git clone https://github.com/yourorg/ml-ids-ips.git
cd ml-ids-ips

# Run installation script
chmod +x deploy_production.sh
./deploy_production.sh
```

#### Option 2: Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure system
cp config/live_config.yaml config/production_config.yaml
nano config/production_config.yaml
```

### Post-Installation Verification

```bash
# Check services
systemctl status ml-ids-ips

# Test dashboard
curl http://localhost:5050/login

# Check logs
tail -f logs/production.log
```

---

## 2. Configuration

### Main Configuration File

Location: `config/production_config.yaml`

#### Key Settings

**Dashboard Settings:**
```yaml
dashboard:
  host: 0.0.0.0
  port: 5050
  debug: false  # Always false for production
  ssl_enabled: true
```

**Email Alert Settings:**
```yaml
email:
  enabled: true
  smtp_server: smtp.gmail.com
  smtp_port: 587
  from_email: ids@yourcompany.com
  from_password: YOUR_PASSWORD
  recipients:
    - security@yourcompany.com
    - admin@yourcompany.com
```

**ML Model Thresholds:**
```yaml
ml:
  prediction_threshold: 0.95  # Only alert on high confidence
  model_thresholds:
    gradient_boosting_comprehensive: 0.95
    random_forest_comprehensive: 0.85
```

### Network Interface Configuration

```bash
# Detect available interfaces
python detect_network_interface.py

# Edit config
nano config/production_config.yaml

# Set interface
network:
  interface: en0  # Your interface name
```

### SSL Certificate Setup

```bash
# Install certbot
sudo apt-get install certbot

# Obtain certificate
sudo certbot certonly --standalone -d your-domain.com

# Update config with certificate paths
```

---

## 3. User Management

### Creating Users

Edit `production_dashboard.py` to add users:

```python
self.users = {
    'newuser': {
        'password_hash': bcrypt.hashpw('PASSWORD'.encode(), bcrypt.gensalt()),
        'role': 'analyst',
        'last_login': None
    }
}
```

### User Roles

#### Admin Role
- Full system access
- IP blocking/unblocking
- User management
- System configuration

#### Analyst Role
- View all data
- Acknowledge alerts
- Generate reports
- Cannot modify system

#### Viewer Role
- Read-only access
- View dashboard
- View statistics
- Cannot see sensitive data

### Password Policy

Recommended settings:
- Minimum 12 characters
- Mixed case, numbers, symbols
- Force password change every 90 days
- Account lockout after 5 failed attempts

---

## 4. System Monitoring

### Health Checks

```bash
# Check system status
systemctl status ml-ids-ips

# Check resource usage
top -p $(pgrep -f production_dashboard)

# Check disk space
df -h

# Check memory
free -h
```

### Monitoring Metrics

Key metrics to monitor:

1. **CPU Usage:** Should be <80%
2. **Memory Usage:** Should be <80%
3. **Disk Usage:** Should be <90%
4. **Network Traffic:** Monitor packet rate
5. **Detection Rate:** Track threats per hour

### Log Management

```bash
# View production logs
tail -f logs/production.log

# View error logs
tail -f logs/error.log

# View audit logs
tail -f logs/audit.log

# Archive old logs
logrotate -f /etc/logrotate.d/ml-ids-ips
```

---

## 5. Maintenance

### Regular Maintenance Tasks

#### Daily
- [ ] Check dashboard is accessible
- [ ] Review alerts for false positives
- [ ] Verify email alerts working
- [ ] Check disk space

#### Weekly
- [ ] Review system performance metrics
- [ ] Update threat intelligence feeds
- [ ] Backup configuration files
- [ ] Review blocked IPs list

#### Monthly
- [ ] Full system backup
- [ ] Update ML models if available
- [ ] Review and optimize thresholds
- [ ] Security audit
- [ ] Performance optimization

### Backup Procedure

```bash
# Backup all important data
tar -czf backup_$(date +%Y%m%d).tar.gz \
    config/ \
    models/ \
    logs/

# Store backup off-site
scp backup_*.tar.gz backup-server:/backups/
```

### Updates

```bash
# Pull latest code
git pull origin main

# Restart services
sudo systemctl restart ml-ids-ips

# Verify update
tail -f logs/production.log
```

---

## 6. Security Best Practices

### System Security

1. **Change Default Passwords**
   - Never use default credentials
   - Implement strong password policy

2. **Enable HTTPS**
   - Always use SSL in production
   - Keep certificates updated

3. **Firewall Configuration**
   - Allow only necessary ports (5050 for dashboard)
   - Restrict admin access to specific IPs

4. **Regular Updates**
   - Apply security patches promptly
   - Keep ML models updated

5. **Access Control**
   - Use VPN for remote access
   - Implement 2FA if possible
   - Log all admin actions

### Data Privacy

- Network data never leaves your infrastructure
- All processing occurs locally
- Audit logs enable compliance reporting
- Regular security reviews recommended

---

## 7. Troubleshooting

### Common Issues

#### System Won't Start

**Problem:** Service fails to start

**Solution:**
```bash
# Check logs
journalctl -u ml-ids-ips -n 50

# Verify configuration
python production_dashboard.py --validate-config

# Check port availability
lsof -i :5050
```

#### No Packets Being Processed

**Problem:** Dashboard shows 0 packets

**Causes:**
- Network interface not accessible
- Insufficient permissions
- Interface in incorrect mode

**Solution:**
```bash
# Test network access
sudo python -c "from scapy.all import sniff; sniff(count=1)"

# Check permissions
ls -la /dev/bpf*

# Verify interface
ifconfig
```

#### High False Positive Rate

**Problem:** Too many alerts being generated

**Solutions:**
1. Adjust ML thresholds in config
2. Add IPs to whitelist
3. Tune model parameters
4. Contact support for assistance

#### Email Alerts Not Sending

**Problem:** No email notifications

**Solutions:**
1. Test SMTP connection
2. Verify email credentials
3. Check firewall rules
4. Review email logs

---

## Appendix: Configuration Reference

### Complete Configuration Example

```yaml
# Full configuration file
dashboard:
  host: 0.0.0.0
  port: 5050
  debug: false
  ssl_enabled: true
  ssl_cert: /etc/ssl/certs/cert.pem
  ssl_key: /etc/ssl/private/key.pem

kafka:
  bootstrap_servers: ['localhost:9092']

redis:
  host: localhost
  port: 6379
  db: 0

network:
  interface: en0
  packet_count: 10000

ml:
  model_path: models
  prediction_threshold: 0.95
  batch_size: 1000

email:
  enabled: true
  smtp_server: smtp.gmail.com
  smtp_port: 587
  from_email: ids@company.com
  recipients:
    - security@company.com

auto_block:
  enabled: true
  block_duration_minutes: 60
  auto_block_threshold: 3

logging:
  level: INFO
  file: logs/production.log
  max_size_mb: 100
```

---

**Support Contact:**
- Email: support@ml-ids-ips.com
- Phone: [Support Number]
- Documentation: docs.ml-ids-ips.com

**Last Updated:** [Date]  
**Version:** 1.0

