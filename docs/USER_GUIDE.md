# ML-IDS-IPS USER GUIDE

## Table of Contents

1. [Getting Started](#getting-started)
2. [Accessing the Dashboard](#accessing-the-dashboard)
3. [Understanding the Dashboard](#understanding-the-dashboard)
4. [Working with Alerts](#working-with-alerts)
5. [Viewing Statistics](#viewing-statistics)
6. [Admin Controls](#admin-controls)
7. [Troubleshooting](#troubleshooting)

---

## 1. Getting Started

### System Requirements
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Network connectivity to the ML-IDS-IPS server
- Valid user credentials

### First-Time Access

1. **Open your browser** and navigate to:
   ```
   http://your-server-ip:5050
   ```

2. **Login** with your credentials:
   - Username: provided by administrator
   - Password: provided by administrator

3. **Accept Terms** (first time only)

---

## 2. Accessing the Dashboard

### Login Process

1. Navigate to the login page
2. Enter your username and password
3. Click "Login"
4. You'll be redirected to the main dashboard

### Password Security

- Change your password regularly
- Use strong passwords (8+ characters, mixed case, numbers, symbols)
- Don't share your password with others

### Logout

Click the "Logout" button in the top-right corner when finished.

---

## 3. Understanding the Dashboard

### Dashboard Sections

#### Connection Status
- **Green indicator:** System is running properly
- **Red indicator:** System needs attention
- Shows real-time connectivity status

#### System Statistics
**Key Metrics:**

- **Uptime:** How long the system has been running
- **Packets Processed:** Total network packets analyzed
- **Predictions Made:** Total ML predictions performed
- **Threats Detected:** Number of security threats found
- **Packet Rate:** Current packets per second
- **Threat Rate:** Current threats per second

#### Recent Alerts
Shows the latest security alerts:
- **HIGH:** Critical threats requiring immediate attention
- **MEDIUM:** Suspicious activity to monitor
- **LOW:** Informational alerts

Each alert shows:
- Severity level (color-coded)
- Description
- Confidence percentage
- Timestamp

#### Recent Predictions
Shows the latest ML predictions:
- Model name
- Dataset used
- Prediction result
- Confidence level
- Timestamp

#### Real-time Performance Chart
Visual graph showing packet processing rate over time.

---

## 4. Working with Alerts

### Understanding Alert Severity

#### HIGH Severity (Red)
Critical threats that require immediate action:
- Immediate danger to network security
- Active attack in progress
- System compromise detected

**Action Required:** Review immediately and take appropriate response

#### MEDIUM Severity (Yellow)
Suspicious activity requiring investigation:
- Unusual patterns detected
- Potential attack preparation
- Policy violations

**Action Required:** Monitor and investigate

#### LOW Severity (Green)
Informational alerts:
- Potential false positives
- Non-critical events
- System status updates

**Action Required:** Review periodically

### Alert Details

Click on any alert to view:
- Source IP address
- Destination IP address
- Protocol type
- Detailed description
- Recommended actions

---

## 5. Viewing Statistics

### System Statistics Panel

Shows real-time metrics:
- **Uptime:** System running time
- **Packets:** Total packets analyzed since startup
- **Predictions:** Total ML predictions made
- **Threats:** Total threats detected
- **Packet Rate:** Current processing speed
- **Threat Rate:** Current threat detection rate

### Interpreting Statistics

- **High Packet Rate:** Heavy network traffic
- **High Threat Rate:** Active attacks occurring
- **Low Predictions:** System processing normally
- **Zero Threats:** Network is secure (good!)

---

## 6. Admin Controls

*Note: Admin-only feature*

### Blocking IP Addresses

To block a malicious IP:

1. Navigate to "Admin Controls" section
2. Enter IP address in "Manual IP Block" field
3. Click "Block IP"
4. IP will be blocked for 60 minutes

### Viewing Blocked IPs

List of currently blocked IPs appears in "Blocked IP Addresses" section.

### Unblocking IPs

- Admin can manually unblock IPs if needed
- Contact support for assistance

---

## 7. Troubleshooting

### Dashboard Not Loading

**Problem:** Dashboard won't load

**Solutions:**
1. Check internet connection
2. Verify server is running
3. Clear browser cache
4. Try different browser

### "Connection Lost" Warning

**Problem:** Red connection indicator

**Solutions:**
1. Check network connectivity
2. Refresh the page (F5)
3. Contact system administrator

### No Alerts Displaying

**Problem:** Dashboard showing zero threats

**Possible Reasons:**
1. Network is actually secure (this is good!)
2. System just started (wait for traffic)
3. Models processing normally

**Solution:** Normal operation - no action needed

### Can't Login

**Problem:** Invalid username/password

**Solutions:**
1. Verify credentials with administrator
2. Reset password if forgotten
3. Check for caps lock
4. Check account hasn't been locked out

---

## Quick Reference

### User Roles

| Role | Access Level |
|------|--------------|
| **Viewer** | Read-only dashboard access |
| **Analyst** | View alerts and statistics |
| **Admin** | Full access including IP blocking |

### Key Shortcuts

- **F5:** Refresh dashboard
- **Ctrl+R:** Reload page
- **Esc:** Close modal windows

### Contact Support

For additional help:
- Email: support@ml-ids-ips.com
- Phone: [Support Number]
- Documentation: docs.ml-ids-ips.com

---

**Last Updated:** [Date]  
**Version:** 1.0

