# How The ML-IDS-IPS Project Works - Complete Explanation

## 🎯 Overview

This project is a **Machine Learning-based Intrusion Detection and Prevention System (IDS/IPS)** that monitors network traffic in real-time, detects threats using AI models, and displays alerts on a web dashboard.

---

## 📊 Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    NETWORK TRAFFIC                              │
│         (Your Computer's Network Interface)                     │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│         STEP 1: PACKET CAPTURE                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ live_data_streaming_system.py                            │  │
│  │ - Uses Scapy to capture ALL network packets             │  │
│  │ - Captures: TCP, UDP, ICMP, HTTP, DNS, etc.              │  │
│  │ - Extracts 40+ features from each packet                 │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│         STEP 2: FEATURE EXTRACTION                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ From Each Packet, Extracts:                               │  │
│  │ • Source/Destination IP addresses                         │  │
│  │ • Source/Destination Ports                                │  │
│  │ • Protocol (TCP/UDP/ICMP)                                  │  │
│  │ • Packet size, flags, TTL                                  │  │
│  │ • Service type (HTTP, HTTPS, SSH, DNS, etc.)              │  │
│  │ • Flow statistics (connection patterns)                  │  │
│  │ • Session features (duration, data transfer)              │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│         STEP 3: DATA STREAMING (Kafka)                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ - Packets sent to Kafka topic: "network_packets"         │  │
│  │ - Kafka acts as message queue                            │  │
│  │ - Allows multiple systems to process data                 │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│         STEP 4: ML MODEL PREDICTION                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ - Reads packets from Kafka                               │  │
│  │ - Prepares features for ML models                        │  │
│  │ - Runs 4 trained ML models:                              │  │
│  │   1. Random Forest                                        │  │
│  │   2. Gradient Boosting                                    │  │
│  │   3. SVM (Support Vector Machine)                         │  │
│  │   4. Bagging Classifier                                   │  │
│  │ - Each model predicts: Normal or Attack                 │  │
│  │ - Confidence score: 0-100%                               │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│         STEP 5: THREAT DETECTION                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ If ML model predicts "Attack" with high confidence:     │  │
│  │ - Creates security alert                                 │  │
│  │ - Sends alert to Kafka topic: "security_alerts"          │  │
│  │ - Stores in Redis for quick access                        │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│         STEP 6: DASHBOARD DISPLAY                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ production_dashboard.py                                  │  │
│  │ - Reads alerts from Kafka                                │  │
│  │ - Displays in real-time via WebSocket                    │  │
│  │ - Shows: Threats, Predictions, Statistics                │  │
│  │ - Access at: http://localhost:5050                        │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔍 Detailed Step-by-Step Process

### **STEP 1: Packet Capture** 
**File: `live_data_streaming_system.py`**

#### What Happens:
1. **Starts packet capture** on your network interface (usually `en0` on macOS)
2. **Uses Scapy library** to sniff all network packets
3. **Captures every packet** that goes through your network:
   - HTTP requests (web browsing)
   - HTTPS connections (secure websites)
   - DNS queries (domain name lookups)
   - SSH connections
   - Email traffic
   - Any other network activity

#### Example:
```
You browse to: google.com

System captures:
- DNS query packet: "What is google.com's IP?"
- HTTP request packet: "GET / HTTP/1.1"
- Response packets: Website content
```

---

### **STEP 2: Feature Extraction**
**File: `live_data_streaming_system.py` → `extract_packet_features()`**

#### What Happens:
For each captured packet, the system extracts **40+ features**:

**IP Layer Features:**
- Source IP: `192.168.1.100`
- Destination IP: `8.8.8.8`
- Protocol: TCP/UDP/ICMP
- Packet size: `1500 bytes`
- TTL (Time To Live): `64`

**TCP Layer Features:**
- Source port: `54321`
- Destination port: `80` (HTTP)
- TCP flags: SYN, ACK, FIN
- Window size: `65535`
- Service type: `http`

**Flow Features:**
- Connection duration
- Number of packets in connection
- Data bytes transferred
- Error rates

**Session Features:**
- Same service connections
- Different service connections
- Connection patterns

#### Example Feature Set:
```python
{
    'src_ip': '192.168.1.100',
    'dst_ip': '8.8.8.8',
    'src_port': 54321,
    'dst_port': 80,
    'protocol': 'TCP',
    'service': 'http',
    'packet_size': 1500,
    'flags': 'SF',
    'count': 5,  # 5th packet in this connection
    'duration': 0.5  # 0.5 seconds
}
```

---

### **STEP 3: Data Streaming (Kafka)**
**File: `live_data_streaming_system.py` → `send_packet_batch()`**

#### What Happens:
1. **Packets are grouped** into batches (100 packets per batch)
2. **Sent to Kafka** topic: `network_packets`
3. **Kafka stores** the data temporarily
4. **Other systems** can read from Kafka

#### Why Kafka?
- **Scalability**: Can handle millions of packets
- **Reliability**: Data is stored, won't be lost
- **Decoupling**: Packet capture and ML processing are separate

---

### **STEP 4: ML Model Prediction**
**File: `live_data_streaming_system.py` → `process_packet_batch()`**

#### What Happens:

1. **Reads packets** from Kafka topic `network_packets`

2. **Prepares features** for ML models:
   - Converts packet data to format models expect
   - Handles missing values
   - Normalizes data

3. **Runs 4 ML models** (trained on attack datasets):
   - **Random Forest**: Tree-based ensemble
   - **Gradient Boosting**: Advanced tree model
   - **SVM**: Support Vector Machine
   - **Bagging**: Bootstrap aggregating

4. **Each model predicts**:
   - `0` = Normal traffic
   - `1` = Attack/Threat
   - Confidence: `0.0` to `1.0` (0% to 100%)

5. **Example Prediction**:
   ```python
   {
       'model': 'gradient_boosting',
       'prediction': 1,  # Attack detected
       'confidence': 0.95,  # 95% confident
       'is_threat': True,
       'src_ip': '192.168.1.100',
       'dst_ip': '8.8.8.8'
   }
   ```

6. **Sends prediction** to Kafka topic: `ml_predictions`

---

### **STEP 5: Threat Detection & Alert Generation**

#### A. ML-Based Threat Detection
**File: `live_data_streaming_system.py` → `create_alert()`**

**What Happens:**
- If ML model predicts **"Attack"** with **confidence ≥ 70%**:
  - Creates security alert
  - Determines severity:
    - **HIGH**: Confidence ≥ 80%
    - **MEDIUM**: Confidence 50-79%
    - **LOW**: Confidence < 50%

**Example Alert:**
```python
{
    'alert_type': 'ml_threat',
    'severity': 'HIGH',
    'description': 'DoS attack detected',
    'confidence': 0.95,
    'src_ip': '192.168.1.100',
    'dst_ip': '8.8.8.8',
    'model': 'gradient_boosting',
    'timestamp': '2024-01-15 10:30:45'
}
```

#### B. URL/Domain Threat Detection
**File: `url_threat_detector.py`**

**What Happens:**
1. **Monitors DNS queries** and HTTP requests
2. **Extracts domain names** from traffic
3. **Checks against threat intelligence**:
   - Known malicious domains
   - Suspicious keywords
   - Suspicious TLDs (.tk, .ml, etc.)
   - Very long domains (>50 chars)

**Example:**
```
You visit: test-malware-demo.com

System detects:
- Domain in malicious list → Threat!
- Creates alert:
  {
    'alert_type': 'url_threat',
    'threat_type': 'malicious_domain',
    'domain': 'test-malware-demo.com',
    'confidence': 0.95
  }
```

#### C. Honeypot Detection
**File: `honeypot_system.py`**

**What Happens:**
- Deploys fake services (SSH, HTTP, FTP, etc.)
- If someone connects → **100% confirmed attacker**
- Creates immediate alert

---

### **STEP 6: Alert Distribution**

**Alerts are sent to:**
1. **Kafka topic**: `security_alerts`
2. **Redis**: For fast access
3. **Dashboard**: Via WebSocket (real-time)

---

### **STEP 7: Dashboard Display**
**File: `production_dashboard.py`**

#### What Happens:

1. **Dashboard starts** on `http://localhost:5050`

2. **Reads alerts** from Kafka topic `security_alerts`

3. **Displays in real-time** using WebSocket:
   - **Security Alerts** section: Shows all threats
   - **Predictions Table**: Shows ML predictions
   - **Statistics**: Threat counts, confidence levels
   - **Charts**: Visual representation

4. **Real-time Updates**:
   - New alerts appear instantly
   - No page refresh needed
   - Uses Socket.IO for live updates

#### Dashboard Sections:

**1. Security Alerts:**
- Shows all detected threats
- Color-coded by severity (HIGH/MEDIUM/LOW)
- Shows threat type, confidence, IPs
- Badges for special alerts (🌐 Malicious Website, 🎣 Honeypot)

**2. Predictions Table:**
- All ML predictions
- Filter by model, confidence, threat type
- Shows source/destination IPs

**3. Statistics:**
- Total packets processed
- Threats detected
- Threat rate
- System health

**4. Charts:**
- Threat distribution
- Confidence levels
- Attack types

---

## 🔄 Complete Flow Example

### **Scenario: You Browse a Malicious Website**

```
1. You type: test-malware-demo.com in browser
   │
   ▼
2. DNS Query Packet Captured
   - Packet: UDP, port 53
   - Domain: test-malware-demo.com
   │
   ▼
3. URL Threat Detector Processes
   - Extracts domain: test-malware-demo.com
   - Checks threat list: ✅ Found!
   - Threat type: malicious_domain
   │
   ▼
4. Alert Created
   - Severity: HIGH
   - Confidence: 95%
   - Sent to Kafka: security_alerts
   │
   ▼
5. Dashboard Receives Alert
   - Reads from Kafka
   - Emits via WebSocket
   │
   ▼
6. Alert Appears in Dashboard
   - Shows in "Security Alerts" section
   - Red badge: "🌐 MALICIOUS WEBSITE"
   - Displays domain, threat type, confidence
   - Time: 2-5 seconds from browsing to alert!
```

---

## 🎯 Key Components Explained

### **1. live_data_streaming_system.py**
- **Purpose**: Main packet capture and ML prediction system
- **Does**: Captures packets → Extracts features → Runs ML models → Creates alerts
- **Output**: Sends predictions and alerts to Kafka

### **2. url_threat_detector.py**
- **Purpose**: Detects malicious websites
- **Does**: Monitors DNS/HTTP → Checks threat lists → Creates alerts
- **Output**: URL threat alerts to Kafka

### **3. honeypot_system.py**
- **Purpose**: Catches attackers trying to access fake services
- **Does**: Runs fake services → Detects connections → Creates alerts
- **Output**: Honeypot alerts to Kafka

### **4. production_dashboard.py**
- **Purpose**: Web interface to view threats
- **Does**: Reads from Kafka → Displays alerts → Real-time updates
- **Output**: Web dashboard at http://localhost:5050

### **5. Kafka (Message Queue)**
- **Purpose**: Stores and distributes data
- **Topics**:
  - `network_packets`: Raw packet data
  - `ml_predictions`: ML model predictions
  - `security_alerts`: All security alerts

### **6. Redis (Cache)**
- **Purpose**: Fast storage for recent alerts
- **Stores**: Recent threats, statistics, session data

---

## 📈 Data Flow Summary

```
Network Traffic
    ↓
Packet Capture (Scapy)
    ↓
Feature Extraction (40+ features)
    ↓
Kafka: network_packets
    ↓
ML Models (4 models)
    ↓
Kafka: ml_predictions
    ↓
Threat Detection
    ↓
Kafka: security_alerts
    ↓
Dashboard (WebSocket)
    ↓
User Sees Alert!
```

---

## 🔧 How to Start Everything

### **1. Start Infrastructure:**
```bash
docker-compose up -d
```
Starts: Kafka, Redis, Zookeeper

### **2. Start Monitoring:**
```bash
./start_monitoring.sh
```
Starts:
- `live_data_streaming_system.py` (packet capture + ML)
- `url_threat_detector.py` (URL threats)

### **3. Start Dashboard:**
```bash
python3 production_dashboard.py
```
Starts web dashboard at `http://localhost:5050`

### **4. Optional - Start Honeypot:**
```bash
python3 honeypot_system.py
```
Deploys fake services to catch attackers

---

## 🎓 Key Concepts

### **1. Real-Time Processing**
- Packets are processed as they arrive
- No delay, instant detection
- Alerts appear within seconds

### **2. Machine Learning**
- Models trained on attack datasets
- Learn patterns of attacks
- Predict new threats automatically

### **3. Multiple Detection Methods**
- **ML Models**: Detect attack patterns
- **URL Detector**: Detect malicious websites
- **Honeypot**: Catch active attackers

### **4. Scalability**
- Kafka handles millions of packets
- Can process high-speed networks
- Multiple systems can run in parallel

---

## 📊 Example: Detecting a DoS Attack

```
1. Attacker sends 1000 packets/second to your server
   │
   ▼
2. System captures all packets
   - Extracts: High packet count, same destination
   │
   ▼
3. ML Model Analyzes:
   - Pattern: Unusual volume
   - Pattern: Same destination IP
   - Pattern: Short connection duration
   │
   ▼
4. Model Predicts:
   - Attack type: DoS
   - Confidence: 92%
   │
   ▼
5. Alert Created:
   - Severity: HIGH
   - Description: "DoS attack detected"
   │
   ▼
6. Dashboard Shows:
   - Red alert in Security Alerts
   - Shows attacker IP
   - Shows confidence level
```

---

## ✅ Summary

**The system works by:**
1. **Capturing** all network packets
2. **Extracting** features from packets
3. **Analyzing** with ML models
4. **Detecting** threats automatically
5. **Alerting** in real-time
6. **Displaying** on web dashboard

**Everything happens in real-time** - from packet capture to alert display takes only **2-5 seconds**!

---

## 🔍 Files and Their Roles

| File | Purpose |
|------|---------|
| `live_data_streaming_system.py` | Main system: packet capture + ML |
| `url_threat_detector.py` | URL/domain threat detection |
| `honeypot_system.py` | Honeypot for catching attackers |
| `production_dashboard.py` | Web dashboard interface |
| `security_alerts.py` | Alert management |
| `docker-compose.yml` | Infrastructure setup |
| `start_monitoring.sh` | Start all monitoring systems |
| `config/*.yaml` | Configuration files |
| `models/*.pkl` | Trained ML models |

---

**This is how your entire ML-IDS-IPS project works!** 🎉

