-- Database Schema for AI Malware Detection System

CREATE TABLE IF NOT EXISTS scan_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    filepath TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    file_hash_md5 TEXT NOT NULL,
    file_hash_sha256 TEXT NOT NULL,
    prediction TEXT NOT NULL,          -- 'Malware', 'Suspicious', 'Clean'
    risk_score REAL NOT NULL,          -- 0.0 to 100.0
    confidence REAL NOT NULL,          -- 0.0 to 100.0
    threat_type TEXT DEFAULT 'N/A',     -- 'Ransomware', 'Trojan', 'Keylogger', etc.
    scan_details TEXT,                 -- JSON payload of scan report
    scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS signatures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hash_type TEXT NOT NULL,           -- 'MD5' or 'SHA256'
    hash_value TEXT UNIQUE NOT NULL,
    threat_name TEXT NOT NULL,
    threat_type TEXT NOT NULL,
    severity TEXT NOT NULL,            -- 'Critical', 'High', 'Medium', 'Low'
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS system_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    total_scans INTEGER DEFAULT 0,
    malware_detected INTEGER DEFAULT 0,
    clean_files INTEGER DEFAULT 0,
    suspicious_files INTEGER DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
