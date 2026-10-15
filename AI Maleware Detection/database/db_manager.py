import sqlite3
import json
import os
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "malware_detector.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    if os.path.exists(SCHEMA_PATH):
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            cursor.executescript(f.read())
    conn.commit()
    
    # Initialize stats row if empty
    cursor.execute("SELECT COUNT(*) FROM system_stats")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO system_stats (total_scans, malware_detected, clean_files, suspicious_files) VALUES (0, 0, 0, 0)")
        conn.commit()
        
    conn.close()

def log_scan_result(
    filename: str,
    filepath: str,
    file_size: int,
    file_hash_md5: str,
    file_hash_sha256: str,
    prediction: str,
    risk_score: float,
    confidence: float,
    threat_type: str,
    scan_details: Dict[str, Any]
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO scan_logs 
        (filename, filepath, file_size, file_hash_md5, file_hash_sha256, prediction, risk_score, confidence, threat_type, scan_details)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        filename,
        filepath,
        file_size,
        file_hash_md5,
        file_hash_sha256,
        prediction,
        risk_score,
        confidence,
        threat_type,
        json.dumps(scan_details)
    ))
    
    scan_id = cursor.lastrowid
    
    # Update aggregate system stats
    cursor.execute("UPDATE system_stats SET total_scans = total_scans + 1, last_updated = CURRENT_TIMESTAMP WHERE id = 1")
    if prediction == "Malware":
        cursor.execute("UPDATE system_stats SET malware_detected = malware_detected + 1 WHERE id = 1")
    elif prediction == "Suspicious":
        cursor.execute("UPDATE system_stats SET suspicious_files = suspicious_files + 1 WHERE id = 1")
    else:
        cursor.execute("UPDATE system_stats SET clean_files = clean_files + 1 WHERE id = 1")
        
    conn.commit()
    conn.close()
    return scan_id

def get_scan_history(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM scan_logs ORDER BY scanned_at DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    results = []
    for r in rows:
        item = dict(r)
        if item.get("scan_details"):
            try:
                item["scan_details"] = json.loads(item["scan_details"])
            except Exception:
                pass
        results.append(item)
    conn.close()
    return results

def get_system_stats() -> Dict[str, Any]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM system_stats WHERE id = 1")
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return {"total_scans": 0, "malware_detected": 0, "clean_files": 0, "suspicious_files": 0}

def add_signature(hash_type: str, hash_value: str, threat_name: str, threat_type: str, severity: str, description: str = "") -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO signatures (hash_type, hash_value, threat_name, threat_type, severity, description)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (hash_type.upper(), hash_value.lower(), threat_name, threat_type, severity, description))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error adding signature: {e}")
        return False
    finally:
        conn.close()

def get_all_signatures() -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM signatures ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
