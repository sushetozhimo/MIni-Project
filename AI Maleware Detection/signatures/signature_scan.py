import sqlite3
import os
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'database', 'signatures.db')

def scan_signature(sha256_hash: str, file_bytes: bytes = None, db_path: str = DB_PATH) -> Dict[str, Any]:
    """
    Queries the local SQLite database for the given SHA-256 hash or checks
    standard malware byte signatures (e.g. EICAR).
    
    Returns:
        Dict containing match status, verdict, malware family, and detection method.
    """
    sha256_clean = sha256_hash.strip().lower()

    # 1. Check known byte signatures (e.g., standard EICAR test file)
    if file_bytes and (b"EICAR-STANDARD-ANTIVIRUS-TEST-FILE" in file_bytes):
        return {
            "found": True,
            "verdict": "Known Malware Detected",
            "malware_family": "EICAR-Standard-Antivirus-Test-File",
            "detection_method": "Signature-Based Detection",
            "file_type": "com",
            "first_seen": "1991-01-01",
            "confidence_score": 100.0,
            "sha256": sha256_clean
        }
    
    if not os.path.exists(db_path):
        return {
            "found": False,
            "verdict": "Unknown",
            "malware_family": None,
            "detection_method": "Signature-Based Detection",
            "message": "Signature database not initialized."
        }
        
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT sha256_hash, malware_family, file_type, file_name, first_seen, source
        FROM malware_signatures
        WHERE sha256_hash = ?
    ''', (sha256_clean,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "found": True,
            "verdict": "Known Malware Detected",
            "malware_family": row["malware_family"],
            "detection_method": "Signature-Based Detection",
            "file_type": row["file_type"],
            "first_seen": row["first_seen"],
            "confidence_score": 100.0,
            "sha256": row["sha256_hash"]
        }
    else:
        return {
            "found": False,
            "verdict": "Clean / Unknown Hash",
            "malware_family": None,
            "detection_method": "Signature-Based Detection",
            "confidence_score": 0.0,
            "sha256": sha256_clean
        }

if __name__ == '__main__':
    test_hash = "ed01deb543c157165138d31c3c8ecd910971017705ed6eebc8743111426d9357" # WannaCry
    result = scan_signature(test_hash)
    print("[*] Signature Scan Result for WannaCry test hash:")
    print(result)
