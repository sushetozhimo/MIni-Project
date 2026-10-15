import json
import os
import hashlib
from typing import Dict, Any, Optional
from database.db_manager import get_all_signatures

JSON_SIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "signatures", "hash_db.json")

def load_signature_database() -> Dict[str, Any]:
    db_data = {"hashes": [], "patterns": []}
    if os.path.exists(JSON_SIG_PATH):
        try:
            with open(JSON_SIG_PATH, "r", encoding="utf-8") as f:
                db_data = json.load(f)
        except Exception as e:
            print(f"Error reading {JSON_SIG_PATH}: {e}")
            
    # Combine with DB signatures dynamically
    try:
        db_sigs = get_all_signatures()
        for s in db_sigs:
            db_data["hashes"].append({
                "hash_type": s["hash_type"],
                "hash_value": s["hash_value"].lower(),
                "threat_name": s["threat_name"],
                "threat_type": s["threat_type"],
                "severity": s["severity"],
                "description": s.get("description", "")
            })
    except Exception:
        pass

    return db_data

def compute_hashes(file_bytes: bytes) -> Dict[str, str]:
    return {
        "md5": hashlib.md5(file_bytes).hexdigest().lower(),
        "sha256": hashlib.sha256(file_bytes).hexdigest().lower(),
        "sha1": hashlib.sha1(file_bytes).hexdigest().lower()
    }

def check_signatures(file_bytes: bytes, hashes: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    if hashes is None:
        hashes = compute_hashes(file_bytes)
        
    sig_db = load_signature_database()
    
    # 1. Exact Hash Match
    for h in sig_db.get("hashes", []):
        val = h.get("hash_value", "").lower()
        if val in (hashes.get("md5"), hashes.get("sha256"), hashes.get("sha1")):
            return {
                "matched": True,
                "match_type": "Exact Hash Match",
                "hash_type": h.get("hash_type"),
                "hash_value": val,
                "threat_name": h.get("threat_name", "Unknown Malware"),
                "threat_type": h.get("threat_type", "Generic Threat"),
                "severity": h.get("severity", "High"),
                "description": h.get("description", "Matched known malware signature")
            }
            
    # 2. String / Pattern Rules Match
    text_content = file_bytes.decode("ascii", errors="ignore")
    for pat in sig_db.get("patterns", []):
        p_str = pat.get("string", "")
        if p_str and p_str in text_content:
            p_sec = pat.get("string_secondary", "")
            if not p_sec or p_sec in text_content:
                return {
                    "matched": True,
                    "match_type": "Rule Pattern Match",
                    "pattern_name": pat.get("pattern_name"),
                    "threat_name": pat.get("threat_name"),
                    "threat_type": pat.get("threat_type"),
                    "severity": pat.get("severity", "High"),
                    "description": f"Matched pattern rule '{pat.get('pattern_name')}'"
                }

    return {
        "matched": False,
        "match_type": "None",
        "threat_name": "N/A",
        "threat_type": "Clean",
        "severity": "None",
        "description": "No known signature matches found."
    }

if __name__ == "__main__":
    test_bytes = b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
    res = check_signatures(test_bytes)
    print("Test EICAR Scan result:", res)
