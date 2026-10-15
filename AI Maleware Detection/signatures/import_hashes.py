import sqlite3
import json
import os
from create_db import init_db, DB_PATH, BASE_DIR

JSON_PATH = os.path.join(BASE_DIR, 'signatures', 'hash_db.json')

def import_hashes_to_db(json_path=JSON_PATH, db_path=DB_PATH):
    """
    Reads malware metadata from hash_db.json and imports/upserts 
    records into the local SQLite database signatures.db.
    """
    # Ensure database tables exist
    init_db(db_path)
    
    if not os.path.exists(json_path):
        print(f"[!] Metadata file not found at: {json_path}")
        return 0
        
    with open(json_path, 'r') as f:
        signatures = json.load(f)
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    imported_count = 0
    for sig in signatures:
        sha256 = sig.get('sha256_hash', '').lower()
        family = sig.get('malware_family', 'Unknown Malware')
        file_type = sig.get('file_type', 'unknown')
        file_name = sig.get('file_name', 'unknown')
        first_seen = sig.get('first_seen', '')
        
        if sha256:
            cursor.execute('''
                INSERT OR IGNORE INTO malware_signatures 
                (sha256_hash, malware_family, file_type, file_name, first_seen, source)
                VALUES (?, ?, ?, ?, ?, 'MalwareBazaar')
            ''', (sha256, family, file_type, file_name, first_seen))
            
            if cursor.rowcount > 0:
                imported_count += 1

    conn.commit()
    conn.close()
    
    print(f"[+] Successfully imported {imported_count} new signatures into SQLite database.")
    return imported_count

if __name__ == '__main__':
    import_hashes_to_db()
