import sqlite3
import os

# Path to SQLite database
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, 'database')
DB_PATH = os.path.join(DB_DIR, 'signatures.db')

def init_db(db_path=DB_PATH):
    """
    Creates the database directory and initializes SQLite tables for:
    1. Malware Signatures (MalwareBazaar hash metadata)
    2. Scan Logs (Audit trail of file scans)
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Table for storing MalwareBazaar signatures & metadata
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS malware_signatures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sha256_hash TEXT UNIQUE NOT NULL,
            malware_family TEXT NOT NULL,
            file_type TEXT,
            file_name TEXT,
            first_seen TEXT,
            source TEXT DEFAULT 'MalwareBazaar',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Index for fast SHA-256 lookups
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_sha256 ON malware_signatures(sha256_hash)
    ''')
    
    # Table for logging scan history
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scan_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            sha256_hash TEXT NOT NULL,
            detection_method TEXT NOT NULL,
            verdict TEXT NOT NULL,
            malware_family TEXT,
            confidence_score REAL,
            scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"[+] Database initialized successfully at: {db_path}")

if __name__ == '__main__':
    init_db()
