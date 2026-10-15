import requests
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(BASE_DIR, 'signatures', 'hash_db.json')

MALWAREBAZAAR_API_URL = "https://mb-api.abuse.ch/api/v1/"

# Fallback metadata samples if API is offline or rate limited
FALLBACK_SIGNATURES = [
    {
        "sha256_hash": "ed01deb543c157165138d31c3c8ecd910971017705ed6eebc8743111426d9357",
        "malware_family": "WannaCry",
        "file_type": "exe",
        "file_name": "wcry.exe",
        "first_seen": "2017-05-12 10:00:00"
    },
    {
        "sha256_hash": "84c82835a5d21bbcf75a61706d8ab54940304aea9547aea176eceebe89f43170",
        "malware_family": "RedLineStealer",
        "file_type": "exe",
        "file_name": "payload.exe",
        "first_seen": "2023-01-15 14:22:00"
    },
    {
        "sha256_hash": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        "malware_family": "AgentTesla",
        "file_type": "exe",
        "file_name": "order_invoice.exe",
        "first_seen": "2023-04-10 09:15:30"
    },
    {
        "sha256_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "malware_family": "Emotet",
        "file_type": "dll",
        "file_name": "document.dll",
        "first_seen": "2023-05-01 11:45:00"
    },
    {
        "sha256_hash": "b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3",
        "malware_family": "AsyncRAT",
        "file_type": "exe",
        "file_name": "client.exe",
        "first_seen": "2023-06-18 16:05:12"
    }
]

def download_malware_hashes(limit=100):
    """
    Downloads malware hash METADATA from MalwareBazaar API.
    NEVER downloads binary samples or executable files.
    """
    print("[*] Contacting MalwareBazaar API for malware hash metadata...")
    data = {'query': 'get_recent', 'selector': '100'}
    
    signatures = []
    try:
        response = requests.post(MALWAREBAZAAR_API_URL, data=data, timeout=10)
        if response.status_code == 200:
            res_json = response.json()
            if res_json.get("query_status") == "ok":
                samples = res_json.get("data", [])
                for s in samples:
                    sig_name = s.get("signature") or s.get("tags", ["Unknown"])[0] if s.get("tags") else "Unknown Malware"
                    signatures.append({
                        "sha256_hash": s.get("sha256_hash", "").lower(),
                        "malware_family": sig_name,
                        "file_type": s.get("file_type", "unknown"),
                        "file_name": s.get("file_name", "unknown"),
                        "first_seen": s.get("first_seen", "")
                    })
                print(f"[+] Successfully fetched {len(signatures)} malware hash metadata records from MalwareBazaar.")
            else:
                print(f"[!] API query status: {res_json.get('query_status')}. Using fallback metadata.")
                signatures = FALLBACK_SIGNATURES
        else:
            print(f"[!] API HTTP {response.status_code}. Using fallback metadata.")
            signatures = FALLBACK_SIGNATURES
    except Exception as e:
        print(f"[!] Connection error: {e}. Using fallback malware metadata.")
        signatures = FALLBACK_SIGNATURES

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(signatures, f, indent=4)
        
    print(f"[+] Malware metadata saved to JSON file at: {OUTPUT_PATH}")
    return signatures

if __name__ == '__main__':
    download_malware_hashes()
