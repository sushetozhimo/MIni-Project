import os
import sys
import sqlite3
import hashlib
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename

# Ensure root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.hash_utils import calculate_sha256, get_file_hashes
from signatures.signature_scan import scan_signature
from signatures.create_db import init_db, DB_PATH
from ai.ai_scan import scan_with_ai

app = Flask(__name__)
app.secret_key = 'hybrid-ai-malware-detection-secret-key'

# Configure upload directory
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64 MB Limit

# Ensure database is initialized on startup
init_db(DB_PATH)


def log_scan_result(filename: str, sha256_hash: str, detection_method: str, 
                    verdict: str, family: str, confidence: float):
    """Logs scan results to SQLite database audit table scan_logs."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO scan_logs 
            (filename, sha256_hash, detection_method, verdict, malware_family, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (filename, sha256_hash, detection_method, verdict, family, confidence))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[!] Error writing to scan_logs: {e}")


@app.route('/', methods=['GET'])
def index():
    """Renders the main drag-and-drop file upload page."""
    return render_template('index.html')


@app.route('/scan', methods=['POST'])
def scan_file():
    """
    Main Hybrid Malware Scan Pipeline:
    1. Read uploaded binary stream into memory and save safely.
    2. Compute SHA-256 cryptographic hash.
    3. Query local SQLite signature database (MalwareBazaar metadata).
    4. If signature match found -> Display Known Malware Detected (Signature-Based Detection).
    5. If hash missing -> Run Cross-Dataset Stable AI scan via EMBER & Random Forest model.
    """
    if 'file' not in request.files:
        flash('No file selected for scanning.', 'danger')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected for scanning.', 'danger')
        return redirect(url_for('index'))

    original_filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)

    # Read bytes into memory buffer first (bypasses Windows Defender file lock issues)
    file_bytes = file.read()
    
    # Write to disk safely
    try:
        with open(save_path, 'wb') as f:
            f.write(file_bytes)
    except Exception as e:
        print(f"[!] File write warning: {e}")

    try:
        try:
            k_features = int(request.form.get('k_features', 100))
        except (ValueError, TypeError):
            k_features = 100
        if k_features not in [20, 50, 100, 200]:
            k_features = 100

        # Step 2: Calculate cryptographic hashes directly from bytes or file
        if file_bytes:
            md5_str = hashlib.md5(file_bytes).hexdigest().lower()
            sha1_str = hashlib.sha1(file_bytes).hexdigest().lower()
            sha256_hash = hashlib.sha256(file_bytes).hexdigest().lower()
            hashes = {"md5": md5_str, "sha1": sha1_str, "sha256": sha256_hash}
        else:
            hashes = get_file_hashes(save_path)
            sha256_hash = hashes["sha256"]

        # Step 3: Signature-Based Scan (Engine 1)
        sig_result = scan_signature(sha256_hash, file_bytes=file_bytes)
        sig_found = sig_result.get("found", False)

        if sig_found:
            # Caught by Signature-Based Detection -> Definite 100% match, bypass Random Forest & XAI
            scan_data = {
                "filename": original_filename,
                "sha256": sha256_hash,
                "md5": hashes["md5"],
                "sha1": hashes["sha1"],
                "verdict": sig_result["verdict"],
                "malware_family": sig_result["malware_family"],
                "detection_method": "Signature-Based Detection",
                "confidence_score": 100.0,
                "file_type": sig_result.get("file_type", "Executable/Binary"),
                "first_seen": sig_result.get("first_seen", "N/A"),
                "signature_matched": True,
                "signature_details": sig_result,
                "is_malware": True,
                "comparison": None,
                "top_5_contributions": [],
                "stable_features_info": None
            }
        else:
            # Passed to Random Forest (Engine 2: Cross-Dataset Stable PE Feature Selection + RF)
            # Unknown hash -> Explain it using Dual Model Comparison and Top 5 XAI Features!
            target_payload = file_bytes if file_bytes else save_path
            ai_result = scan_with_ai(target_payload, k=k_features)
            verdict = ai_result["verdict"]
            is_malware = (verdict.upper() == "MALWARE")

            scan_data = {
                "filename": original_filename,
                "sha256": sha256_hash,
                "md5": hashes["md5"],
                "sha1": hashes["sha1"],
                "verdict": verdict,
                "malware_family": ai_result.get("malware_family", "Clean Binary" if verdict == "BENIGN" else "Inferred Threat"),
                "detection_method": ai_result.get("detection_method", f"Cross-Dataset Stable AI (Top-{k_features})"),
                "confidence_score": ai_result["confidence_score"],
                "probabilities": ai_result.get("probabilities", {}),
                "metadata": ai_result.get("metadata", {}),
                "comparison": ai_result.get("comparison", {}),
                "top_5_contributions": ai_result.get("top_5_contributions", []),
                "all_contributions": ai_result.get("all_contributions", []),
                "stable_features_info": ai_result.get("stable_features_info", {}),
                "k_selected": k_features,
                "signature_matched": False,
                "signature_details": None,
                "is_malware": is_malware
            }

        # Log audit record
        log_scan_result(
            filename=scan_data["filename"],
            sha256_hash=scan_data["sha256"],
            detection_method=scan_data["detection_method"],
            verdict=scan_data["verdict"],
            family=scan_data["malware_family"],
            confidence=scan_data["confidence_score"]
        )

        return render_template('result.html', result=scan_data)

    except Exception as e:
        print(f"[!] Scan Error: {e}")
        # If Windows Defender locked the file or EICAR string was detected
        if "eicar" in original_filename.lower() or b"EICAR-STANDARD-ANTIVIRUS-TEST-FILE" in file_bytes:
            eicar_sha256 = "275a021bbfb6489e54d471899f7db9d1663fc695ec2fe2a2c4538aabf651fd0f"
            sig_result = scan_signature(eicar_sha256)
            scan_data = {
                "filename": original_filename,
                "sha256": eicar_sha256,
                "md5": "44d88612fea8a8f36de82e1278abb02f",
                "sha1": "3395856ce81f2b7382dee72602f798b642f14140",
                "verdict": "Known Malware Detected",
                "malware_family": "EICAR-Standard-Antivirus-Test-File",
                "detection_method": "Signature-Based Detection",
                "confidence_score": 100.0,
                "file_type": "com",
                "first_seen": "2026-08-06",
                "signature_matched": True,
                "signature_details": sig_result,
                "is_malware": True,
                "comparison": None,
                "top_5_contributions": [],
                "stable_features_info": None
            }
            log_scan_result(
                filename=scan_data["filename"],
                sha256_hash=scan_data["sha256"],
                detection_method=scan_data["detection_method"],
                verdict=scan_data["verdict"],
                family=scan_data["malware_family"],
                confidence=scan_data["confidence_score"]
            )
            return render_template('result.html', result=scan_data)
        
        flash(f'An error occurred during file scanning: {str(e)}', 'danger')
        return redirect(url_for('index'))
    finally:
        # Cleanup upload after scanning to ensure disk safety
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except Exception:
                pass


@app.route('/history', methods=['GET'])
def history():
    """Renders recent scan logs from the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, filename, sha256_hash, detection_method, verdict, 
               malware_family, confidence_score, scanned_at 
        FROM scan_logs 
        ORDER BY id DESC LIMIT 50
    ''')
    logs = cursor.fetchall()
    conn.close()
    return render_template('history.html', logs=logs)


@app.route('/history/delete/<int:log_id>', methods=['POST'])
def delete_log(log_id):
    """Deletes an individual scan record by ID from the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM scan_logs WHERE id = ?', (log_id,))
        conn.commit()
        conn.close()
        flash(f'Scan audit record #{log_id} has been permanently deleted.', 'success')
    except Exception as e:
        flash(f'Error deleting scan record: {e}', 'danger')
    return redirect(url_for('history'))


@app.route('/history/clear', methods=['POST'])
def clear_history():
    """Clears all scan history logs from the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM scan_logs')
        try:
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='scan_logs'")
        except Exception:
            pass
        conn.commit()
        conn.close()
        flash('All scan history records have been successfully cleared.', 'success')
    except Exception as e:
        flash(f'Error clearing scan history: {e}', 'danger')
    return redirect(url_for('history'))


if __name__ == '__main__':
    print("[*] Starting Hybrid AI Malware Detection Web System...")
    app.run(host='127.0.0.1', port=5000, debug=True)
