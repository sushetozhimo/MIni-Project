#!/usr/bin/env python3
"""
Diagnostic Pipeline Verification Tool
======================================
Reproducible diagnostic test suite for inspecting and validating the entire
inference pipeline of the Malware Detection System under Distribution Shift:

Uploaded EXE -> PE parsing -> Feature extraction -> Feature selection ->
Random Forest prediction -> Label mapping -> Threat logic -> Final displayed result.

Records all metrics requested in Project Specification Section 6.
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
import joblib
from typing import Dict, Any, List

# Ensure repository paths are configured
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(ROOT_DIR, "AI Maleware Detection")
for d in [ROOT_DIR, APP_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

from utils.ember_features import extract_ember_features
from ai.ai_scan import scan_with_ai, load_model_by_key, get_selected_indices, compute_tree_path_contributions
from signatures.signature_scan import scan_signature
import feature_names

PROCEXP_32 = "C:/Users/User/OneDrive/Documents/Mini Project/Testing/ProcessExplorer/procexp.exe"
PROCEXP_64 = "C:/Users/User/OneDrive/Documents/Mini Project/Testing/ProcessExplorer/procexp64.exe"
NOTEPAD_EXE = "C:/Windows/System32/notepad.exe"
SAMPLE_MALWARE = os.path.join(APP_DIR, "test_samples", "ai_detected_malware_sample.exe")
NON_PE_SAMPLE = os.path.join(APP_DIR, "test_samples", "sample_known_malware.txt")


def check_authenticode_signature(file_path: str) -> Dict[str, Any]:
    """Inspects Authenticode digital signature via PowerShell safely."""
    if not os.path.exists(file_path):
        return {"signed": False, "status": "File not found", "signer": "N/A"}
    try:
        import subprocess
        cmd = f"powershell -NoProfile -Command \"(Get-AuthenticodeSignature -FilePath '{file_path}').Status.ToString() + '|' + (Get-AuthenticodeSignature -FilePath '{file_path}').SignerCertificate.Subject\""
        out = subprocess.check_output(cmd, shell=True, timeout=5, stderr=subprocess.DEVNULL).decode('utf-8', errors='ignore').strip()
        parts = out.split('|', 1)
        status = parts[0].strip()
        signer = parts[1].strip() if len(parts) > 1 else "Unknown"
        return {"signed": (status == "Valid"), "status": status, "signer": signer}
    except Exception as e:
        return {"signed": False, "status": f"Signature check error: {e}", "signer": "N/A"}


def run_single_diagnostic(file_path_or_bytes: Any, display_name: str, k: int = 100) -> Dict[str, Any]:
    """Runs end-to-end diagnostic and records all audit parameters."""
    print("=" * 80)
    print(f" DIAGNOSTIC RUN: {display_name}")
    print("=" * 80)

    is_filepath = isinstance(file_path_or_bytes, str) and os.path.exists(file_path_or_bytes)
    if is_filepath:
        with open(file_path_or_bytes, "rb") as f:
            file_bytes = f.read()
        file_path = file_path_or_bytes
    else:
        file_bytes = file_path_or_bytes if isinstance(file_path_or_bytes, (bytes, bytearray)) else b""
        file_path = display_name

    file_size = len(file_bytes)
    sha256_hash = hashlib.sha256(file_bytes).hexdigest().lower() if file_bytes else "N/A"
    print(f"[*] Filename:               {display_name}")
    print(f"[*] File Size:              {file_size:,} bytes")
    print(f"[*] SHA-256 Hash:           {sha256_hash}")

    # Authenticode check
    sig_info = check_authenticode_signature(file_path) if is_filepath else {"signed": False, "status": "In-memory", "signer": "N/A"}
    print(f"[*] Authenticode Signature: {sig_info['status']} ({sig_info['signer']})")

    # Step 1: PE Parsing Validation
    is_mz = file_bytes.startswith(b"MZ") and file_size >= 64
    pe_parse_ok = False
    parse_warning = None
    vec = None
    meta = {}

    try:
        vec, meta = extract_ember_features(file_bytes)
        pe_parse_ok = meta.get("pe_parse_success", False)
    except Exception as e:
        parse_warning = str(e)

    print(f"[*] DOS 'MZ' Header:        {'YES (Valid)' if is_mz else 'NO (Malformed / Non-PE)'}")
    print(f"[*] PE Parsing Success:     {'YES (Parsed by LIEF)' if pe_parse_ok else 'NO (Fallback / Non-PE)'}")
    if parse_warning:
        print(f"    [!] Parsing Warning:    {parse_warning}")

    extracted_feature_count = len(vec) if vec is not None else 0
    selected_indices = get_selected_indices(k)
    selected_feature_count = len(selected_indices)
    missing_invalid_count = int(np.isnan(vec).sum() + np.isinf(vec).sum()) if vec is not None else -1

    print(f"[*] Extracted Dimensions:   {extracted_feature_count} features (Vector shape: {vec.shape if vec is not None else 'None'})")
    print(f"[*] Selected Dimensions:    {selected_feature_count} features (Subset: Top-{k})")
    print(f"[*] Missing/NaN Features:   {missing_invalid_count}")

    # Check Engine 1: Signature Scan
    sig_match = scan_signature(sha256_hash, file_bytes=file_bytes)
    sig_triggered = sig_match.get("found", False)
    print(f"[*] Engine 1 (Signature):   {'MATCH FOUND' if sig_triggered else 'No match in local catalog'}")
    if sig_triggered:
        print(f"    Verdict:                {sig_match['verdict']}")
        print(f"    Family:                 {sig_match['malware_family']}")

    # Check Engine 2: Random Forest Models
    m_full = load_model_by_key('full')
    m_stable = load_model_by_key(f'stable_k{k}')
    class_labels = list(m_full.classes_)
    print(f"[*] Classifier Classes:     {class_labels} (0 = Benign, 1 = Malware)")

    # Model Predictions
    if not is_mz:
        print("[!] Non-PE File: AI PE Analysis safely rejected in application pipeline.")
        app_verdict = "REJECTED (Non-PE File)"
        raw_pred_full = None
        prob_full = None
        raw_pred_stable = None
        prob_stable = None
        divergence = False
        top_drivers = []
    else:
        X_full = vec.reshape(1, -1)
        raw_pred_full = int(m_full.predict(X_full)[0])
        prob_full = m_full.predict_proba(X_full)[0]

        X_stable = vec[selected_indices].reshape(1, -1)
        raw_pred_stable = int(m_stable.predict(X_stable)[0])
        prob_stable = m_stable.predict_proba(X_stable)[0]

        v_full = "MALWARE" if raw_pred_full == 1 else "BENIGN"
        v_stable = "MALWARE" if raw_pred_stable == 1 else "BENIGN"
        divergence = (raw_pred_full != raw_pred_stable)

        print(f"\n[*] RAW MODEL OUTPUTS:")
        print(f"  -> Full Baseline (2,381 feats): Pred={raw_pred_full} ({v_full}) | P(Benign)={prob_full[0]*100:.1f}%, P(Malware)={prob_full[1]*100:.1f}%")
        print(f"  -> Proposed Stable (Top-{k:3d}): Pred={raw_pred_stable} ({v_stable}) | P(Benign)={prob_stable[0]*100:.1f}%, P(Malware)={prob_stable[1]*100:.1f}%")
        print(f"  -> Model Disagreement:          {'YES (DIVERGENT - False Positive / Distribution Shift Risk)' if divergence else 'NO (Consensus)'}")

        # Exact Tree Path Attribution
        top_drivers = compute_tree_path_contributions(m_stable, X_stable, selected_indices)[:5]
        print(f"\n[*] TOP 5 EXPLAINABLE AI DRIVERS (Exact Tree Path Decomposition):")
        for drv in top_drivers:
            print(f"    #{drv['rank']} [{drv['category']}] {drv['name']:<36} (Idx {drv['global_index']:4d}): Val={drv['value']:12.2f} | Shift={drv['impact_formatted']:>8} ({drv['direction']})")

        ai_res = scan_with_ai(file_bytes, k=k)
        app_verdict = ai_res["verdict"]

    print(f"\n[*] FINAL APPLICATION PREDICTION:")
    print(f"    Application Verdict:     {app_verdict}")
    print(f"    Divergence Flagged:      {divergence}")
    print("=" * 80 + "\n")

    return {
        "filename": display_name,
        "sha256": sha256_hash,
        "is_pe": is_mz,
        "pe_parse_success": pe_parse_ok,
        "extracted_feature_count": extracted_feature_count,
        "selected_feature_count": selected_feature_count,
        "missing_invalid_count": missing_invalid_count,
        "class_labels": class_labels,
        "raw_full_prediction": raw_pred_full,
        "full_probabilities": prob_full.tolist() if prob_full is not None else None,
        "raw_stable_prediction": raw_pred_stable,
        "stable_probabilities": prob_stable.tolist() if prob_stable is not None else None,
        "is_divergent": divergence,
        "final_application_verdict": app_verdict,
        "signature_matched": sig_triggered,
        "top_drivers": top_drivers
    }


def main():
    print("#" * 80)
    print(" MALWARE DETECTION PIPELINE - SYSTEMATIC DIAGNOSTIC SUITE")
    print(" Verifying Feature Consistency, Model Loading, and Prediction Behavior")
    print("#" * 80 + "\n")

    test_targets = [
        ("Microsoft Process Explorer (32-bit)", PROCEXP_32),
        ("Microsoft Process Explorer (64-bit)", PROCEXP_64),
        ("Clean Windows OS Binary (Notepad)", NOTEPAD_EXE),
        ("Verified Malware Sample", SAMPLE_MALWARE),
        ("Non-PE Plain Text File", NON_PE_SAMPLE)
    ]

    results = []
    for label, path in test_targets:
        if os.path.exists(path):
            res = run_single_diagnostic(path, label, k=100)
            results.append(res)
        else:
            print(f"[!] Test file '{label}' not found at: {path}. Skipping.")

    print("\n" + "#" * 80)
    print(" SUMMARY MATRIX OF ALL TEST SAMPLES")
    print("#" * 80)
    print(f"{'Sample':<38} | {'PE?':<5} | {'Full (2381)':<12} | {'Stable (Top-100)':<17} | {'Divergent?':<10} | {'Final Verdict':<14}")
    print("-" * 110)
    for r in results:
        full_str = f"MAL ({r['full_probabilities'][1]*100:.0f}%)" if r['raw_full_prediction'] == 1 else (f"BEN ({r['full_probabilities'][0]*100:.0f}%)" if r['raw_full_prediction'] == 0 else "N/A")
        st_str = f"MAL ({r['stable_probabilities'][1]*100:.0f}%)" if r['raw_stable_prediction'] == 1 else (f"BEN ({r['stable_probabilities'][0]*100:.0f}%)" if r['raw_stable_prediction'] == 0 else "N/A")
        div_str = "YES (FP)" if r['is_divergent'] else "NO"
        print(f"{r['filename']:<38} | {str(r['is_pe']):<5} | {full_str:<12} | {st_str:<17} | {div_str:<10} | {r['final_application_verdict']:<14}")
    print("-" * 110)

    return 0


if __name__ == '__main__':
    sys.exit(main())
