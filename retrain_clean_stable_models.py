#!/usr/bin/env python3
"""
Temporal Leakage Ablation & Lightweight RF Retraining Script
=============================================================
Ablates Feature #626 (COFF Timestamp) from the cross-dataset consensus candidate pool
to eliminate temporal shortcut learning, and retrains lightweight Random Forest models
(K in {20, 50, 100, 200}) on EMBER 2018.

Evaluates out-of-distribution generalization on the BODMAS benchmark and updates:
- results/models/rf_stable_k*.joblib
- AI Maleware Detection/models/rf_stable_k*.joblib
- results/selected_indices.json
- AI Maleware Detection/models/selected_indices.json
- results/top_50_stable_features.csv
- results/lightweight_rf_comparison.csv
- results/experiment_summary.json
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
APP_MODELS_DIR = os.path.join(ROOT_DIR, "AI Maleware Detection", "models")

for d in [RESULTS_DIR, MODELS_DIR, APP_MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "AI Maleware Detection"))
import feature_names
from utils.ember_features import extract_ember_features


def main():
    print("=" * 80)
    print(" TEMPORAL LEAKAGE ABLATION & CLEAN STABLE FEATURE MODEL RETRAINING")
    print("=" * 80)

    # 1. Load Data
    print("[*] Loading datasets (EMBER 2017, EMBER 2018, BODMAS)...")
    d2017 = np.load(os.path.join(ROOT_DIR, "processed_data", "ember2017_benchmark.npz"))
    d2018 = np.load(os.path.join(ROOT_DIR, "processed_data", "ember2018_benchmark.npz"))
    dbodmas = np.load(os.path.join(ROOT_DIR, "BODMAS dataset", "processed_data", "bodmas.npz"))

    X_tr_2017, y_tr_2017 = d2017['X_train'], d2017['y_train']
    X_tr_2018, y_tr_2018 = d2018['X_train'], d2018['y_train']
    X_te_2018, y_te_2018 = d2018['X_test'], d2018['y_test']

    _, X_te_bod, _, y_te_bod = train_test_split(
        dbodmas['X'], dbodmas['y'], test_size=0.2, random_state=42, stratify=dbodmas['y']
    )

    rf_params = dict(n_estimators=100, max_depth=30, random_state=42, n_jobs=-1)

    # 2. Get baseline full models & feature importances
    print("[*] Computing feature importances across EMBER 2017 and 2018...")
    rf_full_path = os.path.join(MODELS_DIR, "rf_full_2381.joblib")
    if os.path.exists(rf_full_path):
        rf_2018 = joblib.load(rf_full_path)
    else:
        rf_2018 = RandomForestClassifier(**rf_params).fit(X_tr_2018, y_tr_2018)
        joblib.dump(rf_2018, rf_full_path, compress=3)

    rf_2017 = RandomForestClassifier(**rf_params).fit(X_tr_2017, y_tr_2017)

    imp_2017 = rf_2017.feature_importances_
    imp_2018 = rf_2018.feature_importances_

    rank_2017 = np.argsort(np.argsort(-imp_2017)) + 1
    rank_2018 = np.argsort(np.argsort(-imp_2018)) + 1
    mean_rank = (rank_2017 + rank_2018) / 2.0
    stable_sorted_all = np.argsort(mean_rank)

    # 3. Ablate Feature 626 (COFF Timestamp)
    clean_stable_indices = [int(idx) for idx in stable_sorted_all if idx != 626]
    print(f"[*] Total clean stable candidates: {len(clean_stable_indices)} features (Timestamp #626 excluded).")

    # Save clean selected_indices.json
    k_values = [20, 50, 100, 200]
    selected_indices = {f"k{k}": clean_stable_indices[:k] for k in k_values}

    for path_dst in [os.path.join(RESULTS_DIR, "selected_indices.json"),
                     os.path.join(APP_MODELS_DIR, "selected_indices.json")]:
        with open(path_dst, "w") as fp:
            json.dump(selected_indices, fp, indent=2)
    print("[+] Updated selected_indices.json saved to results/ and AI Maleware Detection/models/.")

    # Save updated top 50 table
    top_50_records = []
    for rank_pos, feat_idx in enumerate(clean_stable_indices[:50], 1):
        name, cat = feature_names.get_feature_metadata(feat_idx)
        top_50_records.append({
            "Stable_Rank": rank_pos,
            "Feature_Index": int(feat_idx),
            "Feature_Name": name,
            "Category": cat,
            "Importance_2017": round(float(imp_2017[feat_idx]), 6),
            "Importance_2018": round(float(imp_2018[feat_idx]), 6),
            "Rank_2017": int(rank_2017[feat_idx]),
            "Rank_2018": int(rank_2018[feat_idx]),
            "Mean_Rank": round(float(mean_rank[feat_idx]), 1)
        })
    df_top50 = pd.DataFrame(top_50_records)
    df_top50.to_csv(os.path.join(RESULTS_DIR, "top_50_stable_features.csv"), index=False)
    print(f"[+] Updated top_50_stable_features.csv saved.")

    # 4. Retrain lightweight models for each K
    print("\n" + "=" * 80)
    print(" RETRAINING LIGHTWEIGHT STABLE RANDOM FOREST MODELS (TOP-K)")
    print("=" * 80)

    comparison_results = []
    
    # Baseline Full model metrics for reference
    preds_full_bod = rf_2018.predict(X_te_bod)
    probs_full_bod = rf_2018.predict_proba(X_te_bod)[:, 1]
    full_model_size_mb = os.path.getsize(rf_full_path) / (1024 * 1024)
    comparison_results.append({
        "Method": "Full Feature Baseline",
        "K_Features": 2381,
        "Accuracy_BODMAS": float(accuracy_score(y_te_bod, preds_full_bod)),
        "F1_BODMAS": float(f1_score(y_te_bod, preds_full_bod)),
        "ROC_AUC_BODMAS": float(roc_auc_score(y_te_bod, probs_full_bod)),
        "Model_Size_MB": round(full_model_size_mb, 2)
    })

    for k in k_values:
        sub_indices = selected_indices[f"k{k}"]
        print(f"\n[*] Training Clean Stable Model [K={k}] on EMBER 2018...")
        t0 = time.time()
        rf_clean = RandomForestClassifier(**rf_params).fit(X_tr_2018[:, sub_indices], y_tr_2018)
        train_time = time.time() - t0

        # Evaluate on target OOD domain (BODMAS)
        preds_bod = rf_clean.predict(X_te_bod[:, sub_indices])
        probs_bod = rf_clean.predict_proba(X_te_bod[:, sub_indices])[:, 1]
        cm = confusion_matrix(y_te_bod, preds_bod)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) * 100
        acc = accuracy_score(y_te_bod, preds_bod) * 100
        f1 = f1_score(y_te_bod, preds_bod)
        auc = roc_auc_score(y_te_bod, probs_bod)

        print(f"    Trained in {train_time:.1f}s.")
        print(f"    BODMAS OOD: Accuracy={acc:.2f}%, F1={f1:.4f}, ROC-AUC={auc:.4f}, FPR={fpr:.2f}% (FP={fp}, TN={tn})")

        # Save to results/models and web app models directory
        filename = f"rf_stable_k{k}.joblib"
        for out_dir in [MODELS_DIR, APP_MODELS_DIR]:
            target_path = os.path.join(out_dir, filename)
            joblib.dump(rf_clean, target_path, compress=3)

        size_mb = os.path.getsize(os.path.join(MODELS_DIR, filename)) / (1024 * 1024)
        comparison_results.append({
            "Method": f"Clean Stable Top-{k}",
            "K_Features": k,
            "Accuracy_BODMAS": float(acc / 100.0),
            "F1_BODMAS": float(f1),
            "ROC_AUC_BODMAS": float(auc),
            "FPR_BODMAS_pct": round(fpr, 2),
            "Model_Size_MB": round(size_mb, 2)
        })

    df_comp = pd.DataFrame(comparison_results)
    df_comp.to_csv(os.path.join(RESULTS_DIR, "lightweight_rf_comparison.csv"), index=False)
    print("\n" + "=" * 80)
    print(" SUMMARY OF RETRAINED MODELS ON BODMAS OOD BENCHMARK")
    print("=" * 80)
    print(df_comp.to_string(index=False))

    # 5. Live validation on test binaries
    print("\n" + "=" * 80)
    print(" REAL-WORLD BINARY VERIFICATION WITH RETRAINED TOP-100 MODEL")
    print("=" * 80)
    rf_st100 = joblib.load(os.path.join(APP_MODELS_DIR, "rf_stable_k100.joblib"))
    k100_idx = selected_indices["k100"]

    test_bins = [
        ("Process Explorer (32-bit)", "C:/Users/User/OneDrive/Documents/Mini Project/Testing/ProcessExplorer/procexp.exe"),
        ("Process Explorer (64-bit)", "C:/Users/User/OneDrive/Documents/Mini Project/Testing/ProcessExplorer/procexp64.exe"),
        ("Clean Windows Notepad", "C:/Windows/System32/notepad.exe"),
        ("Python Executable", sys.executable),
        ("Verified Malware Sample", os.path.join(ROOT_DIR, "AI Maleware Detection", "test_samples", "ai_detected_malware_sample.exe"))
    ]

    for name, path in test_bins:
        if os.path.exists(path):
            v, _ = extract_ember_features(path)
            p_mal = rf_st100.predict_proba(v[k100_idx].reshape(1, -1))[0, 1]
            pred = rf_st100.predict(v[k100_idx].reshape(1, -1))[0]
            v_str = "MALWARE" if pred == 1 else "BENIGN"
            print(f"  {name:<30}: Verdict={v_str:<8} | P(Malware)={p_mal*100:5.1f}% | P(Benign)={(1-p_mal)*100:5.1f}%")

    print("\n[+] Retraining and validation completed successfully!")


if __name__ == '__main__':
    main()
