import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib

import feature_names

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

for d in [RESULTS_DIR, PLOTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

def load_all_datasets():
    print("=" * 65)
    print(" LOADING DATASETS (EMBER 2017, EMBER 2018, BODMAS)")
    print("=" * 65)
    
    # EMBER 2017
    p2017 = "processed_data/ember2017_benchmark.npz"
    d2017 = np.load(p2017)
    X_tr_2017, y_tr_2017 = d2017['X_train'], d2017['y_train']
    X_te_2017, y_te_2017 = d2017['X_test'], d2017['y_test']
    print(f"EMBER 2017: Train={X_tr_2017.shape}, Test={X_te_2017.shape}")

    # EMBER 2018
    p2018 = "processed_data/ember2018_benchmark.npz"
    d2018 = np.load(p2018)
    X_tr_2018, y_tr_2018 = d2018['X_train'], d2018['y_train']
    X_te_2018, y_te_2018 = d2018['X_test'], d2018['y_test']
    print(f"EMBER 2018: Train={X_tr_2018.shape}, Test={X_te_2018.shape}")

    # BODMAS
    pbodmas = "BODMAS dataset/processed_data/bodmas.npz"
    dbodmas = np.load(pbodmas)
    X_bodmas, y_bodmas = dbodmas['X'], dbodmas['y']
    X_tr_bodmas, X_te_bodmas, y_tr_bodmas, y_te_bodmas = train_test_split(
        X_bodmas, y_bodmas, test_size=0.2, random_state=42, stratify=y_bodmas
    )
    print(f"BODMAS:     Train={X_tr_bodmas.shape}, Test={X_te_bodmas.shape}")
    
    return {
        "2017": {"X_train": X_tr_2017, "y_train": y_tr_2017, "X_test": X_te_2017, "y_test": y_te_2017},
        "2018": {"X_train": X_tr_2018, "y_train": y_tr_2018, "X_test": X_te_2018, "y_test": y_te_2018},
        "BODMAS": {"X_train": X_tr_bodmas, "y_train": y_tr_bodmas, "X_test": X_te_bodmas, "y_test": y_te_bodmas}
    }

def evaluate_model(clf, X_test, y_test, batch_size=1000):
    t0 = time.time()
    y_pred = clf.predict(X_test)
    inference_time_ms = (time.time() - t0) * 1000.0 / (len(y_test) / batch_size)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "latency_ms_per_1k": float(inference_time_ms)
    }

def run_experiment_pipeline():
    data = load_all_datasets()
    
    # -------------------------------------------------------------
    # PHASE 1: Baseline Full Models (2,381 Features)
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print(" PHASE 1: BASELINE RANDOM FOREST MODELS (Full 2,381 Features)")
    print("=" * 65)

    rf_params = dict(n_estimators=100, max_depth=30, random_state=42, n_jobs=-1)
    
    print("Training Baseline Model on EMBER 2017...")
    t0 = time.time()
    rf_2017 = RandomForestClassifier(**rf_params)
    rf_2017.fit(data["2017"]["X_train"], data["2017"]["y_train"])
    train_time_2017 = time.time() - t0
    print(f"Trained in {train_time_2017:.2f}s.")

    print("Training Baseline Model on EMBER 2018...")
    t0 = time.time()
    rf_2018 = RandomForestClassifier(**rf_params)
    rf_2018.fit(data["2018"]["X_train"], data["2018"]["y_train"])
    train_time_2018 = time.time() - t0
    print(f"Trained in {train_time_2018:.2f}s.")

    # Cross-dataset evaluation matrix
    print("\nEvaluating Baseline Cross-Dataset Generalization...")
    matrix_results = {}
    for train_name, model in [("EMBER 2017", rf_2017), ("EMBER 2018", rf_2018)]:
        matrix_results[train_name] = {}
        for test_key, test_name in [("2017", "EMBER 2017 Test"), ("2018", "EMBER 2018 Test"), ("BODMAS", "BODMAS Test")]:
            metrics = evaluate_model(model, data[test_key]["X_test"], data[test_key]["y_test"])
            matrix_results[train_name][test_name] = metrics
            print(f"  [{train_name}] -> [{test_name}]: Acc={metrics['accuracy']*100:.2f}%, F1={metrics['f1']:.4f}, AUC={metrics['roc_auc']:.4f}")

    # Generalization gap onto BODMAS
    gap_2017 = matrix_results["EMBER 2017"]["EMBER 2017 Test"]["accuracy"] - matrix_results["EMBER 2017"]["BODMAS Test"]["accuracy"]
    gap_2018 = matrix_results["EMBER 2018"]["EMBER 2018 Test"]["accuracy"] - matrix_results["EMBER 2018"]["BODMAS Test"]["accuracy"]
    print(f"\nGeneralization Gap onto BODMAS:")
    print(f"  EMBER 2017 Baseline: {gap_2017*100:.2f}% drop")
    print(f"  EMBER 2018 Baseline: {gap_2018*100:.2f}% drop")

    # -------------------------------------------------------------
    # PHASE 2: Cross-Dataset Feature Stability Analysis
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print(" PHASE 2: FEATURE IMPORTANCE STABILITY ANALYSIS")
    print("=" * 65)

    imp_2017 = rf_2017.feature_importances_
    imp_2018 = rf_2018.feature_importances_

    # Spearman rank correlation across all 2381 features
    rho, pval = spearmanr(imp_2017, imp_2018)
    print(f"Spearman Rank Correlation across 2,381 features: rho = {rho:.4f} (p = {pval:.4e})")

    # Ranks (1 = highest importance)
    rank_2017 = np.argsort(np.argsort(-imp_2017)) + 1
    rank_2018 = np.argsort(np.argsort(-imp_2018)) + 1

    # Stability Jaccard Overlap
    jaccard_results = {}
    for k in [20, 50, 100, 200, 500]:
        top_k_2017 = set(np.argsort(-imp_2017)[:k])
        top_k_2018 = set(np.argsort(-imp_2018)[:k])
        inter = len(top_k_2017 & top_k_2018)
        union = len(top_k_2017 | top_k_2018)
        jaccard = inter / union
        jaccard_results[k] = {"intersection": inter, "union": union, "jaccard": jaccard}
        print(f"  Top-{k:3d} Overlap: {inter}/{k} shared features (Jaccard = {jaccard:.3f})")

    # Aggregate Rank (Average of ranks in 2017 and 2018)
    mean_rank = (rank_2017 + rank_2018) / 2.0
    stable_sorted_indices = np.argsort(mean_rank)

    # Save Top Stable Features Table
    top_features_records = []
    for rank_pos, feat_idx in enumerate(stable_sorted_indices[:50], 1):
        name, cat = feature_names.get_feature_metadata(feat_idx)
        top_features_records.append({
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
    df_top = pd.DataFrame(top_features_records)
    df_top.to_csv(os.path.join(RESULTS_DIR, "top_50_stable_features.csv"), index=False)
    print(f"\nTop 50 stable features saved to '{RESULTS_DIR}/top_50_stable_features.csv'.")
    print(df_top[["Stable_Rank", "Feature_Name", "Category", "Importance_2018", "Mean_Rank"]].head(10).to_string(index=False))

    # -------------------------------------------------------------
    # PHASE 3: Lightweight Random Forest Evaluation on BODMAS (OOD)
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print(" PHASE 3: LIGHTWEIGHT RF (TOP-K) EVALUATION ON BODMAS (OOD)")
    print("=" * 65)

    k_values = [20, 50, 100, 200]
    comparison_results = []

    # Benchmark Full Feature Reference (Trained on 2018, Evaluated on BODMAS)
    base_metrics = matrix_results["EMBER 2018"]["BODMAS Test"]
    joblib_path_full = os.path.join(MODELS_DIR, "rf_full_2381.joblib")
    joblib.dump(rf_2018, joblib_path_full, compress=3)
    full_model_size_mb = os.path.getsize(joblib_path_full) / (1024 * 1024)

    comparison_results.append({
        "Method": "Full Feature Baseline",
        "K_Features": 2381,
        "Accuracy_BODMAS": base_metrics["accuracy"],
        "F1_BODMAS": base_metrics["f1"],
        "ROC_AUC_BODMAS": base_metrics["roc_auc"],
        "Latency_ms_per_1k": base_metrics["latency_ms_per_1k"],
        "Model_Size_MB": round(full_model_size_mb, 2),
        "Gen_Gap_to_BODMAS": round(gap_2018, 4)
    })

    for k in k_values:
        # 1. Single-Dataset Selection (Top-K from 2018 alone)
        indices_single = np.argsort(-imp_2018)[:k]
        rf_single = RandomForestClassifier(**rf_params)
        rf_single.fit(data["2018"]["X_train"][:, indices_single], data["2018"]["y_train"])
        in_dist_acc_s = accuracy_score(data["2018"]["y_test"], rf_single.predict(data["2018"]["X_test"][:, indices_single]))
        metrics_single = evaluate_model(rf_single, data["BODMAS"]["X_test"][:, indices_single], data["BODMAS"]["y_test"])
        
        joblib_path_single = os.path.join(MODELS_DIR, f"rf_single_k{k}.joblib")
        joblib.dump(rf_single, joblib_path_single, compress=3)
        size_mb_single = os.path.getsize(joblib_path_single) / (1024 * 1024)

        comparison_results.append({
            "Method": f"Single-Dataset Top-{k}",
            "K_Features": k,
            "Accuracy_BODMAS": metrics_single["accuracy"],
            "F1_BODMAS": metrics_single["f1"],
            "ROC_AUC_BODMAS": metrics_single["roc_auc"],
            "Latency_ms_per_1k": metrics_single["latency_ms_per_1k"],
            "Model_Size_MB": round(size_mb_single, 2),
            "Gen_Gap_to_BODMAS": round(in_dist_acc_s - metrics_single["accuracy"], 4)
        })

        # 2. Proposed Cross-Dataset Stable Selection (Top-K from Mean Rank 2017+2018)
        indices_stable = stable_sorted_indices[:k]
        rf_stable = RandomForestClassifier(**rf_params)
        rf_stable.fit(data["2018"]["X_train"][:, indices_stable], data["2018"]["y_train"])
        in_dist_acc_st = accuracy_score(data["2018"]["y_test"], rf_stable.predict(data["2018"]["X_test"][:, indices_stable]))
        metrics_stable = evaluate_model(rf_stable, data["BODMAS"]["X_test"][:, indices_stable], data["BODMAS"]["y_test"])

        joblib_path_stable = os.path.join(MODELS_DIR, f"rf_stable_k{k}.joblib")
        joblib.dump(rf_stable, joblib_path_stable, compress=3)
        size_mb_stable = os.path.getsize(joblib_path_stable) / (1024 * 1024)

        comparison_results.append({
            "Method": f"Proposed Stable Top-{k}",
            "K_Features": k,
            "Accuracy_BODMAS": metrics_stable["accuracy"],
            "F1_BODMAS": metrics_stable["f1"],
            "ROC_AUC_BODMAS": metrics_stable["roc_auc"],
            "Latency_ms_per_1k": metrics_stable["latency_ms_per_1k"],
            "Model_Size_MB": round(size_mb_stable, 2),
            "Gen_Gap_to_BODMAS": round(in_dist_acc_st - metrics_stable["accuracy"], 4)
        })

        print(f"\n  [K={k:3d}] Single-Dataset: Acc={metrics_single['accuracy']*100:.2f}%, Gap={in_dist_acc_s - metrics_single['accuracy']:.4f}")
        print(f"  [K={k:3d}] Proposed Stable: Acc={metrics_stable['accuracy']*100:.2f}%, Gap={in_dist_acc_st - metrics_stable['accuracy']:.4f}")

    df_comp = pd.DataFrame(comparison_results)
    df_comp.to_csv(os.path.join(RESULTS_DIR, "lightweight_rf_comparison.csv"), index=False)

    print("\n" + "=" * 65)
    print(" SUMMARY OF COMPARATIVE RESULTS (Target Domain: BODMAS)")
    print("=" * 65)
    print(df_comp.to_string(index=False))

    # -------------------------------------------------------------
    # PHASE 4: Generate Thesis-Ready Visualizations
    # -------------------------------------------------------------
    print("\nGenerating Thesis-Ready Plots...")
    sns.set_theme(style="whitegrid")

    # Plot 1: Feature Stability Correlation Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(imp_2017, imp_2018, alpha=0.4, color="#1f77b4", edgecolors='none', s=25)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("EMBER 2017 Feature Importance (log scale)", fontsize=11)
    plt.ylabel("EMBER 2018 Feature Importance (log scale)", fontsize=11)
    plt.title(f"Cross-Dataset Feature Importance Stability (Spearman rho = {rho:.3f})", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_stability_scatter.png"), dpi=300)
    plt.close()

    # Plot 2: OOD Accuracy vs Number of Features (Lightweight Tradeoff)
    plt.figure(figsize=(9, 5))
    df_sub = df_comp[df_comp["K_Features"] < 2381].copy()
    
    k_vals = sorted(list(set(df_sub["K_Features"])))
    acc_single = [df_comp[df_comp["Method"] == f"Single-Dataset Top-{k}"]["Accuracy_BODMAS"].values[0] * 100 for k in k_vals]
    acc_stable = [df_comp[df_comp["Method"] == f"Proposed Stable Top-{k}"]["Accuracy_BODMAS"].values[0] * 100 for k in k_vals]
    full_acc = base_metrics["accuracy"] * 100

    plt.plot(k_vals, acc_stable, marker='o', linewidth=2.2, color='#2ca02c', label='Proposed Stable Selection (2017+2018)')
    plt.plot(k_vals, acc_single, marker='s', linewidth=2.0, linestyle='--', color='#d62728', label='Single-Dataset Selection (2018 only)')
    plt.axhline(full_acc, color='gray', linestyle=':', label=f'Full 2,381 Features Baseline ({full_acc:.2f}%)')
    
    plt.xlabel("Number of Selected PE Features (K)", fontsize=11)
    plt.ylabel("Accuracy on Target OOD (BODMAS) [%]", fontsize=11)
    plt.title("OOD Generalization vs. Feature Count (Lightweight RF)", fontsize=12, fontweight='bold')
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "ood_accuracy_vs_k.png"), dpi=300)
    plt.close()

    # Plot 3: Latency & Model Size Reduction
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    
    methods = [r["Method"] for r in comparison_results]
    latencies = [r["Latency_ms_per_1k"] for r in comparison_results]
    sizes = [r["Model_Size_MB"] for r in comparison_results]

    x = np.arange(len(methods))
    width = 0.35
    ax1.bar(x - width/2, latencies, width, label='Inference Latency (ms / 1k)', color='#4a7bb0')
    ax2.bar(x + width/2, sizes, width, label='Model Size (MB)', color='#f28e2b')

    ax1.set_ylabel('Inference Latency (ms)', color='#4a7bb0', fontsize=11)
    ax2.set_ylabel('Model Size on Disk (MB)', color='#f28e2b', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=25, ha='right', fontsize=9)
    plt.title("Computational Efficiency & Footprint Reduction", fontsize=12, fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "computational_efficiency.png"), dpi=300)
    plt.close()

    # Save summary JSON
    summary_data = {
        "rho": float(rho),
        "jaccard_results": jaccard_results,
        "matrix_results": matrix_results,
        "comparison_results": comparison_results
    }
    with open(os.path.join(RESULTS_DIR, "experiment_summary.json"), 'w') as fp:
        json.dump(summary_data, fp, indent=2)

    print(f"\nAll experiments complete! Artifacts saved to '{RESULTS_DIR}/'.")

if __name__ == '__main__':
    run_experiment_pipeline()
