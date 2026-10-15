import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'random_forest.pkl')

RF_DIR = os.path.join(os.path.dirname(BASE_DIR), 'Random Forest')
X_PATH = os.path.join(RF_DIR, 'processed_data', 'X_train.npy')
Y_PATH = os.path.join(RF_DIR, 'processed_data', 'y_train.npy')

def evaluate():
    print("=" * 60)
    print("      HYBRID AI MALWARE DETECTION - MODEL ACCURACY EVALUATION")
    print("=" * 60)
    
    if not os.path.exists(MODEL_PATH):
        print(f"[!] Model not found at: {MODEL_PATH}")
        return
        
    print(f"[*] Loading trained Random Forest model from: {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    
    if os.path.exists(X_PATH) and os.path.exists(Y_PATH):
        print(f"[*] Loading dataset features from: {X_PATH}...")
        X = np.load(X_PATH)
        y = np.load(Y_PATH)
        
        print(f"    - Total Dataset Samples: {X.shape[0]}")
        print(f"    - Feature Dimensions   : {X.shape[1]}")
        
        # 80/20 train/test split matching model training parameters
        _, X_val, _, y_val = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        
        print(f"[*] Evaluating on {len(y_val)} unseen validation samples...")
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_proba)
        cm = confusion_matrix(y_val, y_pred)
        
        print("\n" + "=" * 50)
        print("          EVALUATION RESULTS METRICS")
        print("=" * 50)
        print(f"  Accuracy  : {acc * 100:.2f}%")
        print(f"  Precision : {prec * 100:.2f}%")
        print(f"  Recall    : {rec * 100:.2f}%")
        print(f"  F1-Score  : {f1 * 100:.2f}%")
        print(f"  ROC-AUC   : {roc_auc:.4f}")
        print("-" * 50)
        print("\nConfusion Matrix:")
        print(f"  True Negatives  (TN - Benign Correct) : {cm[0][0]}")
        print(f"  False Positives (FP - False Alarm)    : {cm[0][1]}")
        print(f"  False Negatives (FN - Missed Malware) : {cm[1][0]}")
        print(f"  True Positives  (TP - Malware Correct): {cm[1][1]}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=["Benign", "Malware"]))
    else:
        print("[!] Cached .npy dataset not found. Evaluating sample predictions...")

if __name__ == '__main__':
    evaluate()
