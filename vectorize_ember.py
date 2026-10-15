import os
import glob
import json
import time
import argparse
import numpy as np
import ember_features

def vectorize_dataset(
    jsonl_dir,
    output_npz_path,
    train_benign_target=50000,
    train_malware_target=50000,
    test_benign_target=15000,
    test_malware_target=15000,
    random_seed=42
):
    """
    Extracts balanced train and test subsets from raw EMBER JSONL files
    and saves them as a compressed .npz archive (X_train, y_train, X_test, y_test).
    Ignores unlabeled samples (label == -1).
    """
    print("=" * 70)
    print(f" Processing dataset from: '{jsonl_dir}'")
    print(f" Target output:           '{output_npz_path}'")
    print(f" Target train: {train_benign_target} Benign + {train_malware_target} Malware = {train_benign_target + train_malware_target}")
    print(f" Target test:  {test_benign_target} Benign + {test_malware_target} Malware = {test_benign_target + test_malware_target}")
    print("=" * 70)

    os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
    extractor = ember_features.PEFeatureExtractor(feature_version=2)

    # 1. Collect Train Data
    train_files = sorted(glob.glob(os.path.join(jsonl_dir, "train_features_*.jsonl")))
    if not train_files:
        raise FileNotFoundError(f"No train_features_*.jsonl files found in '{jsonl_dir}'")

    X_train_b, X_train_m = [], []
    t0 = time.time()
    print("\nReading training files...")

    for fpath in train_files:
        if len(X_train_b) >= train_benign_target and len(X_train_m) >= train_malware_target:
            break
        print(f"  Scanning '{os.path.basename(fpath)}'...")
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if len(X_train_b) >= train_benign_target and len(X_train_m) >= train_malware_target:
                    break
                obj = json.loads(line)
                lbl = obj.get('label', -1)
                if lbl == 0 and len(X_train_b) < train_benign_target:
                    vec = extractor.process_raw_features(obj)
                    X_train_b.append(vec)
                elif lbl == 1 and len(X_train_m) < train_malware_target:
                    vec = extractor.process_raw_features(obj)
                    X_train_m.append(vec)

    print(f"Collected train: {len(X_train_b)} Benign, {len(X_train_m)} Malware in {time.time() - t0:.1f}s")

    # 2. Collect Test Data
    test_files = sorted(glob.glob(os.path.join(jsonl_dir, "test_features*.jsonl")))
    X_test_b, X_test_m = [], []
    t_test = time.time()
    print("\nReading test files...")

    for fpath in test_files:
        if len(X_test_b) >= test_benign_target and len(X_test_m) >= test_malware_target:
            break
        print(f"  Scanning '{os.path.basename(fpath)}'...")
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if len(X_test_b) >= test_benign_target and len(X_test_m) >= test_malware_target:
                    break
                obj = json.loads(line)
                lbl = obj.get('label', -1)
                if lbl == 0 and len(X_test_b) < test_benign_target:
                    vec = extractor.process_raw_features(obj)
                    X_test_b.append(vec)
                elif lbl == 1 and len(X_test_m) < test_malware_target:
                    vec = extractor.process_raw_features(obj)
                    X_test_m.append(vec)

    print(f"Collected test:  {len(X_test_b)} Benign, {len(X_test_m)} Malware in {time.time() - t_test:.1f}s")

    # Stack and shuffle train
    np.random.seed(random_seed)
    X_train = np.vstack([np.array(X_train_b, dtype=np.float32), np.array(X_train_m, dtype=np.float32)])
    y_train = np.concatenate([np.zeros(len(X_train_b), dtype=np.int8), np.ones(len(X_train_m), dtype=np.int8)])
    train_perm = np.random.permutation(len(y_train))
    X_train = X_train[train_perm]
    y_train = y_train[train_perm]

    # Stack and shuffle test
    X_test = np.vstack([np.array(X_test_b, dtype=np.float32), np.array(X_test_m, dtype=np.float32)])
    y_test = np.concatenate([np.zeros(len(X_test_b), dtype=np.int8), np.ones(len(X_test_m), dtype=np.int8)])
    test_perm = np.random.permutation(len(y_test))
    X_test = X_test[test_perm]
    y_test = y_test[test_perm]

    print(f"\nFinal Shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape} (Class balance: {np.mean(y_train):.2f})")
    print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape} (Class balance: {np.mean(y_test):.2f})")

    print(f"Saving to '{output_npz_path}'...")
    np.savez_compressed(
        output_npz_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )
    file_size_mb = os.path.getsize(output_npz_path) / (1024 * 1024)
    print(f"Saved successfully! File size: {file_size_mb:.2f} MB")
    print(f"Total time: {time.time() - t0:.1f}s\n")

def main():
    parser = argparse.ArgumentParser(description="Vectorize EMBER JSONL files to benchmark NPZ")
    parser.add_argument('--dataset', choices=['2017', '2018', 'all'], default='all', help="Which dataset to vectorize")
    parser.add_argument('--train_samples', type=int, default=100000, help="Total train samples (balanced)")
    parser.add_argument('--test_samples', type=int, default=30000, help="Total test samples (balanced)")
    args = parser.parse_args()

    train_half = args.train_samples // 2
    test_half = args.test_samples // 2

    if args.dataset in ['2017', 'all']:
        vectorize_dataset(
            jsonl_dir='ember_2017_2',
            output_npz_path='processed_data/ember2017_benchmark.npz',
            train_benign_target=train_half,
            train_malware_target=train_half,
            test_benign_target=test_half,
            test_malware_target=test_half
        )

    if args.dataset in ['2018', 'all']:
        vectorize_dataset(
            jsonl_dir='EMBER 2018 dataset',
            output_npz_path='processed_data/ember2018_benchmark.npz',
            train_benign_target=train_half,
            train_malware_target=train_half,
            test_benign_target=test_half,
            test_malware_target=test_half
        )

if __name__ == '__main__':
    main()
