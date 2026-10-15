import os
import pandas as pd
import numpy as np

DATASET_CSV = os.path.join(os.path.dirname(__file__), "pe_features.csv")

FEATURE_NAMES = [
    "file_size", "entropy", "string_count", "printable_ratio", "avg_string_len",
    "suspicious_api_count", "suspicious_keyword_count", "is_pe", "num_sections",
    "high_entropy_sections", "is_packed"
] + [f"hist_bin_{i}" for i in range(16)]

def generate_synthetic_dataset(samples_per_class: int = 400) -> pd.DataFrame:
    np.random.seed(42)
    
    # 1. Clean Files Generation
    clean_size = np.random.uniform(50000, 5000000, samples_per_class)
    clean_entropy = np.random.uniform(3.5, 6.2, samples_per_class)
    clean_string_count = np.random.uniform(100, 2500, samples_per_class)
    clean_printable_ratio = np.random.uniform(0.3, 0.75, samples_per_class)
    clean_avg_str_len = np.random.uniform(6.0, 15.0, samples_per_class)
    clean_api_count = np.random.poisson(0.5, samples_per_class)
    clean_kw_count = np.random.poisson(0.2, samples_per_class)
    clean_is_pe = np.random.choice([1, 0], samples_per_class, p=[0.9, 0.1])
    clean_num_sections = np.random.choice([3, 4, 5, 6], samples_per_class)
    clean_high_entropy_sec = np.zeros(samples_per_class)
    clean_is_packed = np.zeros(samples_per_class)
    
    clean_hist = np.random.dirichlet(np.ones(16) * 2, samples_per_class)
    
    clean_data = np.column_stack([
        clean_size, clean_entropy, clean_string_count, clean_printable_ratio,
        clean_avg_str_len, clean_api_count, clean_kw_count, clean_is_pe,
        clean_num_sections, clean_high_entropy_sec, clean_is_packed,
        clean_hist
    ])
    clean_df = pd.DataFrame(clean_data, columns=FEATURE_NAMES)
    clean_df["label"] = 0  # Clean
    
    # 2. Malware Files Generation
    malware_size = np.random.uniform(20000, 8000000, samples_per_class)
    malware_entropy = np.random.uniform(6.5, 7.95, samples_per_class)
    malware_string_count = np.random.uniform(20, 800, samples_per_class)
    malware_printable_ratio = np.random.uniform(0.05, 0.35, samples_per_class)
    malware_avg_str_len = np.random.uniform(4.0, 10.0, samples_per_class)
    malware_api_count = np.random.poisson(4.5, samples_per_class) + 1
    malware_kw_count = np.random.poisson(3.2, samples_per_class)
    malware_is_pe = np.random.choice([1, 0], samples_per_class, p=[0.85, 0.15])
    malware_num_sections = np.random.choice([4, 5, 6, 7, 8], samples_per_class)
    malware_high_entropy_sec = np.random.choice([1, 2, 3], samples_per_class)
    malware_is_packed = np.random.choice([1, 0], samples_per_class, p=[0.7, 0.3])
    
    malware_hist = np.random.dirichlet(np.array([0.5 if i % 2 == 0 else 2.5 for i in range(16)]), samples_per_class)
    
    malware_data = np.column_stack([
        malware_size, malware_entropy, malware_string_count, malware_printable_ratio,
        malware_avg_str_len, malware_api_count, malware_kw_count, malware_is_pe,
        malware_num_sections, malware_high_entropy_sec, malware_is_packed,
        malware_hist
    ])
    malware_df = pd.DataFrame(malware_data, columns=FEATURE_NAMES)
    malware_df["label"] = 1  # Malware
    
    df = pd.concat([clean_df, malware_df], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df

def generate_and_save_dataset():
    df = generate_synthetic_dataset(samples_per_class=500)
    df.to_csv(DATASET_CSV, index=False)
    print(f"Generated dataset saved to {DATASET_CSV} ({len(df)} total rows).")
    return df

if __name__ == "__main__":
    generate_and_save_dataset()
