import os
import math
import numpy as np
from typing import Tuple, Dict, Any, Optional

try:
    import ember
    EMBER_AVAILABLE = True
except ImportError:
    try:
        import sys
        _PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if _PARENT not in sys.path:
            sys.path.insert(0, _PARENT)
        import ember_features as ember
        EMBER_AVAILABLE = True
    except ImportError:
        EMBER_AVAILABLE = False

def extract_ember_features(file_input: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extracts standard EMBER (2381-dimensional) static feature vector 
    from a PE (Portable Executable) file path or raw bytes safely without execution.

    Returns:
        Tuple containing:
        - feature_vector (np.ndarray): 1D array of 2381 float32 values
        - metadata (Dict[str, Any]): Feature summary for UI presentation
    """
    if isinstance(file_input, (bytes, bytearray)):
        file_bytes = bytes(file_input)
    elif isinstance(file_input, str):
        if not os.path.exists(file_input):
            raise FileNotFoundError(f"File not found: {file_input}")
        with open(file_input, "rb") as f:
            file_bytes = f.read()
    else:
        file_bytes = b""

    file_size = len(file_bytes)
    
    # 1. Primary Extraction using EMBER PEFeatureExtractor
    if EMBER_AVAILABLE:
        try:
            extractor = ember.PEFeatureExtractor(feature_version=2, print_feature_warning=False)
            features = extractor.feature_vector(file_bytes)
            feature_vector = np.array(features, dtype=np.float32)
        except Exception as e:
            print(f"[!] EMBER extraction warning: {e}. Utilizing structured fallback.")
            feature_vector = generate_fallback_feature_vector(file_bytes)
    else:
        print("[!] EMBER library not loaded. Utilizing structured fallback feature vector.")
        feature_vector = generate_fallback_feature_vector(file_bytes)

    # 2. Extract basic summary metadata for display in the UI
    metadata = {
        "file_size_bytes": file_size,
        "is_pe_header": file_bytes.startswith(b"MZ"),
        "overall_entropy": round(calculate_entropy(file_bytes), 4),
        "feature_count": len(feature_vector),
        "sample_preview_hex": file_bytes[:16].hex() if file_bytes else ""
    }

    return feature_vector, metadata


def calculate_entropy(data: bytes) -> float:
    """Calculates Shannon Entropy of a byte string (0.0 to 8.0)."""
    if not data:
        return 0.0
    byte_counts = [0] * 256
    for b in data:
        byte_counts[b] += 1
    length = len(data)
    entropy = 0.0
    for count in byte_counts:
        if count == 0:
            continue
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


def generate_fallback_feature_vector(file_bytes: bytes, target_dim: int = 2381) -> np.ndarray:
    """
    Fallback feature generator for 2381 EMBER dimensions if 
    ember package feature extraction encounters corrupted PE headers.
    """
    vector = np.zeros(target_dim, dtype=np.float32)
    file_size = len(file_bytes)
    
    if file_size > 0:
        # Byte Histogram (first 256 dimensions)
        for b in file_bytes:
            vector[b] += 1.0
        vector[:256] /= float(file_size)
        
        # Entropy estimate
        vector[256] = float(calculate_entropy(file_bytes))
        
        # File size
        vector[257] = float(file_size)
        
    return vector

if __name__ == "__main__":
    test_path = __file__
    vec, meta = extract_ember_features(test_path)
    print(f"Extracted EMBER Feature Vector Shape: {vec.shape}")
    print(f"Metadata Summary: {meta}")
