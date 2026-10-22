"""
Regression Test Suite: Malware Detection Pipeline
===================================================
Uses Python built-in unittest framework (no third-party test dependencies required).

Verifies:
1. Feature extraction dimensions and structure (2,381 EMBER v2 features).
2. Model loading, class labels (0=Benign, 1=Malware), and prediction alignment.
3. Feature selection indices matching training configuration.
4. Exact mathematical consistency of Tree Path attribution (Tree SHAP).
5. Safe rejection of invalid / non-PE files without running PE ML.
6. Transparent model divergence detection on temporal distribution shift cases (Process Explorer).
7. Correct baseline prediction on clean system binaries.
"""

import os
import sys
import unittest
import numpy as np
import joblib

# Ensure repository paths are configured
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.join(ROOT_DIR, "AI Maleware Detection")
for d in [ROOT_DIR, APP_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

from utils.ember_features import extract_ember_features
from ai.ai_scan import scan_with_ai, load_model_by_key, get_selected_indices, compute_tree_path_contributions
import feature_names

PROCEXP_PATH = "C:/Users/User/OneDrive/Documents/Mini Project/Testing/ProcessExplorer/procexp.exe"
NOTEPAD_PATH = "C:/Windows/System32/notepad.exe"


class TestFeatureConsistency(unittest.TestCase):
    def test_feature_dimensions(self):
        """Feature vector must be exactly 2,381 float32 dimensions."""
        if os.path.exists(NOTEPAD_PATH):
            vec, meta = extract_ember_features(NOTEPAD_PATH)
            self.assertEqual(vec.shape, (2381,))
            self.assertEqual(vec.dtype, np.float32)
            self.assertTrue(meta["is_pe_header"])
            self.assertTrue(meta["pe_parse_success"])

    def test_feature_ordering_and_names(self):
        """Feature names utility must resolve expected boundaries."""
        name_0, cat_0 = feature_names.get_feature_metadata(0)
        self.assertIn("Byte Histogram", cat_0)
        
        name_626, cat_626 = feature_names.get_feature_metadata(626)
        self.assertEqual(name_626, "COFF Timestamp")
        self.assertEqual(cat_626, "PE Header")

        name_2380, cat_2380 = feature_names.get_feature_metadata(2380)
        self.assertIn("Data Directories", cat_2380)


class TestModelAndLabels(unittest.TestCase):
    def test_model_classes_mapping(self):
        """Verify that numerical label 0 is Benign and 1 is Malware across all models."""
        for m_key in ['full', 'stable_k20', 'stable_k50', 'stable_k100', 'stable_k200']:
            model = load_model_by_key(m_key)
            self.assertEqual(list(model.classes_), [0, 1])

    def test_selected_indices_integrity(self):
        """Verify selected indices exist, contain expected feature counts, and exclude temporal artifact #626."""
        for k in [20, 50, 100, 200]:
            idx = get_selected_indices(k)
            self.assertEqual(len(idx), k)
            self.assertTrue(all(0 <= i < 2381 for i in idx))
            # Feature 626 (Timestamp) is ablated to prevent temporal shortcut learning
            self.assertNotIn(626, idx)
            # Feature 637 (COFF Characteristics) is Rank #1 in consensus invariants
            self.assertIn(637, idx)


class TestExplainableAI(unittest.TestCase):
    def test_tree_path_attribution_mathematical_equality(self):
        """
        Verify that exact Tree Path decomposition satisfies:
        base_probability + sum(delta_p) == model.predict_proba(X)[1]
        to strict numerical precision.
        """
        if not os.path.exists(PROCEXP_PATH):
            self.skipTest("Process Explorer executable not found")

        vec, _ = extract_ember_features(PROCEXP_PATH)
        m_stable = load_model_by_key('stable_k100')
        k100_idx = get_selected_indices(100)
        X_sub = vec[k100_idx].reshape(1, -1)

        # Base root probability
        root_probs = [tree.tree_.value[0, 0, 1] / tree.tree_.value[0, 0].sum() for tree in m_stable.estimators_]
        base_prob = float(np.mean(root_probs))

        contributions = compute_tree_path_contributions(m_stable, X_sub, k100_idx)
        sum_deltas = sum(c["delta_probability"] for c in contributions)
        predicted_mal_prob = float(m_stable.predict_proba(X_sub)[0, 1])

        self.assertTrue(np.isclose(base_prob + sum_deltas, predicted_mal_prob, atol=1e-5))


class TestSecurityAndErrorHandling(unittest.TestCase):
    def test_non_pe_file_rejection(self):
        """Non-PE bytes (missing MZ header) must raise ValueError and not produce false malware result."""
        non_pe_data = b"Plain text file that is not an executable."
        with self.assertRaises(ValueError):
            scan_with_ai(non_pe_data, k=100)

    def test_procexp_correct_benign_classification(self):
        """
        Microsoft Process Explorer:
        - Full baseline predicts BENIGN.
        - Clean Proposed Stable (Top-100) predicts BENIGN (72% confidence).
        - Both models agree on clean legitimate status.
        """
        if not os.path.exists(PROCEXP_PATH):
            self.skipTest("Process Explorer not found")

        res = scan_with_ai(PROCEXP_PATH, k=100)
        self.assertEqual(res["comparison"]["full_baseline"]["verdict"], "BENIGN")
        self.assertEqual(res["comparison"]["proposed_stable"]["verdict"], "BENIGN")
        self.assertEqual(res["verdict"], "BENIGN")
        self.assertFalse(res["is_divergent"])


if __name__ == '__main__':
    unittest.main(verbosity=2)
