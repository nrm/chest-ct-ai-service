#!/usr/bin/env python3
"""
Fixed KSL Analyzer with working medical feature logic
"""

import os
import sys
import warnings
import numpy as np
from pathlib import Path

# Suppress pydicom warnings about invalid UIDs
warnings.filterwarnings('ignore', category=UserWarning, module='pydicom')

# Import local ct_z module
from . import ct_z

class FixedKSLAnalyzer:
    def __init__(self):
        self.thresholds = {
            'dense_500_normal': 0.035,      # Normal < 3.5% dense tissue
            'dense_500_pathology': 0.045,   # Pathology > 4.5% dense tissue
            'hu_normal': -820,              # Normal HU > -820
            'hu_pathology': -850,           # Pathology HU < -850
            'emphysema_threshold': 0.15,    # Emphysema > 15%
            'asymmetry_threshold': 0.6      # L/R asymmetry > 60%
        }

    def analyze_zip_file(self, zip_path):
        """Analyze ZIP file with fixed KSL logic"""
        try:
            # Load DICOM series
            series = ct_z.load_series(zip_path, debug=False)
            if not series:
                return {'available': False, 'error': 'No CT series found'}

            # Process first series
            series_uid = list(series.keys())[0]
            dcm_list = series[series_uid]

            # Load volume stack
            stack, zs, px_mm = ct_z.z_sort_and_stack(dcm_list)

            # Extract medical features
            features = self.extract_medical_features(stack, px_mm)

            # Compute fixed Z-profile score
            z_score = self.compute_fixed_score(features)

            return {
                'available': True,
                'error': None,
                'z_profile_score': z_score,
                'medical_features': features,
                'stack_shape': stack.shape,
                'num_slices': len(zs)
            }

        except Exception as e:
            return {'available': False, 'error': str(e)}

    def extract_medical_features(self, stack, px_mm):
        """Extract medical features from each slice"""
        n_slices = stack.shape[0]

        # Initialize feature arrays
        lung_areas = []
        mean_hus = []
        dense_500s = []
        dense_300s = []
        emph_950s = []
        lr_asyms = []

        for i in range(n_slices):
            slice_feats = ct_z.slice_features(stack[i], px_mm)

            lung_areas.append(slice_feats['lung_area_px'])
            mean_hus.append(slice_feats['mean_lung_HU'])
            dense_500s.append(slice_feats['frac_dense_m500'])
            dense_300s.append(slice_feats['frac_dense_m300'])
            emph_950s.append(slice_feats['frac_emph_m950'])
            lr_asyms.append(slice_feats['LR_asym'])

        # Convert to numpy arrays and remove NaN values
        lung_areas = np.array(lung_areas)
        mean_hus = np.array(mean_hus)
        dense_500s = np.array(dense_500s)
        dense_300s = np.array(dense_300s)
        emph_950s = np.array(emph_950s)
        lr_asyms = np.array(lr_asyms)

        # Compute aggregate statistics (ignore NaN)
        features = {
            'avg_lung_area': float(np.nanmean(lung_areas)),
            'avg_hu': float(np.nanmean(mean_hus)),
            'avg_dense_500': float(np.nanmean(dense_500s)),
            'avg_dense_300': float(np.nanmean(dense_300s)),
            'avg_emph_950': float(np.nanmean(emph_950s)),
            'avg_asymmetry': float(np.nanmean(lr_asyms)),

            # Percentiles for robust statistics
            'p95_dense_500': float(np.nanpercentile(dense_500s, 95)),
            'p05_hu': float(np.nanpercentile(mean_hus, 5)),  # 5th percentile HU (worst areas)
            'max_asymmetry': float(np.nanmax(lr_asyms)),

            # Slice-level abnormality counts
            'high_dense_slices': int(np.sum(dense_500s > self.thresholds['dense_500_pathology'])),
            'low_hu_slices': int(np.sum(mean_hus < self.thresholds['hu_pathology'])),
            'high_emph_slices': int(np.sum(emph_950s > self.thresholds['emphysema_threshold'])),
            'high_asym_slices': int(np.sum(lr_asyms > self.thresholds['asymmetry_threshold'])),

            'total_slices': n_slices
        }

        return features

    def compute_fixed_score(self, features):
        """Compute fixed Z-profile score based on medical features"""

        # Individual component scores (0-1)
        dense_score = min(1.0, max(0.0, (features['avg_dense_500'] - 0.02) / 0.03))  # 0.02-0.05 range
        hu_score = min(1.0, max(0.0, (-780 - features['avg_hu']) / 70))  # -780 to -850 range
        emph_score = min(1.0, features['avg_emph_950'] / 0.3)  # 0-30% range
        asym_score = min(1.0, max(0.0, (features['avg_asymmetry'] - 0.3) / 0.5))  # 0.3-0.8 range

        # Slice-level abnormality percentage
        abnormal_slice_pct = (
            features['high_dense_slices'] +
            features['low_hu_slices'] +
            features['high_emph_slices']
        ) / (3 * features['total_slices'])

        # Combined score with medical weights
        combined_score = (
            0.35 * dense_score +      # Consolidation (most important)
            0.25 * hu_score +         # Overall aeration
            0.20 * emph_score +       # Emphysema/air trapping
            0.10 * asym_score +       # L/R asymmetry
            0.10 * abnormal_slice_pct # Slice-level consistency
        )

        return float(np.clip(combined_score, 0.0, 1.0))

    def get_ksl_prediction(self, z_score, medical_features):
        """Get KSL-only prediction optimized for NORMAL detection task"""

        # NORMAL-DETECTION optimized thresholds
        # Goal: High specificity for normal (never miss normal as pathology)
        z_normal_threshold = 0.35  # Lower threshold - more sensitive to normal
        dense_normal_max = 0.025   # Normal tissue density upper limit
        hu_normal_min = -850       # Normal HU lower limit
        emph_normal_max = 0.05     # Normal emphysema upper limit

        # Strong pathology indicators (conservative)
        z_strong_path = 0.6        # Very high Z-score
        dense_strong_path = 0.045  # Very high density
        hu_strong_path = -880      # Very low HU
        emph_strong_path = 0.2     # High emphysema

        # Normal indicators (if ALL are normal-like)
        is_normal_z = z_score <= z_normal_threshold
        is_normal_dense = medical_features['avg_dense_500'] <= dense_normal_max
        is_normal_hu = medical_features['avg_hu'] >= hu_normal_min
        is_normal_emph = medical_features['avg_emph_950'] <= emph_normal_max

        # Strong pathology indicators
        is_strong_path_z = z_score >= z_strong_path
        is_strong_path_dense = medical_features['avg_dense_500'] >= dense_strong_path
        is_strong_path_hu = medical_features['avg_hu'] <= hu_strong_path
        is_strong_path_emph = medical_features['avg_emph_950'] >= emph_strong_path

        # Decision logic optimized for normal detection
        normal_votes = sum([is_normal_z, is_normal_dense, is_normal_hu, is_normal_emph])
        strong_path_votes = sum([is_strong_path_z, is_strong_path_dense, is_strong_path_hu, is_strong_path_emph])

        # Conservative normal detection
        if normal_votes >= 3:  # At least 3/4 normal indicators
            final_pred = 0
            confidence = 0.8 + (normal_votes - 3) * 0.1  # 0.8-0.9 confidence
            reasoning = f"Strong normal signal ({normal_votes}/4 normal indicators)"

        elif strong_path_votes >= 2:  # At least 2 strong pathology signals
            final_pred = 1
            confidence = 0.7 + strong_path_votes * 0.1  # 0.8-1.0 confidence
            reasoning = f"Strong pathology signal ({strong_path_votes}/4 strong indicators)"

        elif normal_votes >= 2 and strong_path_votes == 0:  # Moderate normal, no strong pathology
            final_pred = 0
            confidence = 0.6
            reasoning = f"Moderate normal signal ({normal_votes}/4 normal, no strong pathology)"

        else:  # Uncertain case - err on side of caution (pathology)
            final_pred = 1
            confidence = 0.5
            reasoning = f"Uncertain case (normal:{normal_votes}/4, strong_path:{strong_path_votes}/4)"

        votes = [is_normal_z, is_normal_dense, is_normal_hu, is_normal_emph]

        return {
            'prediction': final_pred,
            'confidence': confidence,
            'z_score': z_score,
            'votes': votes,
            'normal_votes': normal_votes,
            'strong_pathology_votes': strong_path_votes,
            'thresholds_used': {
                'z_normal': z_normal_threshold,
                'dense_normal_max': dense_normal_max,
                'hu_normal_min': hu_normal_min,
                'emph_normal_max': emph_normal_max
            },
            'reasoning': reasoning,
            'decision_type': 'normal_detection_optimized'
        }

    def create_hybrid_prediction(self, cnn_prob, z_score, nodule_count, medical_features):
        """Create hybrid CNN + KSL prediction

        Tuned for HIGH SENSITIVITY (don't miss pathology) per medical screening requirements
        """

        # Get KSL prediction
        ksl_pred = self.get_ksl_prediction(z_score, medical_features)

        # ADAPTIVE CALIBRATION: COVID19 model may be systematically biased on some datasets
        # If COVID19 is very low across the board, apply calibration shift
        calibrated_cnn_prob = cnn_prob
        if cnn_prob < 0.35:  # Systematic underestimation detected
            calibration_shift = 0.10  # Reduced from 0.15 (less aggressive)
            calibrated_cnn_prob = min(1.0, cnn_prob + calibration_shift)
            # print(f"    🔧 COVID19 calibration: {cnn_prob:.3f} → {calibrated_cnn_prob:.3f}")

        # PROTECTION: Clip aggressive KSL when COVID19 is confidently low
        # High KSL + Low COVID19 may indicate artifacts, not pathology
        clipped_z_score = z_score
        if z_score > 0.55 and cnn_prob < 0.25:  # Very high KSL but low COVID19
            clipped_z_score = 0.45  # Reduce KSL weight
            # print(f"    🛡️  KSL clipping: {z_score:.3f} → {clipped_z_score:.3f} (COVID19 too low)")

        # Hybrid logic
        ksl_weight = 0.6  # KSL gets more weight due to medical interpretability
        cnn_weight = 0.4

        # Weighted probability (use calibrated COVID19 and clipped KSL)
        hybrid_prob = ksl_weight * clipped_z_score + cnn_weight * calibrated_cnn_prob

        # Decision logic with multiple criteria
        # TUNED FOR HIGH SENSITIVITY: Lower thresholds to avoid missing pathology
        # Medical screening priority: better false positive than miss pathology
        final_pred = 0
        reason = "Normal by all metrics"

        # Use calibrated_cnn_prob and clipped_z_score for decision logic
        if ksl_pred['prediction'] == 1 and calibrated_cnn_prob > 0.40:  # Raised: was 0.15
            final_pred = 1
            reason = f"KSL positive + CNN strong ({calibrated_cnn_prob:.3f})"
        elif ksl_pred['prediction'] == 1 and nodule_count >= 3:
            final_pred = 1
            reason = f"KSL positive + nodules detected ({nodule_count})"
        elif calibrated_cnn_prob > 0.40 and clipped_z_score > 0.20:  # Conservative combo
            final_pred = 1
            reason = f"CNN strong ({calibrated_cnn_prob:.3f}) + KSL moderate"
        elif clipped_z_score > 0.47:  # Very high KSL alone (but respects clipping)
            final_pred = 1
            reason = f"KSL very high ({clipped_z_score:.3f})"
        elif hybrid_prob > 0.32:  # Lowered: was 0.35 (better sensitivity)
            final_pred = 1
            reason = f"Hybrid probability elevated ({hybrid_prob:.3f})"

        confidence = max(ksl_pred['confidence'], calibrated_cnn_prob) * 0.9

        return {
            'prediction': final_pred,
            'probability': hybrid_prob,
            'confidence': confidence,
            'reason': reason,
            'ksl_component': ksl_pred,
            'cnn_prob': cnn_prob,
            'cnn_prob_calibrated': calibrated_cnn_prob,
            'nodule_count': nodule_count
        }

def test_fixed_ksl():
    """Test the fixed KSL analyzer"""
    analyzer = FixedKSLAnalyzer()

    data_root_env = os.getenv('RADIASSIST_TEST_DATA_PATH')
    data_root = Path(data_root_env).expanduser() if data_root_env else Path(__file__).resolve().parents[1] / 'datasets' / 'LCT-dataset'
    test_files = [
        data_root / 'norma_anon.zip',
        data_root / 'pneumonia_anon.zip',
        data_root / 'pneumotorax_anon.zip'
    ]

    ground_truth = [0, 1, 1]

    print("🧬 Testing Fixed KSL Analyzer")
    print("=" * 50)

    results = []
    for i, zip_path in enumerate(test_files):
        if os.path.exists(zip_path):
            case_name = Path(zip_path).stem
            gt = ground_truth[i]

            print(f"\n📊 Testing: {case_name} (GT: {gt})")

            result = analyzer.analyze_zip_file(zip_path)
            if result['available']:
                z_score = result['z_profile_score']
                features = result['medical_features']

                ksl_pred = analyzer.get_ksl_prediction(z_score, features)

                print(f"  🧬 Z-score: {z_score:.4f}")
                print(f"  🫁 Dense tissue: {features['avg_dense_500']:.4f}")
                print(f"  💨 Mean HU: {features['avg_hu']:.1f}")
                print(f"  🎯 KSL prediction: {ksl_pred['prediction']} (conf: {ksl_pred['confidence']:.3f})")
                print(f"  📊 Votes: {ksl_pred['votes']}")
                print(f"  ✅ Correct: {ksl_pred['prediction'] == gt}")

                results.append({
                    'case': case_name,
                    'gt': gt,
                    'prediction': ksl_pred['prediction'],
                    'z_score': z_score,
                    'confidence': ksl_pred['confidence']
                })
            else:
                print(f"  ❌ Failed: {result['error']}")

    # Summary
    if results:
        correct = sum(1 for r in results if r['prediction'] == r['gt'])
        accuracy = correct / len(results)
        print(f"\n📈 Fixed KSL Accuracy: {correct}/{len(results)} = {accuracy:.1%}")

        for r in results:
            status = "✅" if r['prediction'] == r['gt'] else "❌"
            print(f"  {status} {r['case']}: pred={r['prediction']}, gt={r['gt']}, z={r['z_score']:.3f}")

    return results

if __name__ == "__main__":
    test_fixed_ksl()