from __future__ import annotations

from typing import Dict, Optional


class MedicalAggregator:
    """Combine model outputs and optional KSL analysis into final decision."""

    def __init__(self, case_ground_truth: Dict[str, int], ksl_analyzer=None):
        self.case_ground_truth = case_ground_truth
        self.ksl_analyzer = ksl_analyzer

    def aggregate(
        self,
        covid_prob: float,
        nodule_count: int,
        case_name: str,
        ksl_result: Optional[dict] = None,
        ground_truth_label: Optional[int] = None,
    ) -> dict:
        if ground_truth_label is None:
            ground_truth_label = self.case_ground_truth.get(case_name, -1)

        print(f"    🧭 Enhanced medical aggregation for {case_name}:")
        print(f"    📊 COVID19 probability: {covid_prob:.4f}")
        print(f"    🔬 Nodule count: {nodule_count}")

        if (
            self.ksl_analyzer
            and ksl_result
            and ksl_result.get("available")
            and not ksl_result.get("error")
        ):
            z_profile_score = ksl_result["z_profile_score"]
            print(f"    🧬 KSL Z-profile score: {z_profile_score:.4f}")

            ksl_pred = self.ksl_analyzer.get_ksl_prediction(
                z_profile_score, ksl_result["medical_features"]
            )
            print(
                f"    🎯 KSL prediction: {ksl_pred['prediction']} "
                f"(confidence: {ksl_pred['confidence']:.3f})"
            )

            hybrid_pred = self.ksl_analyzer.create_hybrid_prediction(
                covid_prob, z_profile_score, nodule_count, ksl_result["medical_features"]
            )
            print(
                f"    🤖 Hybrid prediction: {hybrid_pred['prediction']} "
                f"(prob: {hybrid_pred['probability']:.4f})"
            )

            return {
                "prediction": hybrid_pred["prediction"],
                "probability": hybrid_pred["probability"],
                "confidence": hybrid_pred["confidence"],
                "ground_truth": ground_truth_label,
                "reason": f"Hybrid: {hybrid_pred['reason']}",
                "method": "hybrid_cnn_ksl",
                "ksl_score": z_profile_score,
                "ksl_prediction": ksl_pred["prediction"],
                "ksl_confidence": ksl_pred["confidence"],
            }

        print("    ⚠️  KSL analysis not available, using original logic")

        high_path_threshold = 0.40
        low_normal_threshold = 0.20
        nodule_confirm_threshold = 4
        nodule_prob_floor = 0.30

        if covid_prob >= high_path_threshold:
            final_prediction = 1
            confidence = min(0.95, 0.6 + (covid_prob - high_path_threshold) * 3.0)
            reason = f"COVID19 elevated ({covid_prob:.3f} ≥ {high_path_threshold})"
        elif covid_prob <= low_normal_threshold:
            final_prediction = 0
            confidence = min(0.95, 0.7 + (low_normal_threshold - covid_prob) * 2.0)
            reason = f"COVID19 normal ({covid_prob:.3f} ≤ {low_normal_threshold})"
        elif covid_prob >= nodule_prob_floor and nodule_count >= nodule_confirm_threshold:
            final_prediction = 1
            confidence = 0.85
            reason = (
                f"Moderate COVID19 ({covid_prob:.3f}) with nodules "
                f"({nodule_count} ≥ {nodule_confirm_threshold})"
            )
        else:
            final_prediction = 0
            confidence = 0.55 if nodule_count else 0.65
            reason = f"Moderate COVID19 ({covid_prob:.3f}) without significant nodules"

        confidence = min(confidence, 0.95)

        print(f"    🎯 Decision: {final_prediction} ({reason})")
        print(f"    🎲 Confidence: {confidence:.3f}")
        print(f"    ✅ Ground truth: {ground_truth_label}")

        return {
            "prediction": final_prediction,
            "probability": covid_prob,
            "confidence": confidence,
            "ground_truth": ground_truth_label,
            "reason": reason,
        }


__all__ = ["MedicalAggregator"]
