#!/usr/bin/env python3
"""
Quick test script for new COVID19 Classifier on hackathon cases
"""

import sys
from pathlib import Path

# Add API to path
sys.path.insert(0, str(Path(__file__).parent))

from hackathon.tester import HackathonTester


def main():
    print("=" * 80)
    print("🧪 TESTING NEW COVID19 CLASSIFIER (ResNet50 MIL)")
    print("=" * 80)
    print()
    print("Model: ResNet50 2D MIL with Attention Pooling")
    print("Expected AUC: 0.9711 (avg), Best fold: 0.9839")
    print("Previous model AUC: 0.8833")
    print("Improvement: +8.81%")
    print()
    print("=" * 80)
    print()

    # Initialize tester
    tester = HackathonTester(max_workers=1, disable_validation=False)

    # Run all hackathon test cases
    results = tester.run_all_tests()

    print()
    print("=" * 80)
    print("🎯 COMPARISON WITH PREVIOUS MODEL")
    print("=" * 80)

    if results:
        correct = sum(1 for r in results if r.get("ground_truth") in (0, 1) and r["pathology"] == r["ground_truth"])
        total = sum(1 for r in results if r.get("ground_truth") in (0, 1))

        if total > 0:
            new_accuracy = correct / total
            old_accuracy = 0.33  # Previous model: 1/3

            print(f"Previous model (2.5D CNN): {old_accuracy:.1%} (1/3)")
            print(f"New model (ResNet50 MIL):  {new_accuracy:.1%} ({correct}/{total})")
            print()

            if new_accuracy > old_accuracy:
                improvement = (new_accuracy - old_accuracy) * 100
                print(f"✅ Improvement: +{improvement:.1f} percentage points")
            elif new_accuracy == old_accuracy:
                print("⚠️  Same performance - may need threshold tuning")
            else:
                print("❌ Performance degraded - check preprocessing compatibility")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()