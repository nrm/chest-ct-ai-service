#!/usr/bin/env python3
"""
Test on any validation set with ground truth

Usage:
    python test_validation_set.py --validation-set cancer --limit 10
    python test_validation_set.py --validation-set covid19
    python test_validation_set.py --validation-set luna16 --limit 20
"""
import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from hackathon.tester import HackathonTester


def main():
    parser = argparse.ArgumentParser(
        description='Test RadiAssist on validation datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--validation-set', required=True,
                       choices=['covid19', 'luna16', 'cancer', 'covid19_all'],
                       help='Which validation set to test')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of cases (for quick tests)')
    parser.add_argument('--disable-validation', action='store_true',
                       help='Disable DICOM validation (faster but less safe)')
    args = parser.parse_args()

    # Setup paths
    base_path = Path('/mnt/pcephfs/lct/radiassist_workspace/testsets')
    validation_dir = base_path / f'{args.validation_set}_validation'

    if not validation_dir.exists():
        print(f"❌ Validation set not found: {validation_dir}")
        print(f"   Available sets:")
        for d in base_path.iterdir():
            if d.is_dir() and d.name.endswith('_validation'):
                print(f"   - {d.name}")
        return 1

    # Load ground truth
    gt_file = validation_dir / 'ground_truth.csv'
    if not gt_file.exists():
        print(f"❌ Ground truth not found: {gt_file}")
        return 1

    gt_df = pd.read_csv(gt_file)

    if args.limit:
        gt_df = gt_df.head(args.limit)
        print(f"⚡ Testing first {args.limit} cases (quick test mode)")

    print("=" * 80)
    print(f"📂 VALIDATION SET: {args.validation_set.upper()}")
    print("=" * 80)
    print(f"📋 Total cases: {len(gt_df)}")
    print(f"   Normal: {(gt_df['ground_truth_label'] == 0).sum()}")
    print(f"   Pathology: {(gt_df['ground_truth_label'] == 1).sum()}")

    # Calculate estimated time
    avg_time = 60.7  # seconds per case from recent tests
    estimated_minutes = (len(gt_df) * avg_time) / 60
    print(f"⏱️  Estimated time: {estimated_minutes:.1f} minutes")
    print()

    # Initialize tester
    tester = HackathonTester(max_workers=1, disable_validation=args.disable_validation)

    # Update ground truth
    for _, row in gt_df.iterrows():
        case_name = row['zip_file'].replace('.zip', '')
        tester.case_ground_truth[case_name] = int(row['ground_truth_label'])

        # Store additional info
        tester.case_ground_truth_info[case_name] = {
            'dataset': args.validation_set,
            'category': row.get('category', 'unknown'),
            'confidence': row.get('confidence', 1.0),
            'zip_file': row['zip_file'],
            'study_id': row.get('study_id', 'unknown')
        }

        if 'row' in gt_df.columns:
            # Store full row for detailed analysis
            tester.case_ground_truth_info[case_name]['row'] = row.to_dict()

    # Update test path
    tester.test_data_path = validation_dir

    # Collect specific files from ground truth (respects --limit)
    test_files = []
    for _, row in gt_df.iterrows():
        zip_path = validation_dir / row['zip_file']
        if zip_path.exists():
            test_files.append(zip_path)
        else:
            print(f"⚠️  File not found: {zip_path}")

    # Run tests on specific files
    print("🚀 Starting validation tests...")
    print("=" * 80)
    results = []
    for zip_path in test_files:
        result = tester.test_single_case(str(zip_path))
        if result:
            results.append(result)

    # Save results
    if results:
        from hackathon.reporting import create_excel_output
        create_excel_output(results, str(tester.workspace_path))

    if results:
        print()
        print("=" * 80)
        print("✅ VALIDATION TESTING COMPLETE!")
        print("=" * 80)
        print(f"📊 Processed: {len(results)} cases")
        print(f"📄 Results saved to: /mnt/pcephfs/lct/radiassist_workspace/hackathon_test_results.xlsx")
        print(f"📄 CSV results: /mnt/pcephfs/lct/radiassist_workspace/hackathon_test_results.csv")
        print()

        # Quick summary
        known_results = [r for r in results if r.get("ground_truth") in (0, 1)]
        if known_results:
            correct = sum(1 for r in known_results if r["pathology"] == r["ground_truth"])
            accuracy = correct / len(known_results) if known_results else 0
            print(f"🎯 Quick Summary:")
            print(f"   Accuracy: {accuracy:.1%} ({correct}/{len(known_results)})")

            # Breakdown by category
            if args.validation_set == 'cancer':
                normal_results = [r for r in known_results if r["ground_truth"] == 0]
                pathology_results = [r for r in known_results if r["ground_truth"] == 1]

                if normal_results:
                    normal_correct = sum(1 for r in normal_results if r["pathology"] == 0)
                    print(f"   Normal detection: {normal_correct}/{len(normal_results)} ({normal_correct/len(normal_results):.1%})")

                if pathology_results:
                    pathology_correct = sum(1 for r in pathology_results if r["pathology"] == 1)
                    print(f"   Pathology detection: {pathology_correct}/{len(pathology_results)} ({pathology_correct/len(pathology_results):.1%})")

        print()
        print("💡 Tip: View detailed results with:")
        print("   head -20 /mnt/pcephfs/lct/radiassist_workspace/hackathon_test_results.csv")

        return 0
    else:
        print("❌ No results generated")
        return 1


if __name__ == '__main__':
    sys.exit(main())