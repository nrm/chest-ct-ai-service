#!/usr/bin/env python3
"""
Batch processing script for RadiAssist chest CT analysis.

Usage:
    python batch_process.py --input /path/to/zip_files/ --output results.xlsx
    python batch_process.py --input /path/to/zip_files/ --output results.csv
    python batch_process.py --input /path/to/zip_files/  # Auto-generates output name
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hackathon.tester import HackathonTester
from hackathon.reporting import create_excel_output


def batch_process(input_dir: Path, output_file: Path = None, max_workers: int = 1):
    """Process all ZIP files in a directory.

    Args:
        input_dir: Directory containing ZIP files
        output_file: Output file path (Excel or CSV)
        max_workers: Number of parallel workers (default: 1)
    """
    # Find all ZIP files
    zip_files = sorted(input_dir.glob("*.zip"))

    if not zip_files:
        print(f"❌ No ZIP files found in {input_dir}")
        return

    print(f"📁 Found {len(zip_files)} ZIP files in {input_dir}")
    print(f"{'='*60}")

    # Initialize tester (no ground truth needed)
    print("🔧 Initializing RadiAssist system...")
    tester = HackathonTester(max_workers=max_workers, disable_validation=False)

    print(f"\n✅ Models loaded:")
    print(f"   COVID19: {'YES' if tester.covid_model else 'NO'}")
    print(f"   LUNA16:  {'YES' if tester.luna_model else 'NO'}")
    print(f"   Cancer:  {'YES' if tester.cancer_models else 'NO'}")
    print(f"   KSL:     {'YES' if tester.ksl_analyzer else 'NO'}")

    # Process each file
    results = []
    start_time = time.time()

    for i, zip_path in enumerate(zip_files, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(zip_files)}] Processing: {zip_path.name}")
        print(f"{'='*60}")

        file_start = time.time()
        result = tester.test_single_case(str(zip_path))
        file_time = time.time() - file_start

        if result:
            result['path_to_study'] = zip_path.name
            result['processing_time_sec'] = round(file_time, 2)
            results.append(result)

            print(f"\n✅ Completed in {file_time:.1f}s")
            print(f"   Pathology: {result['pathology']} (prob: {result['probability_of_pathology']:.3f})")
        else:
            print(f"\n❌ Failed to process {zip_path.name}")
            # Add failed result
            results.append({
                'path_to_study': zip_path.name,
                'study_uid': 'UNKNOWN',
                'series_uid': 'UNKNOWN',
                'patient_id': 'UNKNOWN',
                'probability_of_pathology': 0.5,
                'pathology': -1,
                'processing_time_sec': round(file_time, 2),
                'status': 'FAILED',
            })

    total_time = time.time() - start_time

    # Summary
    print(f"\n{'='*60}")
    print(f"📊 BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total files:     {len(zip_files)}")
    print(f"Successful:      {sum(1 for r in results if r.get('status') != 'FAILED')}")
    print(f"Failed:          {sum(1 for r in results if r.get('status') == 'FAILED')}")
    print(f"Total time:      {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Average/file:    {total_time/len(zip_files):.1f}s")

    # Pathology statistics
    pathology_results = [r['pathology'] for r in results if r['pathology'] in [0, 1]]
    if pathology_results:
        pathology_count = sum(pathology_results)
        normal_count = len(pathology_results) - pathology_count
        print(f"\n📈 Results:")
        print(f"   Normal:       {normal_count} ({100*normal_count/len(pathology_results):.1f}%)")
        print(f"   Pathology:    {pathology_count} ({100*pathology_count/len(pathology_results):.1f}%)")

    # Save results
    if output_file is None:
        # Auto-generate output filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = input_dir / f"radiassist_results_{timestamp}.xlsx"

    print(f"\n💾 Saving results to {output_file}...")

    # Convert results to expected format
    formatted_results = []
    for r in results:
        formatted_results.append({
            'path_to_study': r.get('path_to_study', ''),
            'study_uid': r.get('study_uid', 'UNKNOWN'),
            'series_uid': r.get('series_uid', 'UNKNOWN'),
            'probability_of_pathology': r.get('probability_of_pathology', 0.5),
            'pathology': r.get('pathology', -1),
            'processing_status': r.get('status', 'SUCCESS'),
            'time_of_processing': r.get('processing_time_sec', 0),
            'most_dangerous_pathology_type': 'chest_abnormality',
            'pathology_localization': r.get('pathology_localization', 'N/A'),
        })

    # Save based on extension
    if output_file.suffix == '.csv':
        import pandas as pd
        df = pd.DataFrame(formatted_results)
        df.to_csv(output_file, index=False)
    else:
        # Save as Excel (default)
        create_excel_output(formatted_results, str(output_file))

    print(f"✅ Results saved successfully!")
    print(f"\n{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch process chest CT scans with RadiAssist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all ZIP files in a directory
  python batch_process.py --input /path/to/studies/

  # Specify output file
  python batch_process.py --input /path/to/studies/ --output results.xlsx

  # Use CSV format
  python batch_process.py --input /path/to/studies/ --output results.csv

  # Use multiple workers (parallel processing)
  python batch_process.py --input /path/to/studies/ --workers 2
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Directory containing ZIP files"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file path (.xlsx or .csv). Auto-generated if not specified."
    )

    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input.exists():
        print(f"❌ Input directory does not exist: {args.input}")
        sys.exit(1)

    if not args.input.is_dir():
        print(f"❌ Input path is not a directory: {args.input}")
        sys.exit(1)

    # Validate output file extension
    if args.output and args.output.suffix not in ['.xlsx', '.csv']:
        print(f"❌ Output file must be .xlsx or .csv, got: {args.output.suffix}")
        sys.exit(1)

    # Run batch processing
    try:
        batch_process(args.input, args.output, args.workers)
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during batch processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
