from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


def print_resource_usage(torch, psutil):
    """Print current CPU and GPU resource usage and return stats."""
    usage = {
        "cpu_rss_gb": None,
        "gpu": None,
    }

    print("\n🖥️ Resource usage")

    if psutil is not None:
        process = psutil.Process()
        rss_gb = process.memory_info().rss / (1024 ** 3)
        usage["cpu_rss_gb"] = float(rss_gb)
        print(f"  🧠 CPU RSS: {rss_gb:.2f} GB")
    else:
        print("  🧠 CPU RSS: psutil not available")

    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        allocated_gb = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved(device_index) / (1024 ** 3)
        peak_gb = torch.cuda.max_memory_allocated(device_index) / (1024 ** 3)
        usage["gpu"] = {
            "name": props.name,
            "allocated_gb": float(allocated_gb),
            "reserved_gb": float(reserved_gb),
            "peak_gb": float(peak_gb),
        }
        print(f"  💾 GPU: {props.name}")
        print(f"     • Allocated: {allocated_gb:.2f} GB")
        print(f"     • Reserved : {reserved_gb:.2f} GB")
        print(f"     • Peak     : {peak_gb:.2f} GB")
    else:
        print("  💾 GPU: not available")

    return usage


def create_excel_output(results: Iterable[dict], workspace_path: str) -> tuple[Path, Path]:
    """Create Excel and CSV output files in the workspace."""
    output_data = []

    for result in results:
        # Format pathology localization according to hackathon requirements
        localization_str = ""
        if result.get('pathology_localization'):
            loc = result['pathology_localization']
            localization_str = f"{loc[0]},{loc[1]},{loc[2]},{loc[3]},{loc[4]},{loc[5]}"

        # Determine most dangerous pathology type
        pathology_type = ""
        if result['pathology'] == 1:
            if result.get('nodule_count', 0) > 0:
                pathology_type = "nodules_detected"
            elif result.get('ksl_available', False):
                pathology_type = "abnormal_lung_pattern"
            else:
                pathology_type = "pathological_changes"

        output_data.append({
            'path_to_study': result.get('case', 'unknown') + '.zip',
            'study_uid': result['study_uid'],
            'series_uid': result['series_uid'],
            'probability_of_pathology': result['probability_of_pathology'],
            'pathology': result['pathology'],
            'processing_status': 'Success' if result['status'] == 'SUCCESS' else 'Failure',
            'time_of_processing': result['processing_time'],
            'most_dangerous_pathology_type': pathology_type,
            'pathology_localization': localization_str,
            'nodule_count': result.get('nodule_count', 0),
            'luna_confidence': result.get('luna_avg_confidence', 0.0),
            'covid_probability': result.get('covid_probability', 0.5),
            'ksl_score': result.get('ksl_z_profile_score', 0.5),
            'timestamp': datetime.now().isoformat()
        })

    df = pd.DataFrame(output_data)

    # Save Excel file (required by hackathon)
    excel_path = Path(workspace_path) / "hackathon_test_results.xlsx"
    df.to_excel(excel_path, index=False)

    # Save CSV file (for easier viewing)
    csv_path = Path(workspace_path) / "hackathon_test_results.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n📄 Excel output saved: {excel_path}")
    print(f"📄 CSV output saved: {csv_path}")

    return excel_path, csv_path


__all__ = ["print_resource_usage", "create_excel_output"]
