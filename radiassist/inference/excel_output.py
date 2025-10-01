"""
Excel Output Generator for COVID19 Classifier Results
Generates .xlsx files with required columns per hackathon specification
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Single study inference result"""
    path_to_study: str
    study_uid: str
    series_uid: str
    probability_of_pathology: float
    pathology: int  # 0 = normal, 1 = pathology
    processing_status: str  # "Success" or "Failure"
    time_of_processing: str  # ISO format timestamp
    error_message: Optional[str] = None

    def __post_init__(self):
        """Validate fields"""
        # Ensure probability is in [0, 1]
        if not 0.0 <= self.probability_of_pathology <= 1.0:
            raise ValueError(f"Probability must be in [0, 1], got {self.probability_of_pathology}")

        # Ensure pathology is binary
        if self.pathology not in [0, 1]:
            raise ValueError(f"Pathology must be 0 or 1, got {self.pathology}")

        # Ensure status is valid
        if self.processing_status not in ["Success", "Failure"]:
            raise ValueError(f"Status must be 'Success' or 'Failure', got {self.processing_status}")


class ExcelOutputGenerator:
    """
    Generates Excel output files per hackathon specification

    Required columns:
    - path_to_study: Path to original study (ZIP or directory)
    - study_uid: DICOM StudyInstanceUID
    - series_uid: DICOM SeriesInstanceUID
    - probability_of_pathology: Float in [0.0, 1.0]
    - pathology: Binary classification (0=normal, 1=pathology)
    - processing_status: "Success" or "Failure"
    - time_of_processing: ISO 8601 timestamp
    """

    def __init__(self):
        self.results: List[InferenceResult] = []

    def add_result(
        self,
        path_to_study: str,
        study_uid: str,
        series_uid: str,
        probability_of_pathology: float,
        pathology: int,
        processing_status: str = "Success",
        error_message: Optional[str] = None
    ) -> None:
        """
        Add inference result to output

        Args:
            path_to_study: Original study path
            study_uid: DICOM StudyInstanceUID
            series_uid: DICOM SeriesInstanceUID
            probability_of_pathology: Probability of pathology [0.0, 1.0]
            pathology: Binary classification (0 or 1)
            processing_status: "Success" or "Failure"
            error_message: Optional error message if failed
        """
        result = InferenceResult(
            path_to_study=path_to_study,
            study_uid=study_uid,
            series_uid=series_uid,
            probability_of_pathology=probability_of_pathology,
            pathology=pathology,
            processing_status=processing_status,
            time_of_processing=datetime.now().isoformat(),
            error_message=error_message
        )

        self.results.append(result)
        logger.info(f"Added result: {study_uid} - {processing_status} (prob={probability_of_pathology:.3f})")

    def add_failure(
        self,
        path_to_study: str,
        error_message: str,
        study_uid: str = "UNKNOWN",
        series_uid: str = "UNKNOWN"
    ) -> None:
        """
        Add failed processing result

        Args:
            path_to_study: Original study path
            error_message: Error description
            study_uid: DICOM StudyInstanceUID (UNKNOWN if not available)
            series_uid: DICOM SeriesInstanceUID (UNKNOWN if not available)
        """
        self.add_result(
            path_to_study=path_to_study,
            study_uid=study_uid,
            series_uid=series_uid,
            probability_of_pathology=0.0,
            pathology=0,
            processing_status="Failure",
            error_message=error_message
        )

    def save_to_excel(
        self,
        output_path: Path,
        include_error_column: bool = True
    ) -> None:
        """
        Save results to Excel file

        Args:
            output_path: Output .xlsx file path
            include_error_column: Include error_message column (for debugging)
        """
        if not self.results:
            logger.warning("No results to save")
            return

        # Convert results to DataFrame
        df = pd.DataFrame([asdict(r) for r in self.results])

        # Reorder columns (required columns first)
        required_columns = [
            'path_to_study',
            'study_uid',
            'series_uid',
            'probability_of_pathology',
            'pathology',
            'processing_status',
            'time_of_processing'
        ]

        if include_error_column:
            required_columns.append('error_message')

        df = df[required_columns]

        # Format probability to 4 decimal places
        df['probability_of_pathology'] = df['probability_of_pathology'].round(4)

        # Save to Excel
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(output_path, index=False, engine='openpyxl')

        logger.info(f"Saved {len(self.results)} results to {output_path}")

        # Print summary
        success_count = sum(1 for r in self.results if r.processing_status == "Success")
        failure_count = len(self.results) - success_count

        logger.info(f"Summary: {success_count} succeeded, {failure_count} failed")

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of results"""
        if not self.results:
            return {
                'total': 0,
                'success': 0,
                'failure': 0,
                'normal': 0,
                'pathology': 0,
                'mean_probability': 0.0
            }

        success_results = [r for r in self.results if r.processing_status == "Success"]

        return {
            'total': len(self.results),
            'success': len(success_results),
            'failure': len(self.results) - len(success_results),
            'normal': sum(1 for r in success_results if r.pathology == 0),
            'pathology': sum(1 for r in success_results if r.pathology == 1),
            'mean_probability': sum(r.probability_of_pathology for r in success_results) / len(success_results) if success_results else 0.0
        }

    def clear(self) -> None:
        """Clear all results"""
        self.results.clear()


def create_result_entry(
    path: str,
    study_uid: str,
    series_uid: str,
    probability: float,
    threshold: float = 0.5
) -> InferenceResult:
    """
    Convenience function to create result entry

    Args:
        path: Path to study
        study_uid: StudyInstanceUID
        series_uid: SeriesInstanceUID
        probability: Probability of pathology
        threshold: Classification threshold (default 0.5)

    Returns:
        InferenceResult object
    """
    pathology = 1 if probability >= threshold else 0

    return InferenceResult(
        path_to_study=path,
        study_uid=study_uid,
        series_uid=series_uid,
        probability_of_pathology=probability,
        pathology=pathology,
        processing_status="Success",
        time_of_processing=datetime.now().isoformat()
    )


def create_batch_excel(
    results: List[Dict[str, Any]],
    output_path: Path,
    threshold: float = 0.5
) -> None:
    """
    Convenience function to create Excel from batch results

    Args:
        results: List of result dicts with keys:
            - path_to_study
            - study_uid
            - series_uid
            - probability_of_pathology
        output_path: Output Excel file path
        threshold: Classification threshold
    """
    generator = ExcelOutputGenerator()

    for result in results:
        probability = result['probability_of_pathology']
        pathology = 1 if probability >= threshold else 0

        generator.add_result(
            path_to_study=result['path_to_study'],
            study_uid=result['study_uid'],
            series_uid=result['series_uid'],
            probability_of_pathology=probability,
            pathology=pathology,
            processing_status=result.get('processing_status', 'Success'),
            error_message=result.get('error_message', None)
        )

    generator.save_to_excel(output_path)


# Example usage
if __name__ == "__main__":
    # Test Excel generation
    generator = ExcelOutputGenerator()

    # Add success result
    generator.add_result(
        path_to_study="/path/to/norma_anon.zip",
        study_uid="1.2.840.113619.2.1.2411.1031152382.365.1.736169244",
        series_uid="1.2.840.113619.2.1.2411.1031152382.365.2.1",
        probability_of_pathology=0.15,
        pathology=0
    )

    # Add failure result
    generator.add_failure(
        path_to_study="/path/to/corrupted.zip",
        error_message="Failed to load DICOM files"
    )

    # Save to Excel
    output_path = Path("test_results.xlsx")
    generator.save_to_excel(output_path)

    # Print summary
    stats = generator.get_summary_stats()
    print(f"Summary: {stats}")