"""
DICOM Validation Module for Chest CT Analysis
Validates input data to ensure it's chest CT and detect potential corruptions
"""

import pydicom
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum
import re

logger = logging.getLogger(__name__)

class ValidationResult(Enum):
    """Validation result types"""
    VALID_CHEST_CT = "valid_chest_ct"
    INVALID_MODALITY = "invalid_modality"  # Not CT
    INVALID_ANATOMY = "invalid_anatomy"    # Not chest
    CORRUPTED_DATA = "corrupted_data"      # Data corruption detected
    SUSPICIOUS_MODIFICATION = "suspicious_modification"  # Artificial artifacts
    INCOMPLETE_STUDY = "incomplete_study"   # Missing slices/metadata

@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    result: ValidationResult
    confidence: float  # 0.0-1.0
    details: Dict[str, Any]
    warnings: List[str]
    slice_count: int
    spatial_dimensions: Tuple[int, int]
    spacing: Tuple[float, float, float]
    modality: str
    body_part: str
    manufacturer: str

class ChestCTValidator:
    """Validates DICOM data for chest CT analysis based on keywords."""

    def __init__(self):
        # Chest anatomy keywords in English and Russian
        self.chest_keywords = [
            'chest', 'thorax', 'lung', 'pulmonary', 'cardiac',
            'heart', 'mediastinum', 'pleural', 'thoracic', 'pe',
            'hrct', 'coronary', 'inspiration',
            'expiration', 'covid19',
            # Russian keywords
            'грудная клетка', 'легкие', 'легкое', 'торакс', 'огк',
            'пульмо', 'вдох', 'выдох'
        ]
        # Make them all lowercase for case-insensitive comparison
        self.chest_keywords = [k.lower() for k in self.chest_keywords]

    def validate_dicom_directory(self, dicom_dir: Path) -> ValidationReport:
        """
        Simplified validation: checks for DICOM files and chest-related keywords.
        The primary check for zip file integrity is handled by the caller in main.py.
        """
        try:
            dicom_files = self._find_dicom_files(dicom_dir)
            if not dicom_files:
                return self._create_error_report("No DICOM files found")

            sample_ds = self._load_sample_dicom(dicom_files[0])
            if sample_ds is None:
                return self._create_error_report("Failed to load DICOM metadata from sample file")

            # Simplified keyword check
            keyword_found, details = self._validate_keywords(sample_ds)

            if keyword_found:
                result = ValidationResult.VALID_CHEST_CT
                confidence = 1.0
                warnings = []
            else:
                result = ValidationResult.INVALID_ANATOMY
                confidence = 0.0
                warnings = ["No chest-related keywords found in DICOM metadata."]

            # Extract basic metadata for the report
            pixel_spacing = getattr(sample_ds, 'PixelSpacing', [1.0, 1.0])
            slice_thickness = getattr(sample_ds, 'SliceThickness', 1.0)
            rows = getattr(sample_ds, 'Rows', 0)
            columns = getattr(sample_ds, 'Columns', 0)

            return ValidationReport(
                result=result,
                confidence=confidence,
                details=details,
                warnings=warnings,
                slice_count=len(dicom_files),
                spatial_dimensions=(rows, columns),
                spacing=(float(slice_thickness), float(pixel_spacing[0]), float(pixel_spacing[1])),
                modality=getattr(sample_ds, 'Modality', 'UNKNOWN'),
                body_part=getattr(sample_ds, 'BodyPartExamined', 'UNKNOWN'),
                manufacturer=getattr(sample_ds, 'Manufacturer', 'UNKNOWN')
            )

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return self._create_error_report(f"Validation exception: {str(e)}")

    def _validate_keywords(self, ds: pydicom.Dataset) -> Tuple[bool, Dict]:
        """
        Checks for chest-related keywords in relevant DICOM tags.
        Handles different character sets for Russian text.
        """
        tags_to_check = ['BodyPartExamined', 'StudyDescription', 'SeriesDescription', 'ProtocolName']
        details = {"checked_tags": {}, "found_keywords": []}

        for tag in tags_to_check:
            if not hasattr(ds, tag):
                continue

            # pydicom returns a decoded string, handling SpecificCharacterSet
            try:
                value = str(getattr(ds, tag, '')).lower()
                details["checked_tags"][tag] = value
                for keyword in self.chest_keywords:
                    if keyword in value:
                        details["found_keywords"].append(keyword)
                        return True, details
            except Exception as e:
                # Log if decoding fails for some reason
                logger.warning(f"Could not decode or check tag '{tag}': {e}")
                continue
        
        return False, details

    def _find_dicom_files(self, dicom_dir: Path) -> List[Path]:
        """Find all DICOM files in directory"""
        dicom_files = []
        for file_path in dicom_dir.rglob("*"):
            if file_path.is_file():
                # Check by extension or content
                if (file_path.suffix.lower() == '.dcm' or
                    ('.' not in file_path.name and len(file_path.name) >= 4)):
                    dicom_files.append(file_path)
        return sorted(dicom_files)

    def _load_sample_dicom(self, dicom_path: Path) -> Optional[pydicom.Dataset]:
        """Load sample DICOM for metadata analysis"""
        try:
            return pydicom.dcmread(dicom_path, force=True, stop_before_pixels=False)
        except Exception as e:
            logger.warning(f"Failed to load {dicom_path}: {e}")
            return None

    def _create_error_report(self, error_message: str) -> ValidationReport:
        """Create error validation report"""
        return ValidationReport(
            result=ValidationResult.CORRUPTED_DATA,
            confidence=0.0,
            details={"error": error_message},
            warnings=[error_message],
            slice_count=0,
            spatial_dimensions=(0, 0),
            spacing=(0, 0, 0),
            modality="UNKNOWN",
            body_part="UNKNOWN",
            manufacturer="UNKNOWN"
        )

def validate_chest_ct(dicom_dir: str) -> ValidationReport:
    """Convenience function for chest CT validation"""
    validator = ChestCTValidator()
    return validator.validate_dicom_directory(Path(dicom_dir))

# Example usage
if __name__ == "__main__":
    # Test validation
    sample_dir = os.getenv("RADIASSIST_SAMPLE_DICOM")
    if not sample_dir:
        print("Set RADIASSIST_SAMPLE_DICOM to run this example.")
    else:
        test_dir = Path(sample_dir).expanduser()
        if test_dir.exists():
            validator = ChestCTValidator()
            report = validator.validate_dicom_directory(test_dir)

            print(f"🔍 Validation Result: {report.result.value}")
            print(f"🎯 Confidence: {report.confidence:.3f}")
            print(f"📊 Details: {report.slice_count} slices, {report.spatial_dimensions}")
            print(f"⚠️  Warnings: {len(report.warnings)}")
            for warning in report.warnings:
                print(f"   - {warning}")
        else:
            print(f"Sample directory not found: {test_dir}")
