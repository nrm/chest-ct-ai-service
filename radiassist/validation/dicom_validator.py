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
    """Validates DICOM data for chest CT analysis"""

    def __init__(self):
        # Chest anatomy keywords
        self.chest_keywords = {
            'body_part': [
                'chest', 'thorax', 'lung', 'pulmonary', 'cardiac',
                'heart', 'mediastinum', 'pleural', 'thoracic'
            ],
            'protocol': [
                'chest', 'thorax', 'lung', 'pulmonary', 'pe',
                'hrct', 'cardiac', 'coronary', 'cta'
            ],
            'series_description': [
                'chest', 'thorax', 'lung', 'pulmonary', 'axial',
                'inspiration', 'expiration', 'prone', 'supine'
            ]
        }

        # Suspicious modification patterns
        self.suspicious_patterns = {
            'manufacturer_mismatch': r'(test|fake|modified|artificial)',
            'unusual_spacing': (0.1, 10.0),  # mm range
            'slice_count_ranges': {
                'too_few': 1,      # Accept single multi-frame DICOM files
                'too_many': 3000   # > 3000 slices suspicious
            }
        }

    def validate_dicom_directory(self, dicom_dir: Path) -> ValidationReport:
        """Main validation method for DICOM directory"""
        try:
            # Find DICOM files
            dicom_files = self._find_dicom_files(dicom_dir)
            if not dicom_files:
                return ValidationReport(
                    result=ValidationResult.CORRUPTED_DATA,
                    confidence=1.0,
                    details={"error": "No DICOM files found"},
                    warnings=["No DICOM files detected"],
                    slice_count=0,
                    spatial_dimensions=(0, 0),
                    spacing=(0, 0, 0),
                    modality="UNKNOWN",
                    body_part="UNKNOWN",
                    manufacturer="UNKNOWN"
                )

            # Load sample DICOM for metadata analysis
            sample_ds = self._load_sample_dicom(dicom_files[0])
            if sample_ds is None:
                return self._create_error_report("Failed to load DICOM metadata")

            # Perform all validation checks
            modality_check = self._validate_modality(sample_ds)
            anatomy_check = self._validate_anatomy(sample_ds, dicom_files)
            quality_check = self._validate_data_quality(dicom_files)
            corruption_check = self._detect_corruption(dicom_files)
            modification_check = self._detect_modifications(sample_ds, dicom_files)

            # Aggregate results
            return self._aggregate_validation_results(
                sample_ds, dicom_files,
                modality_check, anatomy_check, quality_check,
                corruption_check, modification_check
            )

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return self._create_error_report(f"Validation exception: {str(e)}")

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

    def _validate_modality(self, ds: pydicom.Dataset) -> Dict[str, Any]:
        """Validate imaging modality (should be CT)"""
        modality = getattr(ds, 'Modality', 'UNKNOWN').upper()

        is_valid = modality == 'CT'
        confidence = 1.0 if is_valid else 0.0

        # Check for suspicious modality modifications
        warnings = []
        if modality in ['MR', 'XA', 'US', 'NM']:
            warnings.append(f"Definitely not CT: {modality}")
        elif modality == 'UNKNOWN':
            warnings.append("Modality tag missing or corrupted")

        return {
            'valid': is_valid,
            'confidence': confidence,
            'modality': modality,
            'warnings': warnings
        }

    def _validate_anatomy(self, ds: pydicom.Dataset, dicom_files: List[Path]) -> Dict[str, Any]:
        """Validate anatomical region (should be chest/thorax)"""
        confidence_scores = []
        detected_regions = []
        warnings = []

        # Check BodyPartExamined
        body_part = getattr(ds, 'BodyPartExamined', '').lower()
        if body_part:
            chest_score = self._calculate_keyword_match(body_part, self.chest_keywords['body_part'])
            confidence_scores.append(chest_score)
            detected_regions.append(f"BodyPart: {body_part}")

        # Check StudyDescription
        study_desc = getattr(ds, 'StudyDescription', '').lower()
        if study_desc:
            study_score = self._calculate_keyword_match(study_desc, self.chest_keywords['protocol'])
            confidence_scores.append(study_score * 0.8)  # Lower weight
            detected_regions.append(f"Study: {study_desc}")

        # Check SeriesDescription
        series_desc = getattr(ds, 'SeriesDescription', '').lower()
        if series_desc:
            series_score = self._calculate_keyword_match(series_desc, self.chest_keywords['series_description'])
            confidence_scores.append(series_score * 0.6)  # Lower weight
            detected_regions.append(f"Series: {series_desc}")

        # Analyze slice count (chest CT typically 100-800 slices, but accept multi-frame DICOM)
        slice_count = len(dicom_files)
        if slice_count == 1:
            # Single file might be multi-frame DICOM - check NumberOfFrames
            try:
                ds_check = pydicom.dcmread(dicom_files[0], stop_before_pixels=True)
                if hasattr(ds_check, 'NumberOfFrames') and ds_check.NumberOfFrames > 50:
                    confidence_scores.append(1.0)  # Valid multi-frame DICOM
                else:
                    warnings.append(f"Single file without sufficient frames: {getattr(ds_check, 'NumberOfFrames', 'unknown')}")
                    confidence_scores.append(0.6)  # Lower but still acceptable
            except:
                warnings.append(f"Could not verify single file structure")
                confidence_scores.append(0.5)
        elif slice_count < 20:
            warnings.append(f"Low slice count for chest CT: {slice_count}")
            confidence_scores.append(0.4)
        elif slice_count > 1500:
            warnings.append(f"Unusually high slice count: {slice_count}")
            confidence_scores.append(0.7)
        else:
            confidence_scores.append(1.0)

        # Calculate overall anatomy confidence
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        is_valid = overall_confidence > 0.5

        return {
            'valid': is_valid,
            'confidence': overall_confidence,
            'body_part': body_part,
            'detected_regions': detected_regions,
            'warnings': warnings
        }

    def _calculate_keyword_match(self, text: str, keywords: List[str]) -> float:
        """Calculate keyword match score"""
        if not text:
            return 0.0

        matches = sum(1 for keyword in keywords if keyword in text)
        return min(matches / max(len(keywords) * 0.3, 1), 1.0)

    def _validate_data_quality(self, dicom_files: List[Path]) -> Dict[str, Any]:
        """Validate overall data quality"""
        warnings = []
        quality_score = 1.0

        # Check slice count consistency (but allow single multi-frame files)
        slice_count = len(dicom_files)
        if slice_count < self.suspicious_patterns['slice_count_ranges']['too_few']:
            # Only flag as suspicious if it's 0 files, not single multi-frame
            if slice_count == 0:
                warnings.append(f"No DICOM files found")
                quality_score *= 0.1
            # Single file is acceptable for multi-frame DICOM
        elif slice_count > self.suspicious_patterns['slice_count_ranges']['too_many']:
            warnings.append(f"Excessive slice count: {slice_count}")
            quality_score *= 0.7

        # Sample a few files to check consistency
        sample_size = min(5, len(dicom_files))
        sample_files = dicom_files[::max(1, len(dicom_files)//sample_size)]

        dimensions = []
        spacings = []
        for file_path in sample_files:
            try:
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                if hasattr(ds, 'Rows') and hasattr(ds, 'Columns'):
                    dimensions.append((ds.Rows, ds.Columns))

                if hasattr(ds, 'PixelSpacing'):
                    spacings.append(ds.PixelSpacing)
            except:
                warnings.append(f"Failed to read metadata from {file_path.name}")
                quality_score *= 0.9

        # Check dimension consistency
        if dimensions:
            unique_dims = set(dimensions)
            if len(unique_dims) > 1:
                warnings.append(f"Inconsistent dimensions: {unique_dims}")
                quality_score *= 0.8

        return {
            'valid': quality_score > 0.5,
            'confidence': quality_score,
            'warnings': warnings,
            'slice_count': slice_count,
            'dimensions': dimensions[0] if dimensions else (0, 0)
        }

    def _detect_corruption(self, dicom_files: List[Path]) -> Dict[str, Any]:
        """Detect data corruption"""
        warnings = []
        corruption_indicators = 0
        total_checks = 0

        # Sample files for corruption checks
        sample_size = min(10, len(dicom_files))
        sample_files = dicom_files[::max(1, len(dicom_files)//sample_size)]

        for file_path in sample_files:
            total_checks += 1
            try:
                ds = pydicom.dcmread(file_path, force=True)

                # Check for missing critical tags (PixelSpacing is optional for some formats)
                critical_tags = ['Modality', 'Rows', 'Columns']
                optional_tags = ['PixelSpacing']

                missing_critical = [tag for tag in critical_tags if not hasattr(ds, tag)]
                missing_optional = [tag for tag in optional_tags if not hasattr(ds, tag)]

                if missing_critical:
                    corruption_indicators += 0.8  # Critical tags missing
                    warnings.append(f"Missing critical tags in {file_path.name}: {missing_critical}")
                elif missing_optional:
                    corruption_indicators += 0.1  # Optional tags missing (less severe)
                    warnings.append(f"Missing optional tags in {file_path.name}: {missing_optional}")

                # Check pixel data integrity
                if hasattr(ds, 'pixel_array'):
                    pixel_data = ds.pixel_array
                    if np.all(pixel_data == 0) or np.all(pixel_data == pixel_data.max()):
                        corruption_indicators += 1
                        warnings.append(f"Suspicious pixel data in {file_path.name}")

            except Exception as e:
                corruption_indicators += 1
                warnings.append(f"Corruption detected in {file_path.name}: {str(e)}")

        corruption_score = corruption_indicators / max(total_checks, 1)
        is_valid = corruption_score < 0.3

        return {
            'valid': is_valid,
            'confidence': 1.0 - corruption_score,
            'corruption_score': corruption_score,
            'warnings': warnings
        }

    def _detect_modifications(self, ds: pydicom.Dataset, dicom_files: List[Path]) -> Dict[str, Any]:
        """Detect artificial modifications or suspicious patterns"""
        warnings = []
        suspicion_score = 0.0

        # Check manufacturer information
        manufacturer = getattr(ds, 'Manufacturer', '').lower()
        if re.search(self.suspicious_patterns['manufacturer_mismatch'], manufacturer):
            warnings.append(f"Suspicious manufacturer: {manufacturer}")
            suspicion_score += 0.4

        # Check pixel spacing for unrealistic values
        if hasattr(ds, 'PixelSpacing'):
            spacing = float(ds.PixelSpacing[0])
            min_spacing, max_spacing = self.suspicious_patterns['unusual_spacing']
            if spacing < min_spacing or spacing > max_spacing:
                warnings.append(f"Unusual pixel spacing: {spacing}mm")
                suspicion_score += 0.3

        # Check for artificial slice patterns
        slice_positions = []
        sample_files = dicom_files[::max(1, len(dicom_files)//20)]  # Sample every ~20th file

        for file_path in sample_files:
            try:
                sample_ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                if hasattr(sample_ds, 'SliceLocation'):
                    slice_positions.append(float(sample_ds.SliceLocation))
            except:
                continue

        if len(slice_positions) > 3:
            # Check for perfectly regular spacing (suspicious for real data)
            diffs = np.diff(sorted(slice_positions))
            if len(set(np.round(diffs, 3))) == 1:  # All differences identical to 3 decimal places
                warnings.append("Perfectly regular slice spacing detected")
                suspicion_score += 0.2

        is_valid = suspicion_score < 0.5

        return {
            'valid': is_valid,
            'confidence': 1.0 - suspicion_score,
            'suspicion_score': suspicion_score,
            'warnings': warnings
        }

    def _aggregate_validation_results(self, ds: pydicom.Dataset, dicom_files: List[Path],
                                    modality_check: Dict, anatomy_check: Dict,
                                    quality_check: Dict, corruption_check: Dict,
                                    modification_check: Dict) -> ValidationReport:
        """Aggregate all validation results"""

        all_warnings = []
        all_warnings.extend(modality_check['warnings'])
        all_warnings.extend(anatomy_check['warnings'])
        all_warnings.extend(quality_check['warnings'])
        all_warnings.extend(corruption_check['warnings'])
        all_warnings.extend(modification_check['warnings'])

        # Determine overall result
        if not modality_check['valid']:
            result = ValidationResult.INVALID_MODALITY
            confidence = modality_check['confidence']
        elif not anatomy_check['valid']:
            result = ValidationResult.INVALID_ANATOMY
            confidence = anatomy_check['confidence']
        elif not corruption_check['valid']:
            result = ValidationResult.CORRUPTED_DATA
            confidence = corruption_check['confidence']
        elif not modification_check['valid']:
            result = ValidationResult.SUSPICIOUS_MODIFICATION
            confidence = modification_check['confidence']
        elif not quality_check['valid']:
            result = ValidationResult.INCOMPLETE_STUDY
            confidence = quality_check['confidence']
        else:
            result = ValidationResult.VALID_CHEST_CT
            # Calculate weighted confidence
            confidence = (
                modality_check['confidence'] * 0.3 +
                anatomy_check['confidence'] * 0.3 +
                quality_check['confidence'] * 0.2 +
                corruption_check['confidence'] * 0.1 +
                modification_check['confidence'] * 0.1
            )

        # Extract metadata
        pixel_spacing = getattr(ds, 'PixelSpacing', [1.0, 1.0])
        slice_thickness = getattr(ds, 'SliceThickness', 1.0)

        return ValidationReport(
            result=result,
            confidence=confidence,
            details={
                'modality_check': modality_check,
                'anatomy_check': anatomy_check,
                'quality_check': quality_check,
                'corruption_check': corruption_check,
                'modification_check': modification_check
            },
            warnings=all_warnings,
            slice_count=len(dicom_files),
            spatial_dimensions=quality_check['dimensions'],
            spacing=(float(slice_thickness), float(pixel_spacing[0]), float(pixel_spacing[1])),
            modality=modality_check['modality'],
            body_part=anatomy_check['body_part'],
            manufacturer=getattr(ds, 'Manufacturer', 'UNKNOWN')
        )

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
