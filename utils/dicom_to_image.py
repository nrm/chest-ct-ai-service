"""Convert DICOM files to PNG images for visualization"""
import io
import numpy as np
from PIL import Image
import pydicom
from pathlib import Path
from typing import List, Tuple, Optional


def normalize_dicom_image(pixel_array: np.ndarray, window_center: float = 50, window_width: float = 400) -> np.ndarray:
    """
    Apply windowing to DICOM image for better visualization
    Default window settings for chest CT
    """
    min_value = window_center - window_width / 2
    max_value = window_center + window_width / 2
    
    # Clip values
    windowed = np.clip(pixel_array, min_value, max_value)
    
    # Normalize to 0-255
    windowed = ((windowed - min_value) / (max_value - min_value) * 255).astype(np.uint8)
    
    return windowed


def dicom_to_png_bytes(dicom_path: Path, window_center: float = 50, window_width: float = 400) -> bytes:
    """
    Convert DICOM file to PNG bytes
    
    Args:
        dicom_path: Path to DICOM file
        window_center: Window center for visualization (HU)
        window_width: Window width for visualization (HU)
    
    Returns:
        PNG image as bytes
    """
    try:
        # Read DICOM
        dcm = pydicom.dcmread(str(dicom_path))
        pixel_array = dcm.pixel_array
        
        # Apply rescale if available
        if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
            pixel_array = pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        
        # Apply windowing
        img_array = normalize_dicom_image(pixel_array, window_center, window_width)
        
        # Convert to PIL Image
        img = Image.fromarray(img_array, mode='L')
        
        # Convert to PNG bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return img_bytes.getvalue()
        
    except Exception as e:
        print(f"Error converting DICOM to PNG: {e}")
        # Return a blank image
        blank = Image.new('L', (512, 512), color=0)
        img_bytes = io.BytesIO()
        blank.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes.getvalue()


def get_dicom_files_sorted(directory: Path) -> List[Path]:
    """
    Get sorted list of DICOM files by instance number
    
    Args:
        directory: Directory containing DICOM files
    
    Returns:
        Sorted list of DICOM file paths
    """
    dicom_files = []
    
    for file_path in directory.iterdir():
        if file_path.is_file():
            try:
                dcm = pydicom.dcmread(str(file_path), stop_before_pixels=True)
                instance_num = int(dcm.InstanceNumber) if hasattr(dcm, 'InstanceNumber') else 0
                dicom_files.append((instance_num, file_path))
            except:
                continue
    
    # Sort by instance number
    dicom_files.sort(key=lambda x: x[0])
    
    return [path for _, path in dicom_files]


def get_slice_with_bounding_box(
    dicom_path: Path,
    bbox: Optional[Tuple[float, float, float, float, float, float]] = None,
    window_center: float = 50,
    window_width: float = 400
) -> bytes:
    """
    Get DICOM slice as PNG with optional bounding box overlay
    
    Args:
        dicom_path: Path to DICOM file
        bbox: Bounding box as (x_min, x_max, y_min, y_max, z_min, z_max)
        window_center: Window center for visualization
        window_width: Window width for visualization
    
    Returns:
        PNG image with bounding box as bytes
    """
    try:
        # Read DICOM
        dcm = pydicom.dcmread(str(dicom_path))
        pixel_array = dcm.pixel_array
        
        # Apply rescale
        if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
            pixel_array = pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        
        # Apply windowing
        img_array = normalize_dicom_image(pixel_array, window_center, window_width)
        
        # Convert to RGB for bounding box overlay
        img = Image.fromarray(img_array, mode='L').convert('RGB')
        
        # Draw bounding box if provided
        if bbox:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            x_min, x_max, y_min, y_max, z_min, z_max = bbox
            
            # Draw rectangle
            draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=2)
        
        # Convert to PNG bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return img_bytes.getvalue()
        
    except Exception as e:
        print(f"Error creating slice with bounding box: {e}")
        blank = Image.new('RGB', (512, 512), color=(0, 0, 0))
        img_bytes = io.BytesIO()
        blank.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes.getvalue()

