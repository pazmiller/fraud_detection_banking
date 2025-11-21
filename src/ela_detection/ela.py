from PIL import Image, ImageChops, ImageEnhance
import io
import os
import numpy as np


def ela_detect(image_path: str, threshold: float = 30.0, quality: int = 90) -> dict:
    """
    
    Args:
        image_path: 
        threshold: ELA threshold
        quality: JPEG re-compression quality
        
    Returns:
        dict with keys: avg_difference, max_difference, is_suspicious, verdict
    """
    original = Image.open(image_path).convert('RGB')
    
    buffer = io.BytesIO()
    original.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer).convert('RGB')
    
    diff = ImageChops.difference(original, resaved)
    diff_array = np.array(diff)
    
    avg_diff = float(diff_array.mean())
    max_diff = float(diff_array.max())
    is_suspicious = max_diff > threshold
    
    return {
        'avg_difference': avg_diff,
        'max_difference': max_diff,
        'is_suspicious': is_suspicious,
        'verdict': 'SUSPICIOUS' if is_suspicious else 'CLEAN'
    }


def perform_ela(image_path, output_path, quality=90, enhance_scale=15.0):

    if not os.path.exists(image_path):
        print(f"Not found in {image_path}")
        return

    try:
        original_image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Cannot open {e}")
        return

    temp_buffer = io.BytesIO()
    original_image.save(temp_buffer, format='JPEG', quality=quality)
    
    temp_buffer.seek(0) 
    resaved_image = Image.open(temp_buffer).convert('RGB')

    ela_image = ImageChops.difference(original_image, resaved_image)

    enhancer = ImageEnhance.Brightness(ela_image)
    ela_image_enhanced = enhancer.enhance(enhance_scale)

    ela_image_enhanced.save(output_path)
    print(f"ELA ✔ saved in: {output_path}")

INPUT_FILE = 'ocbc_bank_statement.jpg' 
OUTPUT_FILE = 'ela_result.png' # The saved shoutld be PNG to avoid losses

perform_ela(INPUT_FILE, OUTPUT_FILE)