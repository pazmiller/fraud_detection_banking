from .new_ocr import extract_text_from_image, initialize_ocr_pool
from .new_ocr_backup import extract_text_from_image as extract_text_from_image_backup

__all__ = ['extract_text_from_image', 'initialize_ocr_pool', 'extract_text_from_image_backup']