"""LLM-based tampering detection"""

from .gemini_vision import GeminiTamperingDetector
from .gemini_ocr import GeminiOCRAnalyzer

__all__ = ['GeminiTamperingDetector', 'GeminiOCRAnalyzer']
