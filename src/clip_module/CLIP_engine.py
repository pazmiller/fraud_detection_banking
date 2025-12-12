import torch
import clip
from PIL import Image
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import tempfile
import fitz  # PyMuPDF for PDF conversion


@dataclass
class CLIPConfig:
    """CLIP Model Config"""
    model_name: str = "ViT-B/16"  # ViT-B/32, ViT-B/16, ViT-L/14, from fastest to most accurate
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    confidence_threshold: float = 0.3


class BankStatementCLIPEngine:
    """
    Bank Statement CLIP Detection Engine
    Simplified: Only verifies if document is a bank statement
    """
    
    def __init__(self, config: Optional[CLIPConfig] = None):
        self.config = config or CLIPConfig()
        self.device = self.config.device
        # print(f"[Loading CLIP Model]{self.config.model_name} to {self.device}...")
        self.model, self.preprocess = clip.load(self.config.model_name, device=self.device)
        # print("[Loading Successful]")
            
        self._setup_text_prompts()
    
    def _setup_text_prompts(self):
        """Setting up text prompts for document type classification"""
        self.document_type_prompts = [
            "a bank statement",
            "receipt",
            "fake document",
            "handwritten"
        ]
        self._encode_text_prompts()

    def _encode_text_prompts(self):
        # print("[Encoding text prompts...]")
        text_tokens = clip.tokenize(self.document_type_prompts).to(self.device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(text_tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        # print("[Text prompts encoding successful]")
        
    def _convert_pdf_to_image(self, pdf_path: str) -> List[str]:
        """Convert PDF first page to temporary PNG image"""
        doc = fitz.open(pdf_path)
        temp_dir = Path(tempfile.gettempdir()) / "fraud_detection_temp"
        temp_dir.mkdir(exist_ok=True)
        
        page = doc[0]  # First page only
        pix = page.get_pixmap()  
        temp_path = temp_dir / f"{Path(pdf_path).stem}_page1.png"
        pix.save(str(temp_path))
        
        doc.close()
        return str(temp_path)
    
    def classify_document(self, image_path: str) -> Dict:
        """
        Classify document type
        
        Returns:
            Dict with classification results
        """
        # Auto-convert PDF
        if image_path.lower().endswith('.pdf'):
            image_path = self._convert_pdf_to_image(image_path)
        
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarities
        similarities = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        similarities = similarities.cpu().numpy()[0]
        
        top_idx = similarities.argmax()
        confidence = float(similarities[top_idx])
        predicted_type = self.document_type_prompts[top_idx]
        
        is_bank_statement = (predicted_type == "a bank statement")
        
        result = {
            'is_bank_statement': is_bank_statement,
            'confidence': confidence,
            'predicted_type': predicted_type,
            'all_scores': {self.document_type_prompts[i]: float(similarities[i]) for i in range(len(similarities))}
        }
        
        return result
