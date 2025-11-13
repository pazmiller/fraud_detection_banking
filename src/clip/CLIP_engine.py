import torch
import clip
from typing import Optional
from dataclasses import dataclass


@dataclass
class CLIPConfig:
    """CLIP Model Config"""
    model_name: str = "ViT-B/32"  
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    confidence_threshold: float = 0.3  # Higher -> Reject


class BankStatementCLIPEngine:
    
    def __init__(self, config: Optional[CLIPConfig] = None):
        self.config = config or CLIPConfig()
        self.device = self.config.device
        print(f"[Loading CLIP Model]{self.config.model_name} to {self.device}...Might take sometime to download if first time")
        self.model, self.preprocess = clip.load(self.config.model_name, device=self.device)
        print("[Loading Successful]")
            
        self._setup_text_prompts()
    
    def _setup_text_prompts(self):
        """Setting up text prompts for CLIP, Most important part"""
        # 1. Bank Statement Keywords
        self.document_type_prompts = [
            "a bank statement",
            "a bank account statement", 
            "a financial statement",
            "a credit card statement",
            "an invoice",
            "a receipt",
            "a random document",
            "a screenshot",
            "a photo",
            "a handwritten note",
            "a scanned document",
            "a printed text document"
        ]
        
        # 2. Quality
        self.quality_prompts = [
            "a high quality document",
            "a clear and readable document",
            "a blurry document",
            "a low resolution document",
            "a damaged or corrupted document",
            "a well-formatted document"
        ]
        
        # 3. Anomaly
        self.anomaly_prompts = [
            "a document with tampering signs",
            "a document with visible editing",
            "a document with inconsistent fonts",
            "a document with misaligned text",
            "a document with out of place text",
            "a photoshopped document",
            "an authentic original document",
            "a clean unmodified document"
        ]
        
  