import torch
import clip
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class CLIPConfig:
    """CLIP Model Config"""
    model_name: str = "ViT-B/32"  # ViT-B/32, ViT-B/16, ViT-L/14, from fastest to most accurate
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    confidence_threshold: float = 0.3  # Higher -> Reject


class BankStatementCLIPEngine:
    """
    Bank Statement CLIP Detection Engine
    Features:
    1. Verify if the uploaded file is a bank statement
    2. Identify the type of bank statement and the bank
    3. Detect anomalies and suspicious areas
    4. Verify document quality
    """
    
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
        
        self._encode_text_prompts()
    
    def _encode_text_prompts(self):
        print("[Encoding text prompts...]")
        self.encoded_prompts = {}
        
        prompt_categories = {
            'document_type': self.document_type_prompts,
            'quality': self.quality_prompts,
            'anomaly': self.anomaly_prompts
        }
        
        for category, prompts in prompt_categories.items():
            text_tokens = clip.tokenize(prompts).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            self.encoded_prompts[category] = text_features
        
        print("[Text prompts encoding successful]")
    
    def verify_bank_statement(self, image_path: str) -> Dict:
        """
        Is it a bank statement or not ah
                
        Args:
            image_path: 
            
        Returns:
            Dict containing verification results
        """
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # How similar to a bank statement should be
        text_features = self.encoded_prompts['document_type']
        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        similarities = similarities.cpu().numpy()[0]
        
        top_idx = similarities.argmax()
        confidence = float(similarities[top_idx])
        predicted_type = self.document_type_prompts[top_idx]
        
        # Determine if it is a bank statement
        is_bank_statement = any(term in predicted_type.lower() for term in ['bank statement', 'account statement', 'financial statement'])
        
        result = {
            'is_bank_statement': is_bank_statement,
            'confidence': confidence,
            'predicted_type': predicted_type,
            'all_scores': {self.document_type_prompts[i]: float(similarities[i]) for i in range(len(similarities))},
            'threshold_passed': confidence >= self.config.confidence_threshold
        }
        
        return result
    
    def assess_document_quality(self, image_path: str) -> Dict:
        """
        Assess document quality
        
        Args:
            image_path: 
            
        Returns:
            Dict containing quality assessment results
        """
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        text_features = self.encoded_prompts['quality']
        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        similarities = similarities.cpu().numpy()[0]
        
        # Calculate quality score (average score of high-quality prompts)
        high_quality_indices = [0, 1, 5]
        low_quality_indices = [2, 3, 4]   
        
        quality_score = float(np.mean([similarities[i] for i in high_quality_indices]))
        
        result = {
            'quality_score': quality_score,
            'is_high_quality': quality_score > 0.5,
            'quality_assessment': self.quality_prompts[similarities.argmax()],
            'all_scores': {self.quality_prompts[i]: float(similarities[i]) for i in range(len(similarities))}
        }
        
        return result
    
    def detect_tampering_signs(self, image_path: str) -> Dict:
        """
        Detect tampering signs using CLIP
        
        Args:
            image_path: 
            
        Returns:
            Dict containing tampering detection results
        """
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        text_features = self.encoded_prompts['anomaly']
        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        similarities = similarities.cpu().numpy()[0]
        
        # Tampering risk score
        tampered_indices = [0, 1, 2, 3, 4]  
        authentic_indices = [5, 6]           
        
        tampering_score = float(np.mean([similarities[i] for i in tampered_indices]))
        authentic_score = float(np.mean([similarities[i] for i in authentic_indices]))
        
        result = {
            'tampering_risk_score': tampering_score,
            'authenticity_score': authentic_score,
            'is_suspicious': tampering_score > authentic_score,
            'most_likely_issue': self.anomaly_prompts[similarities.argmax()],
            'all_scores': {self.anomaly_prompts[i]: float(similarities[i]) for i in range(len(similarities))}
        }
        
        return result
    
    def analyze_regional_features(self, image_path: str, regions: List[Tuple[int, int, int, int]]) -> List[Dict]:
        """
        Analyze features in specific regions of the image
        Used to detect local tampering or anomalies
        
        Args:
            image_path: 
            regions: List of regions, each region is (x, y, width, height)
            
        Returns:
            List of analysis results for each region
        """
        image = Image.open(image_path).convert('RGB')
        results = []
        
        for i, (x, y, w, h) in enumerate(regions):
            region_image = image.crop((x, y, x + w, y + h))
            region_input = self.preprocess(region_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                region_features = self.model.encode_image(region_input)
                region_features /= region_features.norm(dim=-1, keepdim=True)
            
            text_features = self.encoded_prompts['anomaly']
            similarities = (100.0 * region_features @ text_features.T).softmax(dim=-1)
            similarities = similarities.cpu().numpy()[0]
            
            tampered_indices = [0, 1, 2, 3, 4]
            authentic_indices = [5, 6]
            
            tampering_score = float(np.mean([similarities[i] for i in tampered_indices]))
            authentic_score = float(np.mean([similarities[i] for i in authentic_indices]))
            
            results.append({
                'region_index': i,
                'coordinates': (x, y, w, h),
                'tampering_score': tampering_score,
                'authenticity_score': authentic_score,
                'is_suspicious': tampering_score > authentic_score
            })
        
        return results
    
    def comprehensive_analysis(self, image_path: str) -> Dict:
        """
        Calculating Risk
        
        Args:
            image_path: 
            
        Returns:
            Dict containing all analysis results
        """
        print(f"\nAnalysing {image_path} using the might of OpenAI CLIP ...")
        
        results = {
            'image_path': image_path,
            'verification': self.verify_bank_statement(image_path),
            'quality_assessment': self.assess_document_quality(image_path),
            'tampering_detection': self.detect_tampering_signs(image_path)
        }
        
        risk_score = 0.0
        risk_factors = []
        
        if not results['verification']['is_bank_statement']:
            risk_score += 1.0
            risk_factors.append("Not a Bank Statement?!")
        
        if not results['quality_assessment']['is_high_quality']:
            risk_score += 0.2
            risk_factors.append("Low Document Quality [could be due to low resolution, blurriness, or device capture]")
        
        if results['tampering_detection']['is_suspicious']:
            risk_score += 0.5
            risk_factors.append("Tampering Signs Detected")
        
        results['overall_risk'] = {
            'risk_score': min(risk_score, 1.0),
            'risk_level': 'HIGH' if risk_score > 0.6 else 'MEDIUM' if risk_score > 0.3 else 'LOW',
            'risk_factors': risk_factors,
            'recommendation': 'REJECT' if risk_score > 0.6 else 'MANUAL_REVIEW' if risk_score > 0.3 else 'ACCEPT'
        }
        
        return results
    
    def compare_documents(self, image_path1: str, image_path2: str) -> Dict:
        """
        Compare two documents for similarity
        
        Args:
            image_path1: First document path
            image_path2: Second document path
            
        Returns:
            Similarity analysis results
        """
        image1 = Image.open(image_path1).convert('RGB')
        image2 = Image.open(image_path2).convert('RGB')
        
        image1_input = self.preprocess(image1).unsqueeze(0).to(self.device)
        image2_input = self.preprocess(image2).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features1 = self.model.encode_image(image1_input)
            features2 = self.model.encode_image(image2_input)
            
            features1 /= features1.norm(dim=-1, keepdim=True)
            features2 /= features2.norm(dim=-1, keepdim=True)
            
            similarity = float((features1 @ features2.T).cpu().numpy()[0, 0])
        
        return {
            'similarity_score': similarity,
            'is_duplicate': similarity > 0.95,
            'is_similar': similarity > 0.85,
            'interpretation': (
                'Identical or near-identical documents' if similarity > 0.95
                else 'Highly similar documents' if similarity > 0.85
                else 'Moderately similar documents' if similarity > 0.7
                else 'Different documents'
            )
        }


def demo_clip_engine():

    engine = BankStatementCLIPEngine()
    # Document path
    test_image = "bank_statement_sample.jpg"  
    results = engine.comprehensive_analysis(test_image)
    
    print("\n=== CLIP Conclusion ===")
    print(f"\n1. Base Verification:")
    print(f"   Is it bank statement: {results['verification']['is_bank_statement']}")
    print(f"   Confidence: {results['verification']['confidence']:.2%}")
    print(f"   Predicted type: {results['verification']['predicted_type']}")
    
    print(f"\n2. Quality Assessment:")
    print(f"   Quality score: {results['quality_assessment']['quality_score']:.2%}")
    print(f"   Quality assessment: {results['quality_assessment']['quality_assessment']}")
    
    print(f"\n3. Tampering Detection:")
    print(f"   Tampering risk score: {results['tampering_detection']['tampering_risk_score']:.2%}")
    print(f"   Authenticity score: {results['tampering_detection']['authenticity_score']:.2%}")
    print(f"   Is suspicious: {results['tampering_detection']['is_suspicious']}")
    
    print(f"\n4. Overall Tampering and related Risks Assessment:")
    print(f"   Risk score: {results['overall_risk']['risk_score']:.2%}")
    print(f"   Risk level: {results['overall_risk']['risk_level']}")
    print(f"   Recommendation: {results['overall_risk']['recommendation']}")
    print(f"   Risk factors: {', '.join(results['overall_risk']['risk_factors']) if results['overall_risk']['risk_factors'] else 'None'}")

if __name__ == "__main__":
    demo_clip_engine()
