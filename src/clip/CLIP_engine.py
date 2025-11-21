import torch
import clip
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import tempfile
import fitz  # PyMuPDF for PDF conversion


@dataclass
class CLIPConfig:
    """CLIP Model Config"""
    model_name: str = "ViT-B/16"  # ViT-B/32, ViT-B/16, ViT-L/14, from fastest to most accurate
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
            "receipt",
            "fake document",
            "handwritten"
        ]
        
        # 2. Quality
        self.quality_prompts = [
            "a high quality document with clear sharp text",
            "a crisp and readable high resolution document",
            "a well-formatted professional document",
            
            "a low quality blurry document with unclear text",
            "a poor resolution pixelated document"
        ]
        
        # 3. Anomaly
        self.anomaly_prompts = [
            "tampering signs",
            "inconsistent text fonts and misaligned text",
            "out of place text",
            "low resolution",
            "clean and consistent text",
            "no visible editing",
            "high resolution",
            "no tampering",
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
        
        
    def _convert_pdf_to_image(self, pdf_path: str) -> List[str]:
        """Convert all PDF pages to temporary PNG images"""
        doc = fitz.open(pdf_path)
        temp_dir = Path(tempfile.gettempdir()) / "fraud_detection_temp"
        temp_dir.mkdir(exist_ok=True)
        
        temp_paths = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap()  
            
            temp_path = temp_dir / f"{Path(pdf_path).stem}_page{page_num+1}.png"
            pix.save(str(temp_path))
            temp_paths.append(str(temp_path))
        
        doc.close()
        print(f"    PDF converted: {len(temp_paths)} pages → PNG")
        return temp_paths
    
    def _process_image(self, image_path: str) -> torch.Tensor:
        """
            single document (auto-convert PDF to first page)
        """
        # Auto-convert PDF to image (use first page for single image processing)
        if image_path.lower().endswith('.pdf'):
            image_paths = self._convert_pdf_to_image(image_path)
            image_path = image_paths[0]  # Use first page
        
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def _process_images_batch(self, image_paths: List[str]) -> torch.Tensor:
        """
        Batch process images (auto-convert PDFs to all pages)
        
        Args:
            image_paths: List of image/PDF paths
            
        Returns:
            torch.Tensor: Batch image features
        """
        images = []
        for path in image_paths:
            if path.lower().endswith('.pdf'): # Cause CLIP cannot process PDF files
                pdf_pages = self._convert_pdf_to_image(path)
                for page_path in pdf_pages:
                    image = Image.open(page_path).convert('RGB')
                    image_input = self.preprocess(image)
                    images.append(image_input)
            else:
                image = Image.open(path).convert('RGB')
                image_input = self.preprocess(image)
                images.append(image_input)
        
        batch_input = torch.stack(images).to(self.device)
        
        with torch.no_grad():
            batch_features = self.model.encode_image(batch_input)
            batch_features /= batch_features.norm(dim=-1, keepdim=True)
        
        return batch_features
    
    def _get_similarities(self, image_features: torch.Tensor, category: str) -> np.ndarray:
        text_features = self.encoded_prompts[category]
        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return similarities.cpu().numpy()[0]
    
    
    def verify_bank_statement(self, image_path: str) -> Dict:
        """
        Is it a bank statement or not ah
            
        Returns:
            Dict containing verification results
        """
        # Auto-convert PDF
        if image_path.lower().endswith('.pdf'):
            pdf_pages = self._convert_pdf_to_image(image_path)
            image_path = pdf_pages[0]  # Use first page
        
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
    
    def assess_document_quality(self, image_path: str, debug: bool = False) -> Dict:
        """
        Args:
            debug: 
            
        Returns:
            Dict containing quality assessment results
        """
        image_features = self._process_image(image_path)  # Already handles PDF conversion
        similarities = self._get_similarities(image_features, 'quality')
        
        # Print detailed probabilities (for debugging)
        if debug:
            print("\n" + "="*80)
            print("Quality Prompts Probability Distribution:")
            print("="*80)
            for i, prompt in enumerate(self.quality_prompts):
                prob = similarities[i]
                group = "✅ High Quality" if i in [0, 1, 2] else "❌ Low Quality"
                bar = "█" * int(prob * 50)
                print(f"[{i}] {group} | {prob:.4f} ({prob*100:.2f}%) {bar}")
                print(f"    Prompt: \"{prompt}\"")
            print("="*80 + "\n")
        
        high_quality_indices = [0, 1, 2]
        low_quality_indices = [3, 4]
        
        # Calculate sum instead of average
        quality_score = float(np.sum([similarities[i] for i in high_quality_indices]))
        poor_quality_score = float(np.sum([similarities[i] for i in low_quality_indices]))
        
        if debug:
            print(f"Calculation Results:")
            print(f"  High quality score (sum of [0,1,2]): {quality_score:.4f} ({quality_score*100:.2f}%)")
            print(f"  Low quality score (sum of [3,4]): {poor_quality_score:.4f} ({poor_quality_score*100:.2f}%)")
        
        if quality_score > 0.7:  
            quality_level = "high"
        elif quality_score > 0.5: 
            quality_level = "medium"
        else:
            quality_level = "low"
        
        if debug:
            print(f"  Final level: {quality_level.upper()}")
            print()
            
        result = {
            'quality_score': quality_score,
            'poor_quality_score': poor_quality_score,
            'quality_level': quality_level,
            'quality_assessment': self.quality_prompts[similarities.argmax()],
            'all_scores': {self.quality_prompts[i]: float(similarities[i]) for i in range(len(similarities))}
        }
        return result

    
    def detect_tampering_signs(self, image_path: str) -> Dict:
        """
        Detect tampering signs using CLIP
            
        Returns:
            Dict containing tampering detection results
        """
        image_features = self._process_image(image_path)  # Already handles PDF conversion
        similarities = self._get_similarities(image_features, 'anomaly')
        
        tampered_indices = [0, 1, 2, 3, 4]
        authentic_indices = [5, 6]
        tampering_score = float(np.mean([similarities[i] for i in tampered_indices]))
        authentic_score = float(np.mean([similarities[i] for i in authentic_indices]))
        
        suspicion_level = "none"
        if tampering_score > authentic_score:
            if tampering_score > 0.4:
                suspicion_level = "very_suspicious"
            elif tampering_score > 0.2:
                suspicion_level = "medium_suspicious"
            else:
                suspicion_level = "mildly_suspicious"
        result = {
            'tampering_risk_score': tampering_score,
            'authenticity_score': authentic_score,
            'suspicion_level': suspicion_level,
            'most_likely_issue': self.anomaly_prompts[similarities.argmax()],
            'all_scores': {self.anomaly_prompts[i]: float(similarities[i]) for i in range(len(similarities))}
        }
        
        return result
    
    
    def comprehensive_analysis(self, image_path: str, debug: bool = False) -> Dict:
        """
        Calculating Risk
        
        Args:
            image_path: 
            debug: Test only ah
        
        Returns:
            Dict containing all analysis results
        """
        print(f"\nAnalysing {image_path} using the might of OpenAI CLIP ...")
        
        results = {
            'image_path': image_path,
            'verification': self.verify_bank_statement(image_path),
            'quality_assessment': self.assess_document_quality(image_path, debug=debug),
            'tampering_detection': self.detect_tampering_signs(image_path)
        }
        
        ''' For individual testing only: Cancel comment out to test'''
        # print(f"\n  1. Document Verification:")
        verification = results['verification']
        # if verification['is_bank_statement']:
        #     print(f"     Confirmed as bank statement")
        # else:
        #     print(f"     ❌ Not a bank statement")
        # print(f"     Confidence: {verification['confidence']:.1%}")
        # print(f"     Predicted Type: {verification['predicted_type']}")
        
        # print(f"\n  2. Quality Assessment:")
        quality = results['quality_assessment']
        quality_level = quality['quality_level']
        # if quality_level == "high":
        #     print(f"     High Quality Document")
        # elif quality_level == "medium":
        #     print(f"     Medium Quality Document")
        # else:
        #     print(f"     ❌ Low Quality Document")
        # print(f"     Quality Level: {quality_level.upper()}")
        # print(f"     Quality Score: {quality['quality_score']:.1%}")
        
        # print(f"\n  3. Tampering Detection:")
        tampering = results['tampering_detection']
        tampering_score = tampering['tampering_risk_score']
        authentic_score = tampering['authenticity_score']
        
        if tampering_score > 0.5:
            print(f"     Highly Suspicious")
        elif tampering_score > 0.35:
            print(f"     Moderately Suspicious")
        elif tampering_score > 0.25:
            print(f"     Slightly Suspicious")
        else:
            print(f"     ✅ No Obvious Anomalies")
        
        print(f"     Tampering Risk: {tampering_score:.1%}")
        print(f"     Authenticity Score: {authentic_score:.1%}")
        print(f"     Most Likely Issue: {tampering['most_likely_issue']}")
        
        # Calculate risk score
        risk_score = 0.0
        risk_factors = []
        
        if not results['verification']['is_bank_statement']:
            risk_score += 1.0
            risk_factors.append("Not a Bank Statement?!")
            
        quality_risk_map = {"high": 0.0, "medium": 0.25, "low": 0.5}
        quality_risk = quality_risk_map.get(quality_level, 0.5)
        risk_score += quality_risk
        if quality_level in ["medium", "low"]:
            risk_factors.append(f"Quality issues detected (level: {quality_level})")

        tampering_risk = tampering_score * 1
        risk_score += tampering_risk
        if tampering_score > 0.25:
            risk_factors.append(f"Tampering signs detected (score: {tampering_score:.2f})")
        
        results['overall_risk'] = {
            'risk_score': min(risk_score, 1.0),
            'risk_level': 'HIGH' if risk_score > 0.6 else 'MEDIUM' if risk_score > 0.3 else 'LOW',
            'risk_factors': risk_factors,
            'recommendation': 'REJECT' if risk_score > 0.6 else 'MANUAL_REVIEW' if risk_score > 0.3 else 'ACCEPT'
        }
        
        
        print(f"\n[CLIP Overall Risk]:")
        print(f"     Risk Score: {results['overall_risk']['risk_score']:.1%}")
        print(f"     Risk Level: {results['overall_risk']['risk_level']}")
        print(f"     Recommendation: {results['overall_risk']['recommendation']}")
        if risk_factors:
            print(f"     Risk Factors:")
            for factor in risk_factors:
                print(f"       • {factor}")
        
        return results
    
    def batch_analysis(self, image_paths: List[str], show_progress: bool = True, debug: bool = False) -> List[Dict]:
        """       
        Args:
            image_paths: 
            show_progress: 
            debug: 
        
        Returns:
            List[Dict]: Analysis results
        """
        if not image_paths:
            return []
        
        results = []
        total = len(image_paths)
        
        if show_progress:
            print(f"\n{'='*80}")
            print(f"Starting batch analysis of {total} files...")
            if debug:
                print("Debug mode enabled - detailed probability distribution will be shown")
            print(f"{'='*80}\n")
        
        for idx, image_path in enumerate(image_paths, 1):
            if show_progress:
                print(f"[{idx}/{total}] Analyzing: {image_path}")
            
            try:
                result = self.comprehensive_analysis(image_path, debug=debug)
                results.append(result)
                
                if show_progress:
                    risk_level = result['overall_risk']['risk_level']
                    print(f"    Risk level: {risk_level}\n")
                    
            except Exception as e:
                error_result = {
                    'image_path': image_path,
                    'error': str(e),
                    'overall_risk': {
                        'risk_level': 'ERROR',
                        'risk_score': 1.0,
                        'recommendation': 'REJECT',
                        'risk_factors': [f'Processing failed: {str(e)}']
                    }
                }
                results.append(error_result)
                
                if show_progress:
                    print(f"    ❌ Error: {e}\n")
        
        if show_progress:
            print(f"{'='*80}")
            print(f"Batch analysis completed!")
            print(f"{'='*80}\n")
            self._print_batch_summary(results)
        
        return results
    
    def _print_batch_summary(self, results: List[Dict]):
        total = len(results)
        accept = sum(1 for r in results if r.get('overall_risk', {}).get('recommendation') == 'ACCEPT')
        review = sum(1 for r in results if r.get('overall_risk', {}).get('recommendation') == 'MANUAL_REVIEW')
        reject = sum(1 for r in results if r.get('overall_risk', {}).get('recommendation') == 'REJECT')
        errors = sum(1 for r in results if 'error' in r)
        
        print("\nBatch Analysis Summary:")
        print(f"   Total: {total} files")
        print(f"   ✅ Accept: {accept} ({accept/total*100:.1f}%)")
        print(f"   Manual Review: {review} ({review/total*100:.1f}%)")
        print(f"   ❌ Reject: {reject} ({reject/total*100:.1f}%)")
        if errors > 0:
            print(f"   Error: {errors} ({errors/total*100:.1f}%)")
        print()
