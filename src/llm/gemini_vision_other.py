"""
Vision API for Bank Statement Tampering Detection
"""
import google.generativeai as genai

from PIL import Image
import os
from typing import Dict
from dotenv import load_dotenv
  # or 'google/gemma-3-27b-it:free'

load_dotenv()
model_of_choice = 'gemini-2.5-pro'

class GeminiTamperingDetectorOther:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Set it in environment or pass as parameter.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_of_choice)
        print("[Gemini Vision Initialized]")
    
    def analyze_tampering(self, image_path: str) -> Dict:
        """
        Analyse bank statement for tampering using Gemini Vision
        
        Args:
            image_path: Path to bank statement image or PDF
            
        Returns:
            Dict with tampering analysis results
        """
        try:
            if image_path.lower().endswith('.pdf'):
                uploaded_file = genai.upload_file(image_path)
                file_input = uploaded_file
            else:
                file_input = Image.open(image_path)

            if image_path.lower().endswith('.png'):
                media_type = "image/png"
            elif image_path.lower().endswith('.pdf'):
                media_type = "image/png" 
            else:
                media_type = "image/jpeg"
            
            prompt = """
You are a meticulous Compliance Officer responsible for validating financial documents.
Your <task> is to scrutinize this document carefully and identify ANY evidence of document tampering based on the <ANALYSIS CRITERIA RULES> below.

<ANALYSIS CRITERIA RULES>
1. **Visual Consistency**
   - If the background "noise" or texture suddenly disappears behind specific numbers (indicating a digital patch)
   - Look at the empty space SURROUNDING the transaction numbers. Does the texture/noise pattern suddenly become "flat" or "white" behind a specific number?
   - Look for out of place  ink, highlighter, irregular colour or drawing that should not appear on financial statements
   - Look for indicators of tampering and editing such as Adobe watermarks, or manually written text, or oddly shaped text that differs from the rest of the same line
   - Especially highlighter marks, that highlights a block of text in one specific colour such as yellow

2. **Alignment & Layout**
   - If a specific number clearly floats outside its column grid while neighbors are aligned
   - Misaligned rows or columns that break the document's grid structure

3. **Text Anomalies**
   - Signs of editing, erasing, or unusual text that appear ONLY around the transaction amount but nowhere else
   - Font inconsistencies within the same line or section
   - Characters that appear slightly different in size, weight, or style compared to surrounding text

## EXCLUSION LIST (Do NOT flag these as tampering)
- Global blurriness (affects entire document equally)
- Math errors or calculation mistakes
- Standard variation in font weight (Bold/Normal) used for emphasis
- Different fonts used for Headers vs. Body vs. Footers (intentional design)

## RESPONSE FORMAT
Provide your analysis in this exact format:

TAMPERING_DETECTED: [YES/NO] - If any rule from ANALYSIS CRITERIA RULES matched , return YES
CONFIDENCE: [0-100]%
RISK_LEVEL: [LOW/MEDIUM/HIGH]
FINDINGS:
- [List each specific finding with page number and location if applicable]
- [Be specific about what you observed and why it's suspicious]
- [If highlighter/color marks found, describe the color, location, and what it covers]

RECOMMENDATION: [ACCEPT/MANUAL_REVIEW/REJECT]
"""
            config = { "temperature": 0.1, }
            response = self.model.generate_content(
                [file_input, prompt],
                generation_config=config
            )
            analysis_text = response.text
            print(f"[DEBUG gemini_vision_other] Raw response:\n{analysis_text[:500]}...")
            
            # Results
            result = self._parse_response(analysis_text, image_path)
            print(f"[DEBUG gemini_vision_other] Parsed - tampering: {result['tampering_detected']}, findings: {result['findings']}")
            return result
            
        except Exception as e:
            return {
                'image_path': image_path,
                'error': str(e),
                'tampering_detected': True,
                'confidence': 0.0,
                'risk_level': 'HIGH',
                'findings': [f'Analysis failed: {str(e)}'],
                'recommendation': 'REJECT',
                'raw_response': None
            }
    
    def _parse_response(self, text: str, image_path: str) -> Dict:
        """Parse Gemini's response into structured format"""
        lines = text.strip().split('\n')
        
        result = {
            'image_path': image_path,
            'tampering_detected': False,
            'confidence': 0.0,
            'risk_level': 'LOW',
            'findings': [],
            'recommendation': 'ACCEPT',
            'raw_response': text
        }
        
        current_section = None
        
        for line in lines:
            # Strip markdown formatting and whitespace
            clean_line = line.strip().replace('**', '').replace('*', '').strip()
            upper_line = clean_line.upper()
            
            if 'TAMPERING_DETECTED' in upper_line and ':' in clean_line:
                result['tampering_detected'] = 'YES' in upper_line
            
            elif 'CONFIDENCE' in upper_line and ':' in clean_line:
                try:
                    conf_str = clean_line.split(':')[1].strip().replace('%', '')
                    result['confidence'] = float(conf_str) / 100.0
                except:
                    result['confidence'] = 0.5
            
            elif 'RISK_LEVEL' in upper_line and ':' in clean_line:
                level = clean_line.split(':')[1].strip().upper()
                if level in ['LOW', 'MEDIUM', 'HIGH']:
                    result['risk_level'] = level
            
            elif 'FINDINGS' in upper_line and ':' in clean_line:
                current_section = 'findings'
            
            elif 'RECOMMENDATION' in upper_line and ':' in clean_line:
                rec = clean_line.split(':')[1].strip().upper()
                if rec in ['ACCEPT', 'MANUAL_REVIEW', 'REJECT']:
                    result['recommendation'] = rec
                current_section = None
            
            elif current_section == 'findings' and (clean_line.startswith('-') or clean_line.startswith('•')):
                finding = clean_line.lstrip('-•').strip()
                if finding and finding.lower() not in ['none', 'no tampering signs detected', '[]']:
                    result['findings'].append(finding)
        
        # Safety checks to prevent false positives
        if result['confidence'] < 0.3 and result['recommendation'] == 'REJECT':
            result['recommendation'] = 'MANUAL_REVIEW'
            result['risk_level'] = 'MEDIUM'
        
        if not result['findings'] or all('no tampering' in f.lower() for f in result['findings']):
            result['tampering_detected'] = False
            result['recommendation'] = 'ACCEPT'
            result['risk_level'] = 'LOW'
        
        if not result['tampering_detected'] and result['recommendation'] in ['REJECT', 'MANUAL_REVIEW']:
            result['recommendation'] = 'ACCEPT'
        
        return result


def demo():
    """Demo usage - Test tampering detection with 2-layer approach"""
    import os
    import time
    from pathlib import Path
    from gemini_split_pages_other import GeminiPageSplitter
    
    # Initialize both layers
    splitter = GeminiPageSplitter()
    detector = GeminiTamperingDetectorOther()
    
    # Test files - modify this path to your test file
    test_folder = Path(__file__).parent.parent.parent / "dataset_other_documents"
    
    # Get PDF files for testing - change the number or remove slice to test all
    # [:1] = first file only, [:3] = first 3 files, remove slice = all files
    test_files = list(test_folder.glob("*.pdf"))  # All PDF files
    
    if not test_files:
        print("No PDF files found in dataset_other_documents folder")
        print("Please add test files or modify the path")
        return
    
    print(f"Found {len(test_files)} PDF files to analyze\n")
    
    # Store results for summary
    all_results = []
    
    for idx, image_path in enumerate(test_files, 1):
        print("\n" + "="*80)
        print(f"GEMINI FINANCIAL DOCUMENT ANALYSIS (2-LAYER) [{idx}/{len(test_files)}]")
        print("="*80)
        print(f"File: {image_path.name}")
        
        # Layer 1: Page Identification & Splitting
        print("\n--- Layer 1: Page Identification ---")
        layer1_start = time.time()
        page_result = splitter.identify_financial_pages(str(image_path))
        
        print(f"Income Pages: {page_result['income_pages']}")
        print(f"Balance Sheet Pages: {page_result['balance_sheet_pages']}")
        print(f"Cash Flow Pages: {page_result['cashflow_pages']}")
        print(f"All Financial Pages: {page_result['all_financial_pages']}")
        
        # Extract pages
        split_pdf_path = None
        all_pages = page_result.get('all_financial_pages', [])
        if all_pages:
            split_pdf_path = splitter.extract_pages_from_pdf(str(image_path), all_pages)
            print(f"Extracted {len(all_pages)} pages to temp file")
        
        layer1_time = time.time() - layer1_start
        print(f"Layer 1 Time: {layer1_time:.2f}s")
        
        time.sleep(1.0)  # Rate limit
        
        # Layer 2: Tampering Detection
        print("\n--- Layer 2: Tampering Detection ---")
        analysis_path = split_pdf_path if split_pdf_path else str(image_path)
        print(f"Analyzing: {'Extracted pages' if split_pdf_path else 'Original file'}")
        
        layer2_start = time.time()
        result = detector.analyze_tampering(analysis_path)
        layer2_time = time.time() - layer2_start
        
        print(f"Tampering Detected: {result['tampering_detected']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Recommendation: {result['recommendation']}")
        print(f"Layer 2 Time: {layer2_time:.2f}s")
        
        if result['findings']:
            print("\nFindings:")
            for finding in result['findings']:
                print(f"  • {finding}")
        
        # Note: Split files are now saved to document_split folder for testing
        if split_pdf_path:
            print(f"\nSplit file saved to: {split_pdf_path}")
        
        print(f"\n--- Total Time: {layer1_time + layer2_time:.2f}s ---")
        print("="*80)
        
        # Store result for summary
        all_results.append({
            'filename': image_path.name,
            'tampering_detected': result['tampering_detected'],
            'confidence': result['confidence'],
            'recommendation': result['recommendation']
        })
    
    # Print final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for idx, r in enumerate(all_results, 1):
        tampering = "YES" if r['tampering_detected'] else "NO"
        print(f"{idx}. {r['filename']}, Tampering Detected: {tampering}, Confidence: {r['confidence']:.0%}, Recommendation: {r['recommendation']}")
    print("="*80)


if __name__ == "__main__":
    demo()
