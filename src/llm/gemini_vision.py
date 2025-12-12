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

class GeminiTamperingDetector:
    """Use OpenAI GPT-5 Nano Vision to detect tampering in bank statements"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Set it in environment or pass as parameter.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_of_choice)
        # print("[Gemini Vision Initialized]")
    
    def analyze_tampering(self, image_path: str) -> Dict:
        """
        Analyze bank statement for tampering using Gemini Vision
        
        Args:
            image_path: Path to bank statement image or PDF
            
        Returns:
            Dict with tampering analysis results
        """
        try:
            # === GEMINI (COMMENTED OUT) ===
            if image_path.lower().endswith('.pdf'):
                uploaded_file = genai.upload_file(image_path)
                file_input = uploaded_file
            else:
                file_input = Image.open(image_path)

            # Determine image type
            if image_path.lower().endswith('.png'):
                media_type = "image/png"
            elif image_path.lower().endswith('.pdf'):
                media_type = "image/png" 
            else:
                media_type = "image/jpeg"
            
            prompt = """
            You are a Compliance Officer responsible for validating bank statements. Your goal is to identify DEFINITIVE evidence of document tampering while ignoring artifacts caused by scanning, printing, or standard PDF generation.

                    Real documents often have minor rendering imperfections. Only flag an issue if it is blatant and unexplainable.

                    First, determine the document type:
                    1. Digital Native: Created directly by software.
                    2. Scanned/Photo: A picture of a physical paper (noise, rotation, blur allowed).
                    ---
                    #ANALYSIS CRITERIA
                    1. Visual Consistency
                    -If the background "noise" or texture suddenly disappears behind specific numbers (indicating a digital patch).
                    -Look at the empty space SURROUNDING the transaction numbers. Does the texture/noise pattern suddenly.
                    become "flat" or "white" behind a specific number, while the rest of the page has paper grain or digital noise?
                    -Look for out of place smudges, ink, irregular colour or drawing that should not appear on financial statements.
                    -Look for indicators of tampering and eidting such as Adobe
                    2. Alignment & Layout
                    -If a specific number clearly floats outside its column grid while neighbors are aligned.
                    3. Artifacts & Compression**
                    - "Ghosting" or "halos" that appear ONLY around the transaction amount but nowhere else.
                    #EXCLUSION LIST
                    -Global blurriness.
                    -Math.
                    -Standard variation in font weight (Bold/Normal) used for emphasis.
                    -Imperfect letter spacing in PDF generation.
                    -Column text that is left-aligned vs. right-aligned numbers (standard accounting format).
                    -Different fonts used for Headers vs. Body vs. Footers.

                    ### RESPONSE FORMAT

                    Provide your analysis in this exact format:

                    TAMPERING_DETECTED: [YES/NO]
                    CONFIDENCE: [0-100]%
                    RISK_LEVEL: [LOW/MEDIUM/HIGH]
                    Type: [Digital Native / Scanned]

                    FINDINGS:
                    - [If none, state "No meaningful anomalies detected."]

                    RECOMMENDATION: [ACCEPT/MANUAL_REVIEW/REJECT]
"""
            config = { "temperature": 0.1, }
            response = self.model.generate_content(
                [file_input, prompt],
                generation_config=config
            )
            analysis_text = response.text
            
            # Results
            result = self._parse_response(analysis_text, image_path)
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
            line = line.strip()
            
            if line.startswith('TAMPERING_DETECTED:'):
                result['tampering_detected'] = 'YES' in line.upper()
            
            elif line.startswith('CONFIDENCE:'):
                try:
                    conf_str = line.split(':')[1].strip().replace('%', '')
                    result['confidence'] = float(conf_str) / 100.0
                except:
                    result['confidence'] = 0.5
            
            elif line.startswith('RISK_LEVEL:'):
                level = line.split(':')[1].strip().upper()
                if level in ['LOW', 'MEDIUM', 'HIGH']:
                    result['risk_level'] = level
            
            elif line.startswith('FINDINGS:'):
                current_section = 'findings'
            
            elif line.startswith('RECOMMENDATION:'):
                rec = line.split(':')[1].strip().upper()
                if rec in ['ACCEPT', 'MANUAL_REVIEW', 'REJECT']:
                    result['recommendation'] = rec
                current_section = None
            
            elif current_section == 'findings' and line.startswith('-'):
                finding = line[1:].strip()
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
    """Demo usage"""
    api_key = "YOUR_GEMINI_API_KEY"
    detector = GeminiTamperingDetector(api_key=api_key)
    image_path = "../clip/statements/ocbc_bank_statement.jpg"
    result = detector.analyze_tampering(image_path)
    
    print("\n" + "="*80)
    print("GEMINI TAMPERING ANALYSIS")
    print("="*80)
    print(f"Image: {result['image_path']}")
    print(f"Tampering Detected: {result['tampering_detected']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Recommendation: {result['recommendation']}")
    print(f"Findings: {result['findings']}")
    
    if result['findings']:
        print("\nFindings:")
        for finding in result['findings']:
            print(f"  • {finding}")
    
    print("\nRaw Response:")
    print(result['raw_response'])
    print("="*80)


if __name__ == "__main__":
    demo()
