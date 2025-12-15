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
             <ANALYSIS CRITERIA RULES>
            1. Visual Consistency
            -If the background "noise" or texture suddenly disappears behind specific numbers (indicating a digital patch).
            -Look at the empty space SURROUNDING the transaction numbers. Does the texture/noise pattern suddenly become "flat" or "white" behind a specific number?
            -Look for out of place smudges, ink, highlight, irregular colour or drawing that should not appear on financial statements.
            -Look for indicators of tampering and editing such as Adobe, or oddly shaped text that differs from the rest of the same line
            2. Alignment & Layout
            -If a specific number clearly floats outside its column grid while neighbors are aligned.
            3. Statement
            - Signs of editing, erasing, or unusual text that appear ONLY around the transaction amount but nowhere else.
            
            
            You are a meticulous Compliance Officer responsible for validating financial documents, you are sensitive to tampering signs. Your goal is to:
            Step1. Identify in the document, the location of 3 types of financial statements: Income(Profit Loss), Balance Sheet, and Cash Flow. 
            Remember, identify not the printed page in the file, identify the physical pdf file page use the offset(the printed page number 1 starts at the physical page 5, so the offest is 4) for calculating the physial pdf file page.
            Remember, these 3 types of statements might locate in more than one places.
            Step2. After identifying, focused on the loacted pages from Step1. Scrutinise these pages and identify evidence of document tampering by referring to  <ANALYSIS CRITERIA RULES>. 
            Step3. Look through the document, if the pages number is not right, for example one page missing inbetween, it is a sign of tampering.
            ---
            #PAGE IDENTIFICATION
            Identify which page(s) contain each financial statement:
            - Income (Profit & Loss): Shows revenue, expenses, net income
            - Balance Sheet: Shows assets, liabilities, equity
            - Cash Flow : Shows operating, investing, financing cash flows
            
            If a statement spans multiple pages, list all pages (e.g., "1-2" or "1,2").
            If not found, write "Not Found".
            ---
           
    
            
            <EXCLUSION LIST>
            -Global blurriness.
            -Math.
            -Standard variation in font weight (Bold/Normal) used for emphasis.
            -Different fonts used for Headers vs. Body vs. Footers.

            ### RESPONSE FORMAT

            Provide your analysis in this exact format:

            INCOME_PAGE: [page number(s) or "Not Found"]
            BALANCE_SHEET_PAGE: [page number(s) or "Not Found"]
            CASHFLOW_PAGE: [page number(s) or "Not Found"]
            
            TAMPERING_DETECTED: [YES/NO] If any one of the rule from <ANALYSIS CRITERIA RULES> matched, return YES in TAMPERING_DETECTED.
            CONFIDENCE: [0-100]%
            FINDINGS:
            - [Must list the findings if you think it is suspicious or fradulous]

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
            'income_page': 'Not Found',
            'balance_sheet_page': 'Not Found',
            'cashflow_page': 'Not Found',
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
            
            if 'INCOME_PAGE' in upper_line and ':' in clean_line:
                result['income_page'] = clean_line.split(':', 1)[1].strip()
            
            elif 'BALANCE_SHEET_PAGE' in upper_line and ':' in clean_line:
                result['balance_sheet_page'] = clean_line.split(':', 1)[1].strip()
            
            elif 'CASHFLOW_PAGE' in upper_line and ':' in clean_line:
                result['cashflow_page'] = clean_line.split(':', 1)[1].strip()
            
            elif 'TAMPERING_DETECTED' in upper_line and ':' in clean_line:
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
    """Demo usage"""
    api_key = "YOUR_GEMINI_API_KEY"
    detector = GeminiTamperingDetector(api_key=api_key)
    image_path = "../clip/statements/ocbc_bank_statement.jpg"
    result = detector.analyze_tampering(image_path)
    
    print("\n" + "="*80)
    print("GEMINI FINANCIAL DOCUMENT ANALYSIS")
    print("="*80)
    print(f"Image: {result['image_path']}")
    print(f"\n--- Page Identification ---")
    print(f"Income Statement Page: {result['income_page']}")
    print(f"Balance Sheet Page: {result['balance_sheet_page']}")
    print(f"Cash Flow Page: {result['cashflow_page']}")
    print(f"\n--- Tampering Analysis ---")
    print(f"Tampering Detected: {result['tampering_detected']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Recommendation: {result['recommendation']}")
    
    if result['findings']:
        print("\nFindings:")
        for finding in result['findings']:
            print(f"  • {finding}")
    
    print("\nRaw Response:")
    print(result['raw_response'])
    print("="*80)


if __name__ == "__main__":
    demo()
