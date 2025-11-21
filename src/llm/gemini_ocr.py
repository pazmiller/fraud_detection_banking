"""
Gemini OCR Fraud Analysis - Analyze extracted OCR data for transaction fraud
"""
import google.generativeai as genai
import json
import os
from typing import Dict
from dotenv import load_dotenv

load_dotenv()


class GeminiOCRAnalyzer:
    """Use Gemini to analyze OCR-extracted bank statement data for fraud"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        print("[Gemini OCR Analyzer Initialized - Using Gemini 2.5 Pro]")
    
    def analyze_transactions(self, ocr_json_path: str) -> Dict:
        """
        Analyze OCR-extracted transaction data for fraud patterns
        
        Args:
            ocr_json_path: Path to OCR output JSON file
            
        Returns:
            Dict with fraud analysis results
        """
        try:
            # Load OCR data
            with open(ocr_json_path, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
            
            full_text = ocr_data.get('full_text', '')
            
            # Craft specialized prompt for transaction fraud analysis
            prompt = f"""You are a Compliance Officer responsible for validating bank statement transacttions for fraud detection.

IMPORTANT Guidelines:
1. Focus on SIGNIFICANT anomalies that clearly indicate fraud
2. Do NOT flag small calculation discrepancies as fraud unless there's a clear pattern
3. Consider legitimate banking operations (fees, interest, currency conversion)

========== Bank Statement TEXT ==========
{full_text}

Focus on:

1. **Major Transaction Anomalies**:
   - Duplicate large transactions (same amount, same merchant, same day)
   - Impossible transaction sequences (e.g., balance going negative when overdraft not allowed)
   - Clearly fabricated transaction descriptions

2. **Significant Numerical Issues**:
   - Balance calculation errors >2%
   - Numerical incorrect patterns

3. **Obvious Formatting Issues**:
   - Inconsistent date formats within the statement
   - Clearly altered text patterns

DO NOT flag as fraud:
- Minor rounding differences
- Normal banking fees or charges
- Standard interest calculations
- Currency conversion

Provide your analysis in this exact format:
FRAUD_DETECTED: [YES/NO]
CONFIDENCE: [0-100]%
RISK_LEVEL: [LOW/MEDIUM/HIGH/CRITICAL]

SUSPICIOUS_TRANSACTIONS:
- [List ONLY clearly fraudulent transactions, or "None detected"]

PATTERNS_DETECTED:
- [List unusual patterns, or "No unusual patterns"]

RECOMMENDATION: [ACCEPT/MANUAL_REVIEW/REJECT]

Reason: [Provide reason if Fraud Detected: YES]
"""
            config = { "temperature": 0.1, }
            response = self.model.generate_content(
                prompt,
                generation_config=config
                )
            analysis_text = response.text
            
            # Parse response
            result = self._parse_response(analysis_text, ocr_json_path)
            return result
            
        except Exception as e:
            return {
                'ocr_file': ocr_json_path,
                'error': str(e),
                'fraud_detected': True,
                'confidence': 0.0,
                'risk_level': 'CRITICAL',
                'suspicious_transactions': [],
                'key_findings': [f'Analysis failed: {str(e)}'],
                'patterns_detected': [],
                'recommendation': 'REJECT',
                'raw_response': None
            }
    
    def _parse_response(self, text: str, ocr_file: str) -> Dict:
        """Parse Gemini's response into structured format"""
        lines = text.strip().split('\n')
        
        result = {
            'ocr_file': ocr_file,
            'fraud_detected': False,
            'confidence': 0.0,
            'risk_level': 'LOW',
            'suspicious_transactions': [],
            'key_findings': [],
            'patterns_detected': [],
            'recommendation': 'ACCEPT',
            'explanation': '',
            'raw_response': text
        }
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('FRAUD_DETECTED:'):
                result['fraud_detected'] = 'YES' in line.upper()
            
            elif line.startswith('CONFIDENCE:'):
                try:
                    conf_str = line.split(':')[1].strip().replace('%', '')
                    result['confidence'] = float(conf_str) / 100.0
                except:
                    result['confidence'] = 0.5
            
            elif line.startswith('RISK_LEVEL:'):
                level = line.split(':')[1].strip().upper()
                if level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
                    result['risk_level'] = level
            
            elif line.startswith('SUSPICIOUS_TRANSACTIONS:'):
                current_section = 'transactions'
            
            elif line.startswith('KEY_FINDINGS:'):
                current_section = 'findings'
            
            elif line.startswith('PATTERNS_DETECTED:'):
                current_section = 'patterns'
            
            elif line.startswith('RECOMMENDATION:'):
                rec = line.split(':')[1].strip().upper()
                if rec in ['ACCEPT', 'MANUAL_REVIEW', 'REJECT']:
                    result['recommendation'] = rec
                current_section = None
            
            elif line.startswith('EXPLANATION:'):
                current_section = 'explanation'
            
            elif line.startswith('-') and current_section:
                item = line[1:].strip()
                if item and item.lower() not in ['none', 'none detected', 'no unusual patterns', 'no fraud indicators detected']:
                    if current_section == 'transactions':
                        result['suspicious_transactions'].append(item)
                    elif current_section == 'findings':
                        result['key_findings'].append(item)
                    elif current_section == 'patterns':
                        result['patterns_detected'].append(item)
            
            elif current_section == 'explanation' and line and not line.startswith(('FRAUD_', 'CONFIDENCE:', 'RISK_', 'SUSPICIOUS_', 'KEY_', 'PATTERNS_', 'RECOMMENDATION:')):
                result['explanation'] += line + ' '
        
        result['explanation'] = result['explanation'].strip()
        
        # Safety checks to prevent false positives
        # 1. If confidence is very low, don't recommend REJECT
        if result['confidence'] < 0.3 and result['recommendation'] == 'REJECT':
            result['recommendation'] = 'MANUAL_REVIEW'
            result['risk_level'] = 'MEDIUM' if result['risk_level'] == 'CRITICAL' else result['risk_level']
        
        # 2. If no findings, force ACCEPT
        if (not result['suspicious_transactions'] and 
            not result['key_findings'] and 
            not result['patterns_detected']):
            result['fraud_detected'] = False
            result['recommendation'] = 'ACCEPT'
            result['risk_level'] = 'LOW'
        
        # 3. If fraud_detected is False but recommendation is REJECT, fix inconsistency
        if not result['fraud_detected'] and result['recommendation'] == 'REJECT':
            result['recommendation'] = 'ACCEPT'
        
        return result


def demo():
    """Demo usage"""
    analyzer = GeminiOCRAnalyzer()
    
    # Path to OCR output JSON
    ocr_json_path = "./src/ocr_test/ocr_output.json"
    
    if not os.path.exists(ocr_json_path):
        print(f"❌ OCR file not found: {ocr_json_path}")
        print("   Run new_ocr.py first to generate OCR data")
        return
    
    print("\n" + "="*80)
    print("GEMINI TRANSACTION FRAUD ANALYSIS")
    print("="*80)
    
    result = analyzer.analyze_transactions(ocr_json_path)
    
    print(f"\nOCR File: {result['ocr_file']}")
    print(f"Fraud Detected: {result['fraud_detected']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Recommendation: {result['recommendation']}")
    
    if result['suspicious_transactions']:
        print("\n🚨 Suspicious Transactions:")
        for trans in result['suspicious_transactions']:
            print(f"  • {trans}")
    
    if result['key_findings']:
        print("\n🔍 Key Findings:")
        for finding in result['key_findings']:
            print(f"  • {finding}")
    
    if result['patterns_detected']:
        print("\n📊 Patterns Detected:")
        for pattern in result['patterns_detected']:
            print(f"  • {pattern}")
    
    if result['explanation']:
        print(f"\n💡 Explanation:")
        print(f"  {result['explanation']}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    demo()
