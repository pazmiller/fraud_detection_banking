"""
Page Splitter for Financial Documents - Layer 1
Identifies and extracts pages containing Income Statement, Balance Sheet, and Cash Flow
"""
import google.generativeai as genai
import os
import tempfile
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
model_of_choice = 'gemini-2.5-pro'  # Use flash for faster page identification


class GeminiPageSplitter:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Set it in environment or pass as parameter.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_of_choice)
        # print("[Gemini Page Splitter Initialized]")
    
    def identify_financial_pages(self, file_path: str) -> Dict:
        """
        Identify which pages contain Income Statement, Balance Sheet, and Cash Flow
        
        Args:
            file_path: Path to PDF or image file
            
        Returns:
            Dict with page identification results
        """
        try:
            if file_path.lower().endswith('.pdf'):
                uploaded_file = genai.upload_file(file_path)
                file_input = uploaded_file
            else:
                from PIL import Image
                file_input = Image.open(file_path)
            
            prompt = """
You are a financial document analyst. Your task is to identify the location of financial statements in this document.

<TASK>
Scan through the ENTIRE document and identify the PHYSICAL PAGE NUMBERS (not printed page numbers) where these 3 types of financial statements are located:

1. Income (Profit & Loss): Shows revenue, expenses, net income/profit
2. Balance Sheet: Shows assets, liabilities, shareholders' equity  
3. Cash Flow Statement: Shows operating, investing, financing cash flows

<RULES>
- Count pages starting from 1 (first physical page = 1)
- If a statement spans multiple pages, list all pages (e.g., "3-5" or "3,4,5")
- If a statement appears multiple times (e.g., different periods), list all occurrences
- If not found, write "Not Found"
- The printed page number may differ from the physical page number (e.g., printed "Page 1" might be physical page 5)

<RESPONSE FORMAT>
Respond in this exact format:

INCOME_PAGES: [page number(s) or "Not Found"]
BALANCE_SHEET_PAGES: [page number(s) or "Not Found"]
CASHFLOW_PAGES: [page number(s) or "Not Found"]
ALL_FINANCIAL_PAGES: [comma-separated list of all unique page numbers containing any financial statement]
"""
            
            config = {"temperature": 0.1}
            response = self.model.generate_content(
                [file_input, prompt],
                generation_config=config
            )
            
            result = self._parse_response(response.text, file_path)
            print(f"[Page Splitter] Found pages - Income: {result['income_pages']}, Balance: {result['balance_sheet_pages']}, CashFlow: {result['cashflow_pages']}")
            
            return result
            
        except Exception as e:
            return {
                'file_path': file_path,
                'error': str(e),
                'total_pages': 0,
                'income_pages': 'Not Found',
                'balance_sheet_pages': 'Not Found',
                'cashflow_pages': 'Not Found',
                'all_financial_pages': [],
                'raw_response': None
            }
    
    def _parse_response(self, text: str, file_path: str) -> Dict:
        """Parse the response into structured format"""
        lines = text.strip().split('\n')
        
        result = {
            'file_path': file_path,
            'total_pages': 0,
            'income_pages': 'Not Found',
            'balance_sheet_pages': 'Not Found',
            'cashflow_pages': 'Not Found',
            'all_financial_pages': [],
            'summary': '',
            'raw_response': text
        }
        
        for line in lines:
            clean_line = line.strip().replace('**', '').replace('*', '').strip()
            upper_line = clean_line.upper()
            
            if 'INCOME_PAGES' in upper_line and ':' in clean_line:
                result['income_pages'] = clean_line.split(':', 1)[1].strip()
            
            elif 'BALANCE_SHEET_PAGES' in upper_line and ':' in clean_line:
                result['balance_sheet_pages'] = clean_line.split(':', 1)[1].strip()
            
            elif 'CASHFLOW_PAGES' in upper_line and ':' in clean_line:
                result['cashflow_pages'] = clean_line.split(':', 1)[1].strip()
            
            elif 'ALL_FINANCIAL_PAGES' in upper_line and ':' in clean_line:
                pages_str = clean_line.split(':', 1)[1].strip()
                result['all_financial_pages'] = self._parse_page_list(pages_str)
            
        
        if not result['all_financial_pages']:
            all_pages = set()
            for key in ['income_pages', 'balance_sheet_pages', 'cashflow_pages']:
                pages = self._parse_page_list(result[key])
                all_pages.update(pages)
            result['all_financial_pages'] = sorted(list(all_pages))
        
        return result
    
    def _parse_page_list(self, pages_str: str) -> List[int]:
        """Parse page string like '3-5' or '3,4,5' or '3, 5-7' into list of integers"""
        if not pages_str or pages_str.lower() == 'not found':
            return []
        
        pages = set()
        # Split by comma first
        parts = pages_str.replace(' ', '').split(',')
        
        for part in parts:
            if '-' in part:
                # Handle range like '3-5'
                try:
                    start, end = part.split('-')
                    for p in range(int(start), int(end) + 1):
                        pages.add(p)
                except:
                    pass
            else:
                # Single page
                try:
                    pages.add(int(part))
                except:
                    pass
        
        return sorted(list(pages))
    
    def extract_pages_from_pdf(self, pdf_path: str, page_numbers: List[int], output_path: str = None, save_to_folder: bool = True) -> Optional[str]:
        """
        Extract specific pages from a PDF file
        
        Args:
            pdf_path: Path to source PDF
            page_numbers: List of page numbers to extract (1-indexed)
            output_path: Optional output path, if None uses document_split folder or temp file
            save_to_folder: If True, save to document_split folder; if False, use temp file
            
        Returns:
            Path to extracted PDF or None if failed
        """
        try:
            import fitz  
            
            if not page_numbers:
                return None
            
            src_doc = fitz.open(pdf_path)
            new_doc = fitz.open()
            
            for page_num in sorted(page_numbers):
                if 1 <= page_num <= len(src_doc):
                    new_doc.insert_pdf(src_doc, from_page=page_num-1, to_page=page_num-1)
            
            if output_path is None:
                if save_to_folder:
                    # Save to document_split folder
                    split_folder = Path(__file__).parent.parent.parent / "document_split"
                    split_folder.mkdir(exist_ok=True)
                    
                    # Create filename based on original file
                    original_name = Path(pdf_path).stem
                    output_path = str(split_folder / f"{original_name}_financial_pages.pdf")
                else:
                    output_path = tempfile.mktemp(suffix='_financial_pages.pdf')
            
            new_doc.save(output_path)
            new_doc.close()
            src_doc.close()
            
            # print(f"[Page Splitter] Extracted {len(page_numbers)} pages to {output_path}")
            return output_path
            
        except ImportError:
            print("[Page Splitter] PyMuPDF not installed. Install with: pip install pymupdf")
            return None
        except Exception as e:
            print(f"[Page Splitter] Error extracting pages: {e}")
            return None


def demo():
    """Demo usage"""
    splitter = GeminiPageSplitter()
    
    # Test with a sample file
    test_file = "./dataset_other_documents/DLEC_medium_115.pdf"
    
    if os.path.exists(test_file):
        result = splitter.identify_financial_pages(test_file)
        
        print("\n" + "="*80)
        print("页面截取结果")
        print("="*80)
        print(f"File: {result['file_path']}")
        print(f"\nFinancial Statement Pages:")
        print(f"  Income Statement: {result['income_pages']}")
        print(f"  Balance Sheet: {result['balance_sheet_pages']}")
        print(f"  Cash Flow: {result['cashflow_pages']}")
        print(f"\nAll Financial Pages: {result['all_financial_pages']}")
        print("="*80)
    else:
        print(f"Test file not found: {test_file}")


if __name__ == "__main__":
    demo()
