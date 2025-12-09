from paddleocr import PaddleOCR
import os
import json
import re
from datetime import datetime
from typing import List, Dict, Optional
import threading

# Global OCR instances pool for multi-threading (？ for now 4 threads)
_ocr_pool = []
_ocr_pool_initialized = False
_ocr_pool_lock = threading.Lock()
_parallel_thread_counter = 0
_parallel_thread_counter_lock = threading.Lock()
_ocr_semaphore = None
_ocr_parallel_count = 4  # Match ThreadPoolExecutor max_workers

def initialize_ocr_pool():
    """Initialize pool of OCR engines for multi-threading"""
    global _ocr_pool, _ocr_pool_initialized, _ocr_semaphore, _ocr_parallel_count
    
    with _ocr_pool_lock:
        if _ocr_pool_initialized:
            return
        
        print(f"[OCR Pool] Initializing {_ocr_parallel_count} independent OCR engines...")
        
        for i in range(_ocr_parallel_count):
            try:
                ocr_engine = PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    show_log=False,
                    det_db_box_thresh=0.2,  # Lower threshold to detect small symbols like +, -, =
                    det_db_unclip_ratio=1.8,  # Expand text detection boxes slightly
                    rec_batch_num=6,
                    use_space_char=True,
                    use_gpu=False,
                    det_limit_side_len=960,
                    det_limit_type='max'
                )
                _ocr_pool.append(ocr_engine)
                print(f"[OCR Pool] Engine {i+1}/{_ocr_parallel_count} initialized")
            except Exception as e:
                print(f"[OCR Pool ERROR] Failed to initialize engine {i+1}: {e}")
                raise
        
        _ocr_semaphore = threading.Semaphore(value=_ocr_parallel_count)
        _ocr_pool_initialized = True
        print(f"[OCR Pool] All {_ocr_parallel_count} engines ready!")


def get_ocr_engine():
    """Get OCR engine from pool (thread-safe allocation)"""
    global _parallel_thread_counter, _ocr_pool, _ocr_semaphore

    if not _ocr_pool_initialized:
        initialize_ocr_pool()
    
    # Thread-safe counter increment and modulo assignment
    current_thread_counter = 0
    try:
        _parallel_thread_counter_lock.acquire()
        _parallel_thread_counter += 1
        current_thread_counter = _parallel_thread_counter % _ocr_parallel_count
    finally:
        _parallel_thread_counter_lock.release()
    
    selected_ocr = _ocr_pool[current_thread_counter]
    return selected_ocr


def extract_text_from_image(image_path: str) -> dict:
    """  
    Extract text and parse transaction patterns from bank statement
    
    Returns:
        dict: {
            'image_path': str,
            'timestamp': str,
            'total_lines': int,
            'full_text': str,
            'text_lines': [{'text': str, 'confidence': float, 'bbox': list}, ...],
            'transactions': [{'date': str, 'description': str, 'amount': float, 'type': str}, ...],
            'summary': {
                'total_transactions': float,
                'total_withdrawals': float,
                'total_deposits': float,
                'date_range': {'start': str, 'end': str},
                'balance': {'opening': float, 'closing': float}
            }
        }
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    try:
        ocr = get_ocr_engine()     
        result = ocr.ocr(image_path, cls=True)
        
    except IndexError as e:
        import traceback
        print(f"[DEBUG] IndexError details:")
        print(traceback.format_exc())
        raise RuntimeError(f"PaddleOCR IndexError: {e}")
    except Exception as e:
        import traceback
        print(f"[DEBUG] Exception details:")
        print(traceback.format_exc())
        raise RuntimeError(f"PaddleOCR execution failed: {e}")

    if result is None:
        raise RuntimeError("PaddleOCR returned None - possible model loading failure")
    
    if not isinstance(result, list):
        raise RuntimeError(f"PaddleOCR returned unexpected type: {type(result)}")
    
    if len(result) == 0:
        raise RuntimeError("PaddleOCR returned empty list - no text detected")
    
    # Extract and structure data
    text_lines = []
    full_text = []
    
    if result[0]:
        for idx, line in enumerate(result[0]):
            try:
                # Validate line structure
                if not isinstance(line, (list, tuple)) or len(line) < 2:
                    print(f"Warning: Skipping malformed line {idx}: {line}")
                    continue
                
                bbox = line[0]
                text_tuple = line[1]
                
                if not isinstance(text_tuple, (list, tuple)) or len(text_tuple) < 2:
                    print(f"Warning: Skipping line {idx} with bad text_tuple: {text_tuple}")
                    continue
                
                text = text_tuple[0]
                confidence = text_tuple[1]
                
                line_data = {
                    "line_number": idx + 1,
                    "text": text,
                    "confidence": round(confidence, 4),
                    "bbox": [[int(coord) for coord in point] for point in bbox]
                }
                text_lines.append(line_data)
                full_text.append(text)
            except Exception as e:
                print(f"Warning: Error processing line {idx}: {e}")
                continue
    
    full_text_str = "\n".join(full_text)
    transactions = _parse_transactions(full_text_str)
    summary = _calculate_summary(transactions, full_text_str)
    
    return {
        "image_path": image_path,
        "timestamp": datetime.now().isoformat(),
        "total_lines": len(text_lines),
        "full_text": full_text_str,
        "text_lines": text_lines,
        "transactions": transactions,
        "summary": summary
    }


def _parse_transactions(text: str) -> List[Dict]:
    """
    Parse transaction patterns from OCR text with adaptive column detection
    Handles different bank statement formats automatically
    
    Returns:
        List of transactions with date, description, amount, type
    """
    transactions = []
    lines = text.split('\n')
    
    format_info = _detect_statement_format(lines)
    
    date_patterns = [
        r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',  # DD/MM/YYYY or MM/DD/YYYY
        r'\b(\d{2}\s+[A-Z]{3}\s+\d{4})\b',        # DD MMM YYYY
        r'\b([A-Z]{3}\s+\d{1,2},?\s+\d{4})\b',    # MMM DD, YYYY
        r'\b(\d{4}-\d{2}-\d{2})\b'                # YYYY-MM-DD
    ]
    
    amount_pattern = r'[\$€£¥]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
    
    for i, line in enumerate(lines):
        # Skip empty, header, or very short lines (1 char cases)
        if not line.strip() or len(line.strip()) < 2:
            continue
        
        if i < format_info.get('header_line_index', 0) + 2:
            continue
        
        date_match = None
        for pattern in date_patterns:
            date_match = re.search(pattern, line, re.IGNORECASE)
            if date_match:
                break
        
        if not date_match:
            continue
        
        amounts = re.findall(amount_pattern, line)
        clean_amounts = []
        for amt in amounts:
            try:
                clean_amt = float(amt.replace(',', '').replace(' ', ''))
                clean_amounts.append(clean_amt)
            except:
                continue
        
        if not clean_amounts:
            continue
        
        trans_info = _classify_transaction(
            line, 
            clean_amounts, 
            format_info
        )
        
        date_str = date_match.group(0)
        
        desc_start = date_match.end()
        desc_end = line.find(str(clean_amounts[0]))
        description = line[desc_start:desc_end].strip() if desc_end > desc_start else line[desc_start:].strip()
        
        transactions.append({
            'date': date_str,
            'description': description[:100],
            'amount': trans_info['amount'],
            'type': trans_info['type'],
            'balance': trans_info.get('balance'),
            'line_number': i + 1
        })
    
    return transactions


def _detect_statement_format(lines: List[str]) -> Dict:
    """
    Auto-detect bank statement format by analyzing header patterns
    
    Returns format info:
    {
        'format_type': 'dual_column' | 'single_column' | 'in_out',
        'column_names': {...},
        'header_line_index': int
    }
    """
    format_info = {
        'format_type': 'unknown',
        'column_names': {},
        'header_line_index': -1,
        'has_balance_column': False
    }
    
    # Keywords for different column types
    withdrawal_keywords = ['withdraw', 'debit', 'out', 'payment', 'dr']
    deposit_keywords = ['deposit', 'credit', 'in', 'receipt', 'cr']
    amount_keywords = ['amount', 'value', 'sum']
    balance_keywords = ['balance', 'running', 'current']
    date_keywords = ['date', 'transaction date', 'posting date']
    desc_keywords = ['description', 'particulars', 'details', 'narration']
    
    # Scan first 20 lines for headers
    for i, line in enumerate(lines[:20]):
        line_lower = line.lower()
        
        # Check if this line contains multiple column headers
        keyword_matches = {
            'withdrawal': any(kw in line_lower for kw in withdrawal_keywords),
            'deposit': any(kw in line_lower for kw in deposit_keywords),
            'amount': any(kw in line_lower for kw in amount_keywords),
            'balance': any(kw in line_lower for kw in balance_keywords),
            'date': any(kw in line_lower for kw in date_keywords),
            'description': any(kw in line_lower for kw in desc_keywords)
        }
        
        match_count = sum(keyword_matches.values())
        
        if match_count >= 3:
            format_info['header_line_index'] = i
            
            # Determine format type
            if keyword_matches['withdrawal'] and keyword_matches['deposit']:
                format_info['format_type'] = 'dual_column'  # Separate debit/credit columns
            elif keyword_matches['amount'] and not keyword_matches['withdrawal']:
                format_info['format_type'] = 'single_column'  # One amount column with +/-
            elif 'in' in line_lower and 'out' in line_lower:
                format_info['format_type'] = 'in_out'
            
            format_info['has_balance_column'] = keyword_matches['balance']
            format_info['column_names'] = keyword_matches
            
            break
    
    return format_info


def _classify_transaction(line: str, amounts: List[float], format_info: Dict) -> Dict:
    """
    Classify transaction type based on detected format and line content
    
    Returns:
    {
        'amount': float,
        'type': 'withdrawal' | 'deposit',
        'balance': float | None
    }
    """
    result = {
        'amount': amounts[0] if amounts else 0.0,
        'type': 'unknown',
        'balance': None
    }
    
    line_lower = line.lower()
    format_type = format_info.get('format_type', 'unknown')
    
    # Method 1: Check for explicit debit/credit indicators
    debit_indicators = ['dr', 'debit', 'withdrawal', '-', 'out', 'payment']
    credit_indicators = ['cr', 'credit', 'deposit', '+', 'in', 'receipt']
    
    has_debit = any(ind in line_lower for ind in debit_indicators)
    has_credit = any(ind in line_lower for ind in credit_indicators)
    
    if has_debit and not has_credit:
        result['type'] = 'withdrawal'
    elif has_credit and not has_debit:
        result['type'] = 'deposit'
    
    # Method 2: Use format-specific logic
    elif format_type == 'dual_column':
        # Two amount columns: withdrawal | deposit
        if len(amounts) >= 2:
            # Typically: withdrawal comes before deposit in column order
            if amounts[0] > 0 and amounts[1] == 0:
                result['type'] = 'withdrawal'
                result['amount'] = amounts[0]
            elif amounts[1] > 0 and amounts[0] == 0:
                result['type'] = 'deposit'
                result['amount'] = amounts[1]
        elif len(amounts) == 1:
            result['type'] = _infer_from_keywords(line_lower)
    
    elif format_type == 'single_column':
        # One amount column with sign or type indicator
        if len(amounts) >= 1:
            result['amount'] = amounts[0]
            result['type'] = _infer_from_keywords(line_lower)
    
    elif format_type == 'in_out':
        # IN/OUT format
        if 'out' in line_lower:
            result['type'] = 'withdrawal'
        elif 'in' in line_lower:
            result['type'] = 'deposit'
    
    # Method 3: Infer from transaction description keywords
    if result['type'] == 'unknown':
        result['type'] = _infer_from_keywords(line_lower)
    
    # Extract balance if present (usually the last amount)
    if format_info.get('has_balance_column') and len(amounts) >= 2:
        result['balance'] = amounts[-1]
    
    return result


def _infer_from_keywords(text: str) -> str:
    """
    Infer transaction type from description keywords
    """
    withdrawal_keywords = [
        'payment', 'transfer', 'withdraw', 'atm', 'purchase', 'fee', 'charge',
        'bill', 'debit', 'pos', 'online', 'card', 'cheque'
    ]
    deposit_keywords = [
        'deposit', 'salary', 'interest', 'refund', 'credit', 'income',
        'transfer in', 'received', 'giro'
    ]
    
    withdrawal_score = sum(1 for kw in withdrawal_keywords if kw in text)
    deposit_score = sum(1 for kw in deposit_keywords if kw in text)
    
    if withdrawal_score > deposit_score:
        return 'withdrawal'
    elif deposit_score > withdrawal_score:
        return 'deposit'
    else:
        return 'unknown'


def _calculate_summary(transactions: List[Dict], full_text: str) -> Dict:
    """
    Calculate transaction summary statistics
    """
    summary = {
        'total_transactions': len(transactions),
        'total_withdrawals': 0.0,
        'total_deposits': 0.0,
        'withdrawal_count': 0,
        'deposit_count': 0,
        'date_range': {'start': None, 'end': None},
        'balance': {'opening': None, 'closing': None},
        'unusual_patterns': []
    }
    
    if not transactions:
        return summary
    
    # Calculate totals
    for trans in transactions:
        if trans['type'] == 'withdrawal':
            summary['total_withdrawals'] += trans['amount']
            summary['withdrawal_count'] += 1
        elif trans['type'] == 'deposit':
            summary['total_deposits'] += trans['amount']
            summary['deposit_count'] += 1
    
    # Extract date range
    dates = [t['date'] for t in transactions if t['date']]
    if dates:
        summary['date_range']['start'] = dates[0]
        summary['date_range']['end'] = dates[-1]
    
    # Try to extract balance information from text
    balance_patterns = [
        r'(?:opening|previous|balance\s+b/f)\s*:?\s*[\$€£¥]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
        r'(?:closing|current|balance\s+c/f)\s*:?\s*[\$€£¥]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
    ]
    
    opening_match = re.search(balance_patterns[0], full_text, re.IGNORECASE)
    closing_match = re.search(balance_patterns[1], full_text, re.IGNORECASE)
    
    if opening_match:
        try:
            summary['balance']['opening'] = float(opening_match.group(1).replace(',', ''))
        except:
            pass
    
    if closing_match:
        try:
            summary['balance']['closing'] = float(closing_match.group(1).replace(',', ''))
        except:
            pass
    
    # Detect unusual patterns
    if transactions:
        amounts = [t['amount'] for t in transactions]
        
        # Check for duplicate amounts
        from collections import Counter
        amount_counts = Counter(amounts)
        duplicates = [amt for amt, count in amount_counts.items() if count > 2]
        if duplicates:
            summary['unusual_patterns'].append(f'Duplicate amounts detected: {duplicates[:3]}')
        
        # Check for suspiciously round numbers
        round_amounts = [amt for amt in amounts if amt % 100 == 0 and amt >= 1000]
        if len(round_amounts) > len(amounts) * 0.5:
            summary['unusual_patterns'].append('High frequency of round-number transactions')
    
    return summary


''' This is for testing only:'''
# def main():
#     """Demo usage - run this file directly to test OCR"""
#     image_path = "./dataset/ocbc_bank_statement.jpg"
    
#     if not os.path.exists(image_path):
#         print(f"❌ Image not found: {image_path}")
#         return
    
#     print(f"Extracting text from: {image_path}")
    
#     # Extract text
#     extracted_data = extract_text_from_image(image_path)
    
#     print("\n" + "="*80)
#     print("OCR Results:")
#     print("="*80)
    
#     for line_data in extracted_data["text_lines"]:
#         print(f"\n[{line_data['line_number']}] {line_data['text']}")
#         print(f"    Confidence: {line_data['confidence']:.2%}")
    
#     # Save to JSON
#     output_file = "./src/ocr_test/ocr_output.json"
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(extracted_data, f, indent=2, ensure_ascii=False)
    
#     print("\n" + "="*80)
#     print(f"Total lines detected: {extracted_data['total_lines']}")
#     print(f"✅ OCR data saved to: {output_file}")
#     print("="*80)

# if __name__ == "__main__":
#     main()
