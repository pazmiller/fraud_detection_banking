"""
Bank Statement Fraud Detection - ELA + Gemini Integration
Workflow: Image → Metadata → ELA → Gemini Vision → Final Decision
"""
import warnings
# Suppress Google API Python version warnings
warnings.filterwarnings('ignore', message='.*Python version.*past its end of life.*')
warnings.filterwarnings('ignore', category=FutureWarning, module='google.api_core')

# Fix for Python 3.9: packages_distributions not available
import importlib.metadata
if not hasattr(importlib.metadata, 'packages_distributions'):
    def _packages_distributions():
        return {}
    importlib.metadata.packages_distributions = _packages_distributions

import os
import sys
import json
import time
import random
import threading
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ela_detection import ela_detect
from src.llm.gemini_vision_other import GeminiTamperingDetectorOther
from src.llm.gemini_ocr import GeminiOCRAnalyzer
from src.ocr_test import extract_text_from_image
from src.metadata import extract_metadata, check_tampering_indicators

load_dotenv()

results_dir = Path(__file__).parent.parent / "results_other_documents"
results_dir.mkdir(exist_ok=True)
json_results_store_path = results_dir / "other_documents_results_record_Gemini-2.5-pro.json"

class FraudDetectionSystem:
    """Combined ELA + Metadata + Gemini + OCR + Gemini OCR Analysis (4-Layer Pipeline) [Considering adding 1 more Gemini layer of page extraction]"""
    
    def __init__(self, ela_threshold=30.0, use_gemini=True, use_ocr=True, verbose=True, summary_type='simple'):
        self.ela_threshold = ela_threshold
        self.use_gemini = use_gemini
        self.use_ocr = use_ocr
        self.verbose = verbose
        self.summary_type = summary_type
        self.gemini_vision_time = 0.0
        self.gemini_ocr_time = 0.0
        self.print_lock = threading.Lock()
        
        # Initialise
        if self.use_gemini:
            try:
                self.gemini_detector = GeminiTamperingDetectorOther()
                print("[Gemini Vision Initialized in FraudDetectionSystem]")
            except Exception as e:
                print(f"⚠️  Vision initialization failed: {e}")
                self.use_gemini = False

        if self.use_ocr:
            try:
                self.gemini_ocr_analyzer = GeminiOCRAnalyzer()
                print("[Gemini OCR Initialized in FraudDetectionSystem]")
            except Exception as e:
                print(f"⚠️  OCR Analyzer initialization failed: {e}")
                self.use_ocr = False
    
    def _vprint(self, msg: str):
        """Verbose print-for testing only now"""
        if self.verbose:
            with self.print_lock:
                print(msg)
    
    def batch_analyze(self, folder_path: str) -> list:
        """Batch analysis using ThreadPoolExecutor (Max 4 threads)"""
        start_time = time.time()
        self.gemini_vision_time = 0.0
        self.gemini_ocr_time = 0.0
        
        image_paths = self._get_image_paths(folder_path)
        if not image_paths:
            return []
        
        results = []
        max_workers = 4
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self._analyze_single_safe_capture, img, idx, len(image_paths)): img 
                for idx, img in enumerate(image_paths, 1)
            }
            for future in as_completed(future_to_file):
                img_path = future_to_file[future]
                try:
                    single_result, _ = future.result() 
                    results.append(single_result)
                    
                except Exception as exc:
                    with self.print_lock:
                        print(f'\n❌ {img_path} generated an exception: {exc}')
                    results.append({'image_path': img_path, 'error': str(exc), 'recommendation': 'REJECT'})

        results.sort(key=lambda x: x['image_path'])

        elapsed_time = time.time() - start_time
        if self.summary_type == 'simple':
            self._print_batch_summary_simple(results)
        else:
            self._print_batch_summary(results, elapsed_time, self.gemini_vision_time, self.gemini_ocr_time)
        return results

    def _analyze_single_safe_capture(self, img_path: str, idx: int, total: int):
        """Wrapper: Add random delay and perform analysis"""
        sleep_time = random.uniform(0.5, 1.5)
        time.sleep(sleep_time)
        
        filename = Path(img_path).name
        print(f"[DEBUG] Starting analysis for {filename}...")
        
        try:
            result = self._analyze_single(img_path, idx, total)
            print(f"[DEBUG] Finished analysis for {filename}")
            return result, ""  
            
        except Exception as e:
            import traceback
            print(f"[DEBUG] Exception in {filename}: {e}")
            traceback.print_exc()
            return {'image_path': img_path, 'error': str(e), 'recommendation': 'REJECT'}, ""
    
    def _get_image_paths(self, folder_path: str) -> list:
        folder = Path(folder_path)
        if not folder.exists():
            print(f"❌ Error: Folder '{folder_path}' does not exist")
            print(f"   Please create the folder or check the path")
            return []
        
        paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png','*.pdf']:
            paths.extend(folder.glob(ext))
            paths.extend(folder.glob(ext.upper()))
        return sorted(set(str(p) for p in paths))
    
    def _analyze_single(self, img_path: str, idx: int, total: int) -> dict:
        """Analyze single image (Logic unchanged, output is captured by wrapper)"""
        self._vprint(f"\n[{idx}/{total}] {Path(img_path).name} (Processing...)")
        
        try:
            metadata_result = self._run_metadata(img_path)
            
            # Early Termination: High-risk editing software
            high_risk_software = ['Photoshop', 'iLovePDF', 'Smallpdf', 'Adobe Illustrator']
            detected_software = None
            for flag in metadata_result.get('flags', []):
                for sw in high_risk_software:
                    if sw.lower() in flag.lower():
                        detected_software = sw
                        break
                if detected_software:
                    break
            
            if detected_software:
                self._vprint(f"\n  REJECT: High-risk editing software detected: {detected_software}")
                self._vprint(f"  This document was edited with professional software commonly used for forgery.")
                self._vprint(f"  Skipping further analysis - CONFIRMED FRAUDULENT.")
                result = {
                    'image_path': img_path,
                    'ela': {'skipped': True},
                    'metadata': metadata_result,
                    'gemini': {'skipped': True, 'reason': f'High-risk editing software: {detected_software}'},
                    'ocr': {'skipped': True, 'reason': f'High-risk editing software: {detected_software}'},
                    'gemini_ocr': {'skipped': True, 'reason': f'High-risk editing software: {detected_software}'},
                    'final_risk_score': 1.0,
                    'final_risk_level': 'CRITICAL',
                    'recommendation': 'REJECT',
                    'early_termination': True,
                    'termination_reason': f'Metadata detected high-risk editing software: {detected_software}'
                }
                return result

            ela_result = self._run_ela(img_path)
            
            gemini_result = None
            financial_pages = None
            if self.use_gemini:
                print(f"[DEBUG] Calling Gemini Vision for {Path(img_path).name}...")
                gemini_result = self._run_gemini(img_path)
                print(f"[DEBUG] Gemini Vision result: tampering={gemini_result.get('tampering_detected')}, findings={gemini_result.get('findings')}")
                # Extract page identification from Gemini Vision result
                financial_pages = {
                    'income': gemini_result.get('income_page', 'Not Found'),
                    'balance_sheet': gemini_result.get('balance_sheet_page', 'Not Found'),
                    'cashflow': gemini_result.get('cashflow_page', 'Not Found')
                }
                print(f"[DEBUG] Financial pages: {financial_pages}")
                time.sleep(1.0)
            
            '''OCR Later_'''
            # OCR Running (currently disabled)
            # print(f"[DEBUG] Calling OCR...")
            # ocr_result = self._run_ocr(img_path, financial_pages) if self.use_ocr else None
            # print(f"[DEBUG] OCR result: {ocr_result}")
            
            # print(f"[DEBUG] Calling Gemini OCR...")
            # gemini_ocr_result = self._run_gemini_ocr(ocr_result, Path(img_path).name) if (self.use_ocr and ocr_result) else None
            # print(f"[DEBUG] Gemini OCR result: {gemini_ocr_result}")
            
            # Since OCR is disabled, set results to None
            ocr_result = None
            gemini_ocr_result = None
            
            # Calculate final risk
            print(f"[DEBUG] Calculating risk - ELA: {ela_result.get('is_suspicious')}, Metadata flags: {metadata_result.get('flags')}, time_diff: {metadata_result.get('time_diff_seconds')}")
            final_risk_score, calc_steps = self._calculate_risk(ela_result, metadata_result, gemini_result, gemini_ocr_result)
            print(f"[DEBUG] Risk calculation: score={final_risk_score}, steps={calc_steps}")
            
            # Build result (including fraud recommendation)
            result = {
                'image_path': img_path,
                'ela': ela_result,
                'metadata': metadata_result,
                'gemini': gemini_result,
                'ocr': ocr_result,
                'gemini_ocr': gemini_ocr_result,
                'risk_calculation_steps': calc_steps,
                'final_risk_score': final_risk_score,
                'final_risk_level': 'HIGH' if final_risk_score > 0.6 else 'MEDIUM' if final_risk_score >= 0.3 else 'LOW',
                'recommendation': 'REJECT' if final_risk_score > 0.6 else 'MANUAL_REVIEW' if final_risk_score >= 0.3 else 'ACCEPT'
            }
            
            return result
            
        except Exception as e:
            return {'image_path': img_path, 'error': str(e), 'recommendation': 'REJECT'}
    
    def _run_ela(self, img_path: str) -> dict:
        if img_path.lower().endswith('.pdf') or img_path.lower().endswith('.PDF') or img_path.lower().endswith('.png'):
            return {'verdict': 'SKIPPED', 'is_suspicious': False, 'max_difference': 0.0, 'skipped': True}
        
        result = ela_detect(img_path, threshold=self.ela_threshold)
        result['skipped'] = False
        return result
    
    def _run_gemini(self, img_path: str) -> dict:
        start = time.time()
        result = self.gemini_detector.analyze_tampering(img_path)
        gemini_vision_time = time.time() - start
        self.gemini_vision_time += gemini_vision_time
        result['gemini_vision_time'] = gemini_vision_time
        return result
    
    def _run_metadata(self, img_path: str) -> dict:
        try:
            metadata = extract_metadata(img_path)
            flags = check_tampering_indicators(metadata)
            time_diff = metadata.get('file_info', {}).get('time_diff_seconds', 0)
            has_editing_software = any('editing_software' in flag.lower() for flag in flags)
            
            return {
                'metadata': metadata,
                'flags': flags,
                'time_diff_seconds': time_diff,
                'editing_software_detected': has_editing_software
            }
        except Exception as e:
            return {'error': str(e), 'flags': [], 'time_diff_seconds': 0, 'editing_software_detected': False}
    
    def _run_ocr(self, img_path: str, financial_pages: dict = None) -> dict:
        try:            
            start = time.time()
            # Pass financial_pages to OCR function to only extract from relevant pages (but with 1 more layer nah maybe no need)
            if img_path.lower().endswith('.pdf') and financial_pages:
                # For PDFs, only extract from identified pages
                ocr_data = extract_text_from_image(img_path, target_pages=financial_pages)
            else:
                # For images, extract all
                ocr_data = extract_text_from_image(img_path)
            
            ocr_time = time.time() - start     
            ocr_dir = Path("ocr_results")
            ocr_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = Path(img_path).stem

            rnd_suffix = random.randint(100, 9999) 
            ocr_file = ocr_dir / f"{filename}_{timestamp}_{rnd_suffix}.json"
            
            # Add page information to saved data
            if financial_pages:
                ocr_data['financial_pages'] = financial_pages
            
            with open(ocr_file, 'w', encoding='utf-8') as f:
                json.dump(ocr_data, f, ensure_ascii=False, indent=2)
            
            total_lines = len(ocr_data.get('text_lines', []))
            full_text_len = len(ocr_data.get('full_text', ''))
            
            return {
                'ocr_data': ocr_data,
                'ocr_file': str(ocr_file),
                'total_lines': total_lines,
                'text_length': full_text_len,
                'ocr_time': ocr_time,
                'financial_pages': financial_pages
            }
        except Exception as e:
            return {'error': str(e), 'ocr_time': 0.0}
    
    def _run_gemini_ocr(self, ocr_result: dict, filename: str = "") -> dict:
        try:
            if 'error' in ocr_result:
                return None
            
            start = time.time()
            analysis = self.gemini_ocr_analyzer.analyze_transactions(
                ocr_result['ocr_file']
            )
            gemini_ocr_time = time.time() - start
            self.gemini_ocr_time += gemini_ocr_time
            
            analysis['gemini_ocr_time'] = gemini_ocr_time
            return analysis
        except Exception as e:
            return {'error': str(e), 'gemini_ocr_time': 0.0}
    
    def _calculate_risk(self, ela_result: dict, metadata_result: dict = None, gemini_result: dict = None, gemini_ocr_result: dict = None) -> tuple:
        """Calculate combined risk score (4-layer) with detailed breakdown"""
        contributions = {}
        risk = 0.0
        
        # ELA
        ela_contribution = 0.0
        if not ela_result.get('skipped', False) and ela_result['is_suspicious']:
            ela_contribution = 0.15
            risk = min(risk + ela_contribution, 1.0)
        contributions['ELA'] = ela_contribution
        
        # Metadata Analysis
        metadata_contribution = 0.0
        if metadata_result and 'error' not in metadata_result:
            time_diff = metadata_result.get('time_diff_seconds', 0)            
            if time_diff > 500:
                metadata_contribution += 0.25
                risk = min(risk + 0.3, 1.0)
            elif time_diff > 100:
                metadata_contribution += 0.15
                risk = min(risk + 0.2, 1.0)
            
            if metadata_result.get('editing_software_detected', False):
                flags = metadata_result.get('flags', [])
                is_pdfium_only = any('pdf_editing_software_low_risk' in f for f in flags) and \
                                 not any('pdf_editing_software:' in f for f in flags)
                if is_pdfium_only:
                    metadata_contribution += 0.3
                    risk = min(risk + 0.3, 1.0)
                else:
                    metadata_contribution += 0.3
                    risk = min(risk + 0.3, 1.0)
        contributions['Metadata'] = metadata_contribution
        
        # Gemini Vision
        gemini_contribution = 0.0
        if gemini_result:
            if gemini_result['tampering_detected']:
                gemini_contribution += 0.3
                risk = min(risk + 0.3, 1.0)
            
            if gemini_result['recommendation'] == 'REJECT':
                old_risk = risk
                risk = max(risk, 0.6)
                gemini_contribution += (risk - old_risk)
        contributions['Gemini Vision'] = gemini_contribution

        # Gemini OCR
        ocr_contribution = 0.0
        if gemini_ocr_result and 'error' not in gemini_ocr_result:
            if gemini_ocr_result.get('fraud_detected'):
                ocr_contribution += 0.3
                risk = min(risk + 0.3, 1.0)
            
            if gemini_ocr_result.get('recommendation') == 'REJECT':
                old_risk = risk
                risk = max(risk, 0.7)
                ocr_contribution += (risk - old_risk)
        contributions['Gemini OCR'] = ocr_contribution
        
        contributions['TOTAL'] = risk
        
        return risk, contributions
    
    def _print_batch_summary_simple(self, results: list):
        """Print simplified batch summary with detailed format"""
        for idx, r in enumerate(results, 1):
            filename = Path(r.get('image_path', 'Unknown')).name
            recommendation = r.get('recommendation', 'UNKNOWN')
            
            tampering_detected = "NO" if recommendation == 'ACCEPT' else "YES"
            
            # Extract what and where
            what_where = ""
            rationale = ""
            
            if tampering_detected == "YES":
                # Check for early termination reasons first
                if r.get('early_termination'):
                    term_reason = r.get('termination_reason', '')
                    
                    # Metadata detected high-risk editing software
                    if 'editing software' in term_reason.lower():
                        what_where = "Metadata"
                        rationale = term_reason
                    
                    else:
                        rationale = term_reason
                # Flow
                else:
                    metadata = r.get('metadata') or {}
                    gemini = r.get('gemini') or {}
                    gemini_ocr = r.get('gemini_ocr') or {}
                    
                    # Check if Metadata is the primary issue
                    metadata_issues = []
                    if metadata.get('editing_software_detected'):
                        metadata_issues.append('Editing software detected')
                    if metadata.get('time_diff_seconds', 0) > 100:
                        metadata_issues.append('File edited long after its creation')
                    
                    # If metadata issues found, set what_where to Metadata
                    if metadata_issues:
                        what_where = "Metadata"
                        rationale = '; '.join(metadata_issues)
                    else:
                        # Extract from Gemini Vision findings
                        findings = gemini.get('findings', [])
                        if findings:
                            what_where = "; ".join(findings[:2])  # First 2 findings
                        
                        # Get rationale from Gemini OCR
                        rationale = gemini_ocr.get('explanation', '')
                        
                        # Fallback to Gemini Vision findings
                        if not rationale and findings:
                            rationale = findings[0]
                        
                        if not rationale and gemini.get('tampering_detected'):
                            what_where = "Gemini Vision"
                            rationale = f"Visual tampering detected (confidence: {gemini.get('confidence', 0):.0%})"

                        if not rationale and gemini_ocr.get('fraud_detected'):
                            what_where = "Gemini OCR"
                            rationale = "Cross-statement inconsistency detected"
                        
                        # Final fallback
                        if not rationale:
                            rationale = 'Suspicious activity detected'
            
            print(f"{idx}: {filename}, tampering detection: {tampering_detected}, what and where: \"{what_where}\", rationale: \"{rationale}\"")


def main():
    system = FraudDetectionSystem(
        use_gemini=True, 
        use_ocr=True,
        summary_type='simple'
    )
    
    folder_path = "./dataset_other_documents"
    results = system.batch_analyze(folder_path)


if __name__ == "__main__":
    main()