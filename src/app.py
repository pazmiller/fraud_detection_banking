"""
Bank Statement Fraud Detection - ELA + CLIP + Gemini Integration
Workflow: Image → ELA → CLIP → Gemini Vision → Final Decision
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import threading 
import random 
from concurrent.futures import ThreadPoolExecutor, as_completed


load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / 'clip'))
sys.path.insert(0, str(Path(__file__).parent / 'ela_detection'))
sys.path.insert(0, str(Path(__file__).parent / 'llm'))
sys.path.insert(0, str(Path(__file__).parent / 'ocr_test'))
sys.path.insert(0, str(Path(__file__).parent / 'metadata'))

from clip_engine import BankStatementCLIPEngine
from ela_detection.ela import ela_detect
from llm.gemini1 import GeminiTamperingDetector
from ocr_test.new_ocr import extract_text_from_image
from llm.gemini_ocr import GeminiOCRAnalyzer
from metadata_analysis import extract_metadata, check_tampering_indicators
import json
from datetime import datetime
import time

json_results_store_path = "results_record_flash.json"

class FraudDetectionSystem:
    """Combined ELA + CLIP + Metadata + Gemini + OCR + Gemini OCR Analysis (6-Layer Pipeline)"""
    
    def __init__(self, ela_threshold=30.0, use_gemini=True, use_ocr=True, gemini_api_key=None):
        self.clip_engine = BankStatementCLIPEngine()
        self.ela_threshold = ela_threshold
        self.use_gemini = use_gemini
        self.use_ocr = use_ocr
        self.gemini_vision_time = 0.0
        self.gemini_ocr_time = 0.0
        self.print_lock = threading.Lock()
        
        if self.use_gemini:
            try:
                self.gemini_detector = GeminiTamperingDetector(api_key=gemini_api_key)
                print("[Gemini Vision enabled for enhanced tampering detection]")
            except Exception as e:
                print(f"⚠️  Gemini Vision initialization failed: {e}")
                print("   Continuing without Gemini Vision...")
                self.use_gemini = False
        if self.use_ocr and gemini_api_key:
            try:
                self.gemini_ocr_analyzer = GeminiOCRAnalyzer(api_key=gemini_api_key)
                print("[OCR + Gemini OCR Analysis enabled]")
            except Exception as e:
                print(f"⚠️  Gemini OCR initialization failed: {e}")
                print("   Continuing without OCR analysis...")
                self.use_ocr = False
    
    def batch_analyze(self, folder_path: str) -> list:
        """Batch analysis using ThreadPoolExecutor (Max 4 threads)"""
        start_time = time.time()
        self.gemini_vision_time = 0.0
        self.gemini_ocr_time = 0.0
        
        image_paths = self._get_image_paths(folder_path)
        if not image_paths:
            return []
        
        layers = "ELA + CLIP + Metadata + Gemini Vision" + (" + OCR + Gemini OCR" if self.use_ocr else "")
        print(f"\n{'='*80}\nBatch Analysis (Parallel): {len(image_paths)} files ({layers})\n{'='*80}\n")
        
        results = []
        max_workers = 4 # Depends on CPU threads
        
        print(f"Starting ThreadPool with {max_workers} workers")
        
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
        # Total time spend, could be inaccuracies due to multi-threading
        self._print_batch_summary(results, elapsed_time, self.gemini_vision_time, self.gemini_ocr_time)
        return results

    def _analyze_single_safe_capture(self, img_path: str, idx: int, total: int):
        """
        Wrapper: Add random delay and perform analysis
        """
        sleep_time = random.uniform(0.5, 1.5) # Jitter
        time.sleep(sleep_time)
        
        filename = Path(img_path).name
        
        # Thread lock for thread safety
        with self.print_lock:
            print(f"\n{'='*80}")
            print(f"[Thread {threading.current_thread().name}] 🚀 Starting: {filename}")
            print(f"{'='*80}")
        
        try:
            result = self._analyze_single(img_path, idx, total)
            with self.print_lock:
                print(f"\n{'='*80}")
                print(f"[Thread {threading.current_thread().name}] ✅ Completed: {filename} → {result.get('recommendation', 'UNKNOWN')}")
                print(f"{'='*80}\n")
            
            return result, ""  
            
        except Exception as e:
            with self.print_lock:
                print(f"\n❌ [Thread {threading.current_thread().name}] ERROR processing {filename}: {e}\n")
            return {'image_path': img_path, 'error': str(e), 'recommendation': 'REJECT'}, ""
    
    def _get_image_paths(self, folder_path: str) -> list:
        """Get all image paths from folder"""
        folder = Path(folder_path)
        paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png','*.pdf']:
            paths.extend(folder.glob(ext))
            paths.extend(folder.glob(ext.upper()))
        
        paths = sorted(set(str(p) for p in paths))
        if not paths:
            print("❌ No images found")
        return paths
    
    def _analyze_single(self, img_path: str, idx: int, total: int) -> dict:
        """Analyze single image (Logic unchanged, output is captured by wrapper)"""
        # Save the printing to buffer
        print(f"\n[{idx}/{total}] {Path(img_path).name} (Processing...)")
        
        try:
            # 1. Metadata
            metadata_result = self._run_metadata(img_path)

            
            # 1.5. Check for high-risk editing software (Early Termination)
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
                print(f"\n  ⚠️  REJECT: High-risk editing software detected: {detected_software}")
                print(f"  This document was edited with professional software commonly used for forgery.")
                print(f"  Skipping further analysis - CONFIRMED FRAUDULENT.")
                result = {
                    'image_path': img_path,
                    'ela': {'skipped': True},
                    'clip': clip_result,
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
                self._print_result(result)
                return result
            
            # 2. CLIP
            clip_result = self._run_clip(img_path)
            if not clip_result['verification']['is_bank_statement']:
                print(f"\n  REJECT: Not a bank statement, detected by CLIP, skip.")
                print(f"Please check if you have submitted the right bank statement. Please report if this is a misjudgement. Thank you!")              
                result = {
                    'image_path': img_path,
                    'ela': {'skipped': True},
                    'clip': clip_result,
                    'metadata': metadata_result,
                    'gemini': {'skipped': True, 'reason': 'Not a bank statement'},
                    'ocr': {'skipped': True, 'reason': 'Not a bank statement'},
                    'gemini_ocr': {'skipped': True, 'reason': 'Not a bank statement'},
                    'final_risk_score': 1.0,
                    'final_risk_level': 'CRITICAL',
                    'recommendation': 'REJECT',
                    'early_termination': True,
                    'termination_reason': 'CLIP detected non-bank statement document'
                }
                self._print_result(result)
                return result

            # 3. ELA
            ela_result = self._run_ela(img_path)
            
            # 4. Gemini Vision (API Call)
            gemini_result = None
            if self.use_gemini:
                gemini_result = self._run_gemini(img_path)
                time.sleep(1.0) 

            # 5. OCR (Local or API)
            ocr_result = self._run_ocr(img_path) if self.use_ocr else None

            # 6. Gemini OCR Analysis (API Call)
            gemini_ocr_result = self._run_gemini_ocr(ocr_result) if (self.use_ocr and ocr_result) else None
            
            # Calculate final risk
            final_risk_score = self._calculate_risk(ela_result, clip_result, metadata_result, gemini_result, gemini_ocr_result)
            
            # Build result
            result = {
                'image_path': img_path,
                'ela': ela_result,
                'clip': clip_result,
                'metadata': metadata_result,
                'gemini': gemini_result,
                'ocr': ocr_result,
                'gemini_ocr': gemini_ocr_result,
                'final_risk_score': final_risk_score,
                'final_risk_level': 'HIGH' if final_risk_score > 0.6 else 'MEDIUM' if final_risk_score > 0.3 else 'LOW',
                'recommendation': 'REJECT' if final_risk_score > 0.6 else 'MANUAL_REVIEW' if final_risk_score > 0.3 else 'ACCEPT'
            }
            
            self._print_result(result)
            return result
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return {'image_path': img_path, 'error': str(e), 'recommendation': 'REJECT'}
    
    def _run_ela(self, img_path: str) -> dict:
        """Run ELA detection (skip for PDF)"""
        if img_path.lower().endswith('.pdf'):
            print(f"\n  [ELA Detection - SKIPPED for PDF]")
            return {'verdict': 'SKIPPED', 'is_suspicious': False, 'max_difference': 0.0, 'skipped': True}
        
        print(f"\n  [ELA Detection]")
        result = ela_detect(img_path, threshold=self.ela_threshold)
        result['skipped'] = False
        print(f"    Verdict: {result['verdict']}, Max Diff: {result['max_difference']:.1f}")
        return result
    
    def _run_clip(self, img_path: str) -> dict:
        """Run CLIP analysis"""
        print(f"\n  [CLIP Analysis]")
        return self.clip_engine.comprehensive_analysis(img_path, debug=False)
    
    def _run_gemini(self, img_path: str) -> dict:
        """Run Gemini vision analysis"""
        print(f"\n  [Gemini Vision Analysis]")
        start = time.time()
        result = self.gemini_detector.analyze_tampering(img_path)
        gemini_vision_time = time.time() - start
        self.gemini_vision_time += gemini_vision_time  # Might not be thread-safe
        print(f"    {result['tampering_detected'] and 'Tampering' or 'Clean'} | "
              f"Confidence: {result['confidence']:.1%} | Risk: {result['risk_level']}")
        if result['findings']:
            print(f"    Findings: {', '.join(result['findings'][:2])}")
        print(f"    ⏱️  Gemini Vision Time: {gemini_vision_time:.2f}s")
        result['gemini_vision_time'] = gemini_vision_time
        return result
    
    def _run_metadata(self, img_path: str) -> dict:
        """Run Metadata Analysis"""
        print(f"\n  [Metadata Analysis]")
        try:
            metadata = extract_metadata(img_path)
            flags = check_tampering_indicators(metadata)
            time_diff = metadata.get('file_info', {}).get('time_diff_seconds', 0) # Check for key indicators of tampering in Metadata
            has_editing_software = any('editing_software' in flag.lower() for flag in flags)
            
            print(f"    Time Difference: {time_diff}s")
            print(f"    Editing Software Detected: {has_editing_software}")
            if flags:
                print(f"    Flags: {', '.join(flags[:3])}")
            
            return {
                'metadata': metadata,
                'flags': flags,
                'time_diff_seconds': time_diff,
                'editing_software_detected': has_editing_software
            }
        except Exception as e:
            print(f"    ⚠️ Metadata extraction failed: {e}")
            return {'error': str(e), 'flags': [], 'time_diff_seconds': 0, 'editing_software_detected': False}
    
    def _run_ocr(self, img_path: str) -> dict:
        """Run OCR extraction"""
        print(f"\n  [OCR Extraction]")
        try:
            start = time.time()
            ocr_data = extract_text_from_image(img_path)
            ocr_time = time.time() - start     
            ocr_dir = Path("ocr_results")
            ocr_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = Path(img_path).stem

            rnd_suffix = random.randint(100, 9999) 
            ocr_file = ocr_dir / f"{filename}_{timestamp}_{rnd_suffix}.json"
            
            with open(ocr_file, 'w', encoding='utf-8') as f:
                json.dump(ocr_data, f, ensure_ascii=False, indent=2)
            
            total_lines = len(ocr_data.get('text_lines', []))
            full_text_len = len(ocr_data.get('full_text', ''))
            
            print(f"    Total Text Lines: {total_lines}")
            print(f"    Full Text Extracted: {full_text_len:,} characters")
            print(f"    OCR Output saved to: {ocr_file}")
            print(f"    ⏱️  OCR Extraction Time: {ocr_time:.2f}s")
            
            return {
                'ocr_data': ocr_data,
                'ocr_file': str(ocr_file),
                'total_lines': total_lines,
                'text_length': full_text_len,
                'ocr_time': ocr_time
            }
        except Exception as e:
            print(f"    ❌ OCR failed: {e}")
            return {'error': str(e), 'ocr_time': 0.0}
    
    def _run_gemini_ocr(self, ocr_result: dict) -> dict:
        """Run Gemini OCR analysis"""
        print(f"\n  [Gemini OCR Analysis]")
        try:
            if 'error' in ocr_result:
                print(f"    Skipped (OCR failed)")
                return None
            
            start = time.time()
            analysis = self.gemini_ocr_analyzer.analyze_transactions(
                ocr_result['ocr_file']
            )
            gemini_ocr_time = time.time() - start
            self.gemini_ocr_time += gemini_ocr_time
            
            fraud_detected = analysis.get('fraud_detected', False)
            confidence = analysis.get('confidence', 0.0)
            risk_level = analysis.get('risk_level', 'UNKNOWN')
            
            print(f"    FRAUD_DETECTED: {fraud_detected and 'YES' or 'NO'}")
            print(f"    CONFIDENCE: {confidence:.1%}")
            print(f"    RISK_LEVEL: {risk_level}")
            
            # Print suspicious transactions
            suspicious_trans = analysis.get('suspicious_transactions', [])
            print(f"\n    SUSPICIOUS_TRANSACTIONS:")
            if suspicious_trans:
                for trans in suspicious_trans:
                    print(f"      - {trans}")
            else:
                print(f"      - None detected")
            
            # Print patterns detected
            patterns = analysis.get('patterns_detected', [])
            print(f"\n    PATTERNS_DETECTED:")
            if patterns:
                for pattern in patterns:
                    print(f"      - {pattern}")
            else:
                print(f"      - No unusual patterns")
            
            print(f"\n RECOMMENDATION: {analysis.get('recommendation', 'UNKNOWN')}")
            print(f"Gemini OCR Time: {gemini_ocr_time:.2f}s")
            
            # Print reason/explanation if fraud detected
            if fraud_detected:
                explanation = analysis.get('explanation', '')
                if explanation:
                    print(f"\n    Reason: {explanation}")
            
            analysis['gemini_ocr_time'] = gemini_ocr_time
            return analysis
        except Exception as e:
            print(f"    ❌ Gemini OCR analysis failed: {e}")
            return {'error': str(e), 'gemini_ocr_time': 0.0}
    
    def _calculate_risk(self, ela_result: dict, clip_result: dict, metadata_result: dict = None, gemini_result: dict = None, gemini_ocr_result: dict = None) -> float:
        """Calculate combined risk score (6-layer)"""
        risk = clip_result['overall_risk']['risk_score']
        
        # ELA  (skipped if PDF)
        if not ela_result.get('skipped', False) and ela_result['is_suspicious']:
            risk = min(risk + 0.15, 1.0)
        
        # CLIP tampering
        if clip_result['tampering_detection']['tampering_risk_score'] > 0.3:
            risk = max(risk, 0.3)
        
        # Metadata Analysis
        if metadata_result and 'error' not in metadata_result:
            # Time difference between created and modified > 100 seconds
            time_diff = metadata_result.get('time_diff_seconds', 0)
            if time_diff > 100:
                risk = min(risk + 0.2, 1.0)
            
            # Editing software detected
            if metadata_result.get('editing_software_detected', False):
                risk = min(risk + 0.3, 1.0)
        
        # Gemini Vision
        if gemini_result:
            if gemini_result['tampering_detected']:
                risk = min(risk + gemini_result['confidence'] * 0.3, 1.0)
            if gemini_result['recommendation'] == 'REJECT':
                risk = max(risk, 0.6)
        
        # OCR + Gemini OCR - Transaction Analysis
        if gemini_ocr_result and 'error' not in gemini_ocr_result:
            if gemini_ocr_result.get('fraud_detected'):
                ocr_confidence = gemini_ocr_result.get('confidence', 0.5)
                risk = min(risk + ocr_confidence * 0.4, 1.0)
            if gemini_ocr_result.get('recommendation') == 'REJECT':
                risk = max(risk, 0.7)
        
        return risk
    
    def _print_result(self, result: dict):
        """Print analysis result"""
        layers = 6 if self.use_ocr else 4
        print(f"\n  [Combined Result ({layers}-Layer Analysis)]")
        print(f"    Final Risk Score: {result['final_risk_score']:.2f}")
        print(f"    Final Risk Level: {result['final_risk_level']}")
        print(f"    Recommendation: {result['recommendation']}")
        print(f"    {'='*60}")
    
    def _print_batch_summary(self, results: list, elapsed_time: float = None, gemini_vision_time: float = 0.0, gemini_ocr_time: float = 0.0):
        """Print batch summary with file lists and timing breakdown"""
        total = len(results)
        accept_files = [Path(r['image_path']).name for r in results if r.get('recommendation') == 'ACCEPT']
        review_files = [Path(r['image_path']).name for r in results if r.get('recommendation') == 'MANUAL_REVIEW']
        reject_files = [Path(r['image_path']).name for r in results if r.get('recommendation') == 'REJECT']
        
        # Calculate layer-specific times from individual results
        ocr_time_total = sum(r.get('ocr', {}).get('ocr_time', 0) for r in results)
        gemini_vision_time_total = sum(r.get('gemini', {}).get('gemini_vision_time', 0) for r in results if r.get('gemini'))
        gemini_ocr_time_total = sum(r.get('gemini_ocr', {}).get('gemini_ocr_time', 0) for r in results if r.get('gemini_ocr'))
        
        print(f"\n{'='*80}")
        print("BATCH SUMMARY")
        print(f"{'='*80}")
        if elapsed_time is not None:
            minutes = int(elapsed_time // 60)
            seconds = elapsed_time % 60
            print(f"⏱️  Total Time: {minutes}m {seconds:.2f}s ({elapsed_time:.2f}s)")
            print(f"\n   Layer Timing Breakdown:")
            if ocr_time_total > 0:
                print(f"   • OCR Extraction: {ocr_time_total:.2f}s ({ocr_time_total/elapsed_time*100:.1f}%)")
            if gemini_vision_time_total > 0:
                print(f"   • Gemini Vision: {gemini_vision_time_total:.2f}s ({gemini_vision_time_total/elapsed_time*100:.1f}%)")
            if gemini_ocr_time_total > 0:
                print(f"   • Gemini OCR: {gemini_ocr_time_total:.2f}s ({gemini_ocr_time_total/elapsed_time*100:.1f}%)")
        print(f"\nTotal: {total} files")
        print(f"✅ Accept: {len(accept_files)} ({len(accept_files)/total*100:.1f}%)")
        print(f"⚠️  Manual Review: {len(review_files)} ({len(review_files)/total*100:.1f}%)")
        print(f"❌ Reject: {len(reject_files)} ({len(reject_files)/total*100:.1f}%)")
        
        # Print file lists
        if accept_files:
            print(f"\n✅ Accepted Clean Files:")
            for filename in accept_files:
                print(f"   • {filename}")
        
        if review_files:
            print(f"\n⚠️ Could Use Manual Review:")
            for filename in review_files:
                print(f"   • {filename}")
        
        if reject_files:
            print(f"\n❌ REJECTED Files (Require Manual Review):")
            for filename in reject_files:
                print(f"   • {filename}")
        
        print(f"\n{'='*80}\n")
        
        self._save_results_to_json(accept_files, review_files, reject_files, elapsed_time, total)
    
    def _save_results_to_json(self, accept_files: list, review_files: list, reject_files: list, elapsed_time: float, total: int):
        """Save batch results to JSON file"""
        try:
            record_file = Path(json_results_store_path)
            if record_file.exists():
                with open(record_file, 'r', encoding='utf-8') as f:
                    try:
                        all_records = json.load(f)
                    except json.JSONDecodeError:
                         all_records = []
            else:
                all_records = []
            
            new_record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_files": total,
                "elapsed_time_seconds": round(elapsed_time, 2) if elapsed_time else None,
                "summary": {
                    "accept_count": len(accept_files),
                    "review_count": len(review_files),
                    "reject_count": len(reject_files)
                },
                "accepted_files": accept_files,
                "manual_review_files": review_files,
                "rejected_files": reject_files
            }
            
            all_records.append(new_record)
            with open(record_file, 'w', encoding='utf-8') as f:
                json.dump(all_records, f, ensure_ascii=False, indent=2)
            
            print(f"📝 Results saved to: {record_file.absolute()}")
            
        except Exception as e:
            print(f"⚠️  Failed to save results to JSON: {e}")


def main():
    overall_start_time = time.time()
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    
    if not gemini_api_key:
        print("⚠️  GEMINI_API_KEY not found in .env file")
        print("   Set use_gemini=False or add GEMINI_API_KEY to .env")
    
    # Initialise system with full 6-layer pipeline (specific ones can be disabled by = False)
    system = FraudDetectionSystem(
        use_gemini=True, 
        use_ocr=True, 
        gemini_api_key=gemini_api_key
    )
    
    folder_path = "./dataset"
    # folder_path = "./src/clip/statements"
    results = system.batch_analyze(folder_path)
    
    # Final summary
    accept_count = sum(1 for r in results if r.get('recommendation') == 'ACCEPT')
    if accept_count == len(results):
        print("✅ All documents ACCEPTED")
    else:
        print(f"⚠️  {len(results) - accept_count} documents need attention")
    
    # Print total execution time
    total_time = time.time() - overall_start_time
    minutes = int(total_time // 60)
    seconds = total_time % 60
    print(f"\nTotal Execution Time: {minutes}m {seconds:.2f}s")


if __name__ == "__main__":
    main()