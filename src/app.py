"""
Bank Statement Fraud Detection - ELA + CLIP + Gemini Integration
Workflow: Image → ELA → CLIP → Gemini Vision → Final Decision
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add paths
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


class FraudDetectionSystem:
    """Combined ELA + CLIP + Metadata + Gemini + OCR + Gemini OCR Analysis (6-Layer Pipeline)"""
    
    def __init__(self, ela_threshold=30.0, use_gemini=True, use_ocr=True, gemini_api_key=None):
        self.clip_engine = BankStatementCLIPEngine()
        self.ela_threshold = ela_threshold
        self.use_gemini = use_gemini
        self.use_ocr = use_ocr
        self.gemini_vision_time = 0.0
        self.gemini_ocr_time = 0.0
        
        # Initialize Gemini Vision (optional)
        if self.use_gemini:
            try:
                self.gemini_detector = GeminiTamperingDetector(api_key=gemini_api_key)
                print("[Gemini Vision enabled for enhanced tampering detection]")
            except Exception as e:
                print(f"⚠️  Gemini Vision initialization failed: {e}")
                print("   Continuing without Gemini Vision...")
                self.use_gemini = False
        
        # Initialize Gemini OCR Analyzer (optional)
        if self.use_ocr and gemini_api_key:
            try:
                self.gemini_ocr_analyzer = GeminiOCRAnalyzer(api_key=gemini_api_key)
                print("[OCR + Gemini OCR Analysis enabled]")
            except Exception as e:
                print(f"⚠️  Gemini OCR initialization failed: {e}")
                print("   Continuing without OCR analysis...")
                self.use_ocr = False
    
    def batch_analyze(self, folder_path: str) -> list:
        """Batch analysis for folder"""
        start_time = time.time()
        self.gemini_vision_time = 0.0
        self.gemini_ocr_time = 0.0
        
        image_paths = self._get_image_paths(folder_path)
        if not image_paths:
            return []
        
        layers = "ELA + CLIP + Metadata + Gemini Vision" + (" + OCR + Gemini OCR" if self.use_ocr else "")
        print(f"\n{'='*80}\nBatch Analysis: {len(image_paths)} files ({layers})\n{'='*80}\n")
        
        results = [self._analyze_single(img, idx, len(image_paths)) 
                   for idx, img in enumerate(image_paths, 1)]
        
        elapsed_time = time.time() - start_time
        self._print_batch_summary(results, elapsed_time, self.gemini_vision_time, self.gemini_ocr_time)
        return results
    
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
        """Analyze single image"""
        print(f"\n[{idx}/{total}] {Path(img_path).name}")
        
        try:
            ela_result = self._run_ela(img_path)
            clip_result = self._run_clip(img_path)
            
            # Terminate EARLY: If CLIP detects it's not a bank statement, REJECT immediately
            if not clip_result['verification']['is_bank_statement']:
                print(f"\n  REJECT: Not a bank statement, detected by CLIP, skip.")
                print(f"Please check if you have submitted the right bank statement. Please report if this is a misjudgement. Thank you!")              
                result = {
                    'image_path': img_path,
                    'ela': ela_result,
                    'clip': clip_result,
                    'metadata': {'skipped': True, 'reason': 'Not a bank statement'},
                    'gemini': {'skipped': True, 'reason': 'Not a bank statement'},
                    'ocr': {'skipped': True, 'reason': 'Not a bank statement'},
                    'gemini_ocr': {'skipped': True, 'reason': 'Not a bank statement'},
                    'final_risk_score': 1.0,
                    'final_risk_level': 'HIGH',
                    'recommendation': 'REJECT',
                    'early_termination': True,
                    'termination_reason': 'CLIP detected non-bank statement document'
                }
                
                self._print_result(result)
                return result
            
            # Continue with remaining layers if it's a bank statement
            metadata_result = self._run_metadata(img_path)
            gemini_result = self._run_gemini(img_path) if self.use_gemini else None
            ocr_result = self._run_ocr(img_path) if self.use_ocr else None
            gemini_ocr_result = self._run_gemini_ocr(ocr_result) if (self.use_ocr and ocr_result) else None
            
            # Calculate final risk
            '''超1.0 reject, 满分2分'''
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
        self.gemini_vision_time += time.time() - start
        print(f"    {result['tampering_detected'] and 'Tampering' or 'Clean'} | "
              f"Confidence: {result['confidence']:.1%} | Risk: {result['risk_level']}")
        if result['findings']:
            print(f"    Findings: {', '.join(result['findings'][:2])}")
        return result
    
    def _run_metadata(self, img_path: str) -> dict:
        """Run Metadata Analysis"""
        print(f"\n  [Metadata Analysis]")
        try:
            metadata = extract_metadata(img_path)
            flags = check_tampering_indicators(metadata)
            
            # Check for key indicators
            time_diff = metadata.get('file_info', {}).get('time_diff_seconds', 0)
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
            ocr_data = extract_text_from_image(img_path)
            
            # Save OCR result to file
            ocr_dir = Path("ocr_results")
            ocr_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = Path(img_path).stem
            ocr_file = ocr_dir / f"{filename}_{timestamp}.json"
            
            with open(ocr_file, 'w', encoding='utf-8') as f:
                json.dump(ocr_data, f, ensure_ascii=False, indent=2)
            
            total_lines = len(ocr_data.get('text_lines', []))
            full_text_len = len(ocr_data.get('full_text', ''))
            
            print(f"    Total Text Lines: {total_lines}")
            print(f"    Full Text Extracted: {full_text_len:,} characters")
            print(f"    OCR Output saved to: {ocr_file}")
            
            return {
                'ocr_data': ocr_data,
                'ocr_file': str(ocr_file),
                'total_lines': total_lines,
                'text_length': full_text_len
            }
        except Exception as e:
            print(f"    ❌ OCR failed: {e}")
            return {'error': str(e)}
    
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
            self.gemini_ocr_time += time.time() - start
            
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
            
            print(f"\n    RECOMMENDATION: {analysis.get('recommendation', 'UNKNOWN')}")
            
            # Print reason/explanation if fraud detected
            if fraud_detected:
                explanation = analysis.get('explanation', '')
                if explanation:
                    print(f"\n    Reason: {explanation}")
            
            return analysis
        except Exception as e:
            print(f"    ❌ Gemini OCR analysis failed: {e}")
            return {'error': str(e)}
    
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
        """Print batch summary with file lists"""
        total = len(results)
        accept_files = [Path(r['image_path']).name for r in results if r.get('recommendation') == 'ACCEPT']
        review_files = [Path(r['image_path']).name for r in results if r.get('recommendation') == 'MANUAL_REVIEW']
        reject_files = [Path(r['image_path']).name for r in results if r.get('recommendation') == 'REJECT']
        
        print(f"\n{'='*80}")
        print("BATCH SUMMARY")
        print(f"{'='*80}")
        if elapsed_time is not None:
            minutes = int(elapsed_time // 60)
            seconds = elapsed_time % 60
            print(f"⏱️  Total Time: {minutes}m {seconds:.2f}s ({elapsed_time:.2f}s)")
            if gemini_vision_time > 0:
                print(f"   • Gemini Vision: {gemini_vision_time:.2f}s")
            if gemini_ocr_time > 0:
                print(f"   • Gemini OCR: {gemini_ocr_time:.2f}s")
        print(f"Total: {total} files")
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
        
        # Save results to JSON file
        self._save_results_to_json(accept_files, review_files, reject_files, elapsed_time, total)
    
    def _save_results_to_json(self, accept_files: list, review_files: list, reject_files: list, elapsed_time: float, total: int):
        """Save batch results to JSON file"""
        try:
            record_file = Path("results_record_flash.json")
            if record_file.exists():
                with open(record_file, 'r', encoding='utf-8') as f:
                    all_records = json.load(f)
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
    """Demo usage"""
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