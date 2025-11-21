from pathlib import Path
from typing import List
import sys

# Import modules from the same directory
from clip_engine import BankStatementCLIPEngine


def main():
    DEBUG_MODE = True # for quality stats double detailed checkings
    engine = BankStatementCLIPEngine()
    
    # Location of the ingested bank statements
    folder_path = './dataset'
    
    if not folder_path:
        print("❌ No folder input found")
        return
    
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"❌ Does not exist. Try again: {folder_path}")
        return
    
    # Support common image formats
    image_paths = []
    extensions = ['*.jpg', '*.png', '*.bmp', '*.gif', '*.tiff']
    for ext in extensions:
        image_paths.extend(folder.glob(ext))
        image_paths.extend(folder.glob(ext.upper()))
    
    image_paths = [str(p) for p in image_paths]
    
    if not image_paths:
        print(f"\n❌ No image files found in folder: {folder_path}")
        print("Supported formats: JPG, JPEG, PNG, BMP, GIF, TIFF")
        return
    
    print(f"\n✅ Found {len(image_paths)} image files")
    
    # Execute batch analysis (with debug info)
    results = engine.batch_analysis(image_paths, show_progress=True, debug=DEBUG_MODE)
    
    # Ask if user wants to see detailed results
    print("\nView detailed analysis for each file? (y/n, default: n): ", end="")
    sys.stdout.flush()
    show_details = input().strip().lower() == 'y'
    
    if show_details:
        for result in results:
            if 'error' not in result:
                print_analysis_result(result)
                sys.stdout.flush()
    
    # Export option
    print("\nExport results to file? (y/n, default: n): ", end="")
    sys.stdout.flush()
    export = input().strip().lower() == 'y'
    
    if export:
        export_results(results)


def print_analysis_result(result: dict):
    """Format and print analysis results"""
    
    print("\n" + "="*80)
    print(" CLIP Analysis Results ")
    print("="*80)
    
    # 1. Document Verification
    print("\n1. Document Verification")
    print("-" * 80)
    verification = result['verification']
    
    if verification['is_bank_statement']:
        print("✅ Confirmed as bank statement")
    else:
        print("❌ Not a bank statement")
    
    print(f"   Confidence: {verification['confidence']:.1%}")
    print(f"   Predicted Type: {verification['predicted_type']}")
    
    print("\n   All Classification Scores (Top 5):")
    sorted_scores = sorted(verification['all_scores'].items(), 
                          key=lambda x: x[1], reverse=True)[:5]
    for doc_type, score in sorted_scores:
        print(f"   • {doc_type}: {score:.1%}")
    
    # 2. Quality Assessment
    print("\n2. Document Quality Assessment")
    print("-" * 80)
    quality = result['quality_assessment']
    
    quality_level = quality['quality_level']
    quality_score = quality['quality_score']
    
    # Display based on quality_level
    if quality_level == "high":
        print("✅ High Quality Document")
    elif quality_level == "medium":
        print("Medium Quality Document")
    else:
        print("❌ Low Quality Document")
    
    print(f"   Quality Level: {quality_level.upper()}")
    print(f"   Quality Score: {quality_score:.1%}")
    print(f"   Assessment: {quality['quality_assessment']}")
    
    # 3. Tampering Detection
    print("\n3. Tampering Detection (CLIP)")
    print("-" * 80)
    tampering = result['tampering_detection']
    
    tampering_score = tampering['tampering_risk_score']
    authentic_score = tampering['authenticity_score']
    
    # Display based on score
    if tampering_score > 0.5:
        print("Highly Suspicious")
    elif tampering_score > 0.35:
        print("Moderately Suspicious")
    elif tampering_score > 0.25:
        print("Slightly Suspicious")
    else:
        print("✅ No Obvious Anomalies Detected")
    
    print(f"   Tampering Risk Score: {tampering_score:.1%}")
    print(f"   Authenticity Score: {authentic_score:.1%}")
    print(f"   Most Likely Issue: {tampering['most_likely_issue']}")
    
    # 4. Overall Risk Assessment
    print("\n4. Overall Risk Assessment")
    print("-" * 80)
    risk = result['overall_risk']
    
    print(f"   Risk Level: {risk['risk_level']}")
    print(f"   Risk Score: {risk['risk_score']:.1%}")
    print(f"   Recommendation: {risk['recommendation']}")
    
    if risk['risk_factors']:
        print("\n   Risk Factors:")
        for factor in risk['risk_factors']:
            print(f"   • {factor}")
    else:
        print("\n   ✅ No significant risk factors found")
    
    # Summary
    print("\n" + "="*80)
    print(" Conclusion ")
    print("="*80)
    
    if risk['recommendation'] == 'ACCEPT':
        print("✅ Recommendation: ACCEPT this document")
        print("   Document passed all CLIP verification checks")
    elif risk['recommendation'] == 'MANUAL_REVIEW':
        print("Recommendation: MANUAL REVIEW required")
        print("   Some suspicious signs detected, manual review recommended")
    else:  # Reject
        print("❌ Recommendation: REJECT this document")
        print("   Serious issues detected, rejection recommended")
    
    print("\n" + "="*80 + "\n")


def export_results(results: List[dict]):
    """Export batch analysis results"""
    import json
    from datetime import datetime
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"clip_batch_results_{timestamp}.json"
    
    # Prepare export data
    export_data = {
        'timestamp': timestamp,
        'total_files': len(results),
        'summary': {
            'accept': sum(1 for r in results if r.get('overall_risk', {}).get('recommendation') == 'ACCEPT'),
            'review': sum(1 for r in results if r.get('overall_risk', {}).get('recommendation') == 'MANUAL_REVIEW'),
            'reject': sum(1 for r in results if r.get('overall_risk', {}).get('recommendation') == 'REJECT'),
            'errors': sum(1 for r in results if 'error' in r)
        },
        'results': results
    }
    
    # Write to file
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Results exported to: {filename}")
    except Exception as e:
        print(f"\n❌ Export failed: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nUser interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
