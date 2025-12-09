import json
from pathlib import Path


def normalise_filename(filename: str) -> str:
    """Normalise filename for comparison (handle minor naming differences)"""
    normalised = filename.lower().strip()
    return normalised


def load_json(filepath: str) -> list:
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_accuracy(predicted_files: list, target_files: list, total_files: int = 16) -> dict:
    predicted_set = set(normalise_filename(f) for f in predicted_files)
    target_set = set(normalise_filename(f) for f in target_files)
    
    # Calculate matches
    true_positives = predicted_set & target_set  # Correctly flagged
    false_positives = predicted_set - target_set  # Wrongly flagged (should be clean)
    false_negatives = target_set - predicted_set  # Missed (should be flagged)
    
    # True Negatives = Total - TP - FP - FN (correctly identified as clean)
    true_negatives = total_files - len(true_positives) - len(false_positives) - len(false_negatives)
    
    # Accuracy = (TP + TN) / Total = (correctly classified) / Total
    accuracy = (len(true_positives) + true_negatives) / total_files
    
    # Precision = TP / (TP + FP)
    precision = len(true_positives) / len(predicted_set) if predicted_set else 0
    
    # Recall = TP / (TP + FN)  
    recall = len(true_positives) / len(target_set) if target_set else 0
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'correct_matches': len(true_positives),
        'total_predicted': len(predicted_files),
        'total_target': len(target_files),
        'true_negatives': true_negatives,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': list(true_positives),
        'false_positives': list(false_positives),
        'false_negatives': list(false_negatives)
    }


def main():
    # Load target (the corect sampling set for cross-comparing results)
    target_path = './results_taget.json'
    target_data = load_json(target_path)
    target_info = target_data[1]  # Second element has the file lists
    
    target_rejected = target_info.get('rejected_files', [])
    target_flagged = target_info.get('manual_review_files + rejected_files', [])

    results_dir = Path('./results')
    result_files = list(results_dir.glob('results_record_*.json'))
    
    print("FRAUD DETECTION PIPELINE - ACCURACY CALCULATOR")
    print(f"\nTarget file: {target_path}")
    print(f"Target rejected_files count: {len(target_rejected)}")
    print(f"Target flagged_files count: {len(target_flagged)}")
    
    print(f"\nAvailable result files:")
    for i, f in enumerate(result_files, 1):
        print(f"  {i}. {f.name}")
        
    try:
        choice = int(input("\nSelect file number (or 0 for all): "))
    except ValueError:
        choice = 0
    
    if choice == 0:
        selected_files = result_files
    else:
        selected_files = [result_files[choice - 1]]
    
    # Process each selected file
    for result_file in selected_files:
        print(f"Analyzing: {result_file.name}")      
        try:
            results_data = load_json(result_file)
        except json.JSONDecodeError:
            print(f"Skipped (empty or invalid JSON)")
            continue
        
        if not results_data:
            print(f" Skipped (no data)")
            continue
        

        all_metrics = []
        print(f"\nTotal records in file: {len(results_data)}")
        
        for idx, record in enumerate(results_data, 1):
            predicted_flagged = record.get('manual_review_files + rejected_files', [])
            metrics = calculate_accuracy(predicted_flagged, target_flagged)
            all_metrics.append(metrics)
            
            print(f"\n  Record #{idx} ({record.get('timestamp', 'N/A')})")
            print(f"    Accuracy: {metrics['accuracy']:.2%} | Precision: {metrics['precision']:.2%} | Recall: {metrics['recall']:.2%} | F1: {metrics['f1_score']:.2%}")
        
        # Calculate averages
        if all_metrics:
            avg_accuracy = sum(m['accuracy'] for m in all_metrics) / len(all_metrics)
            avg_precision = sum(m['precision'] for m in all_metrics) / len(all_metrics)
            avg_recall = sum(m['recall'] for m in all_metrics) / len(all_metrics)
            avg_f1 = sum(m['f1_score'] for m in all_metrics) / len(all_metrics)
            
            print(f"\n{'─'*60}")
            print(f"  AVERAGE across {len(all_metrics)} records:")
            print(f"    Accuracy:  {avg_accuracy:.2%}")
            print(f"    Precision: {avg_precision:.2%}")
            print(f"    Recall:    {avg_recall:.2%}")
            print(f"    F1 Score:  {avg_f1:.2%}")
    
    print(f"\n{'='*80}")
    print("Done!")


if __name__ == "__main__":
    main()