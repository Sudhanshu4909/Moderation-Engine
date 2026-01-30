import os
import sys

# FIX MACOS THREADING ISSUES
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
import csv
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def load_hate_speech_model(model_path="IMSyPP/hate_speech_en"):
    """Load hate speech detection model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading hate speech model from: {model_path}")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!\n")
    
    return model, tokenizer, device

def classify_hate_speech(text, model, tokenizer, device):
    """
    Classify hate speech in text
    Returns:
    - 0: acceptable
    - 1: inappropriate  
    - 2: offensive
    - 3: violent
    """
    if not text or not text.strip():
        return {"label": 0, "label_name": "acceptable", "confidence": 1.0}
    
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        predicted_class_idx = predictions.argmax().item()
        confidence = predictions.max().item()
        
        # Map model output to our labels
        label_map = {0: "acceptable", 1: "inappropriate", 2: "offensive", 3: "violent"}
        label_name = label_map.get(predicted_class_idx, "unknown")
        
        return {
            "label": predicted_class_idx,
            "label_name": label_name,
            "confidence": round(confidence, 3),
            "all_probabilities": {i: round(predictions[0][i].item(), 3) for i in range(len(predictions[0]))}
        }
    except Exception as e:
        return {"label": 0, "label_name": "acceptable", "confidence": 0.0, "error": str(e)}

def evaluate_model_on_csv(csv_path, model_path="IMSyPP/hate_speech_en", save_results=True):
    """
    Evaluate the hate speech model on a CSV file
    
    CSV should have columns: text, label, subcategory, source
    label: 0 = non-hate speech, 1 = hate speech
    """
    
    print(f"Reading CSV file: {csv_path}")
    
    # Read CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Validate columns
    required_columns = ['text', 'label']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in CSV")
            print(f"Available columns: {list(df.columns)}")
            return
    
    print(f"Total rows in CSV: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Load model
    model, tokenizer, device = load_hate_speech_model(model_path)
    
    # Process each text
    results = []
    true_labels = []
    predicted_labels = []
    predicted_labels_binary = []  # For binary classification (hate/non-hate)
    
    print("\nProcessing texts...")
    
    for idx, row in df.iterrows():
        text = str(row['text']) if pd.notna(row['text']) else ""
        true_label = int(row['label'])
        
        # Get prediction
        prediction = classify_hate_speech(text, model, tokenizer, device)
        predicted_label = prediction['label']
        
        # Convert our 4-class prediction to binary (0=acceptable, 1/2/3=hate speech)
        predicted_binary = 0 if predicted_label == 0 else 1
        
        result = {
            'text': text,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'predicted_label_name': prediction['label_name'],
            'predicted_binary': predicted_binary,
            'confidence': prediction['confidence'],
            'subcategory': row.get('subcategory', ''),
            'source': row.get('source', ''),
            'match': true_label == predicted_binary,
            'all_probabilities': prediction.get('all_probabilities', {})
        }
        
        results.append(result)
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)
        predicted_labels_binary.append(predicted_binary)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(df)} texts...")
    
    print(f"\nCompleted processing {len(results)} texts")
    
    # Calculate metrics
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    # Binary classification metrics (hate vs non-hate)
    print("\n### BINARY CLASSIFICATION (Hate Speech Detection) ###")
    print(f"Your CSV labels: 0=non-hate, 1=hate")
    print(f"Model prediction: 0=acceptable (non-hate), 1/2/3=hate speech")
    
    accuracy = accuracy_score(true_labels, predicted_labels_binary)
    precision = precision_score(true_labels, predicted_labels_binary, average='binary', zero_division=0)
    recall = recall_score(true_labels, predicted_labels_binary, average='binary', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels_binary, average='binary', zero_division=0)
    
    print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels_binary)
    print("\nConfusion Matrix:")
    print("                  Predicted")
    print("                Non-Hate  Hate")
    print(f"Actual Non-Hate    {cm[0][0]:4d}    {cm[0][1]:4d}")
    print(f"Actual Hate        {cm[1][0]:4d}    {cm[1][1]:4d}")
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nTrue Positives (TP):  {tp} - Correctly identified hate speech")
    print(f"True Negatives (TN):  {tn} - Correctly identified non-hate speech")
    print(f"False Positives (FP): {fp} - Non-hate classified as hate")
    print(f"False Negatives (FN): {fn} - Hate speech missed")
    
    # Detailed breakdown by model's 4 categories
    print("\n### MODEL'S 4-CATEGORY BREAKDOWN ###")
    print("Model categories: 0=acceptable, 1=inappropriate, 2=offensive, 3=violent")
    
    for label in [0, 1, 2, 3]:
        count = predicted_labels.count(label)
        percentage = (count / len(predicted_labels)) * 100
        label_names = {0: "acceptable", 1: "inappropriate", 2: "offensive", 3: "violent"}
        print(f"{label} ({label_names[label]}): {count} samples ({percentage:.2f}%)")
    
    # Distribution by true label
    print("\n### DISTRIBUTION BY TRUE LABEL ###")
    true_label_counts = pd.Series(true_labels).value_counts().sort_index()
    for label, count in true_label_counts.items():
        percentage = (count / len(true_labels)) * 100
        label_name = "Non-Hate Speech" if label == 0 else "Hate Speech"
        print(f"{label} ({label_name}): {count} samples ({percentage:.2f}%)")
    
    # Category distribution for hate speech samples
    print("\n### HATE SPEECH SAMPLES - MODEL CATEGORY DISTRIBUTION ###")
    hate_speech_predictions = [results[i]['predicted_label'] for i in range(len(results)) if results[i]['true_label'] == 1]
    if hate_speech_predictions:
        for label in [0, 1, 2, 3]:
            count = hate_speech_predictions.count(label)
            percentage = (count / len(hate_speech_predictions)) * 100
            label_names = {0: "acceptable", 1: "inappropriate", 2: "offensive", 3: "violent"}
            print(f"{label} ({label_names[label]}): {count} samples ({percentage:.2f}%)")
    
    # Examples of misclassifications
    print("\n### SAMPLE MISCLASSIFICATIONS ###")
    misclassified = [r for r in results if not r['match']]
    
    if misclassified:
        print(f"\nTotal misclassifications: {len(misclassified)}")
        
        # False Negatives (hate speech missed)
        false_negatives = [r for r in misclassified if r['true_label'] == 1 and r['predicted_binary'] == 0]
        if false_negatives:
            print(f"\nFalse Negatives (Hate speech missed): {len(false_negatives)} samples")
            print("Examples (first 5):")
            for i, r in enumerate(false_negatives[:5], 1):
                text_preview = r['text'][:100] + "..." if len(r['text']) > 100 else r['text']
                print(f"{i}. Predicted: {r['predicted_label_name']} (confidence: {r['confidence']})")
                print(f"   Text: {text_preview}\n")
        
        # False Positives (non-hate classified as hate)
        false_positives = [r for r in misclassified if r['true_label'] == 0 and r['predicted_binary'] == 1]
        if false_positives:
            print(f"\nFalse Positives (Non-hate classified as hate): {len(false_positives)} samples")
            print("Examples (first 5):")
            for i, r in enumerate(false_positives[:5], 1):
                text_preview = r['text'][:100] + "..." if len(r['text']) > 100 else r['text']
                print(f"{i}. Predicted: {r['predicted_label_name']} (confidence: {r['confidence']})")
                print(f"   Text: {text_preview}\n")
    
    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_df = pd.DataFrame(results)
        csv_filename = f"evaluation_results_{timestamp}.csv"
        results_df.to_csv(csv_filename, index=False, encoding='utf-8')
        print(f"\nDetailed results saved to: {csv_filename}")
        
        # Save metrics summary
        metrics_summary = {
            "evaluation_date": datetime.now().isoformat(),
            "csv_file": csv_path,
            "total_samples": len(results),
            "binary_metrics": {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4)
            },
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp)
            },
            "model_category_distribution": {
                "0_acceptable": predicted_labels.count(0),
                "1_inappropriate": predicted_labels.count(1),
                "2_offensive": predicted_labels.count(2),
                "3_violent": predicted_labels.count(3)
            },
            "true_label_distribution": {
                "0_non_hate": true_labels.count(0),
                "1_hate": true_labels.count(1)
            }
        }
        
        json_filename = f"evaluation_metrics_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(metrics_summary, jsonfile=f, indent=2)
        print(f"Metrics summary saved to: {json_filename}")
    
    return results, metrics_summary

def main():
    """Main function"""
    
    print("="*70)
    print("HATE SPEECH MODEL EVALUATION")
    print("="*70)
    
    csv_path = input("\nEnter path to CSV file: ").strip()
    
    if not csv_path:
        print("No file path provided")
        return
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
    
    model_path = "IMSyPP/hate_speech_en"
    
    print(f"\nEvaluating model on: {csv_path}")
    print(f"Using model: {model_path}\n")
    
    evaluate_model_on_csv(csv_path, model_path, save_results=True)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()