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
import pandas as pd
import json
from datetime import datetime

# Label mapping
LABEL_MAPPING = {
    0: "non-hate",
    1: "community",
    2: "racial_ethnic",
    3: "religious",
    4: "nationality"
}

def load_model(model_path):
    """Load the fine-tuned model and tokenizer"""
    print(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Set to evaluation mode
    
    print("Model loaded successfully!")
    print(f"Number of labels: {model.config.num_labels}")
    
    return model, tokenizer

def classify_batch(texts, model, tokenizer, batch_size=32):
    """
    Classify multiple texts in batches
    """
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(batch, return_tensors='pt', truncation=True, max_length=512, padding=True)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Process each result in batch
        for j, text in enumerate(batch):
            predicted_label = logits[j].argmax().item()
            confidence = probabilities[j].max().item()
            
            all_probs = {LABEL_MAPPING[k]: round(probabilities[j][k].item(), 4) for k in range(len(LABEL_MAPPING))}
            
            results.append({
                "predicted_label": predicted_label,
                "predicted_label_name": LABEL_MAPPING[predicted_label],
                "confidence": round(confidence, 4),
                "all_probabilities": all_probs
            })
        
        if (i + batch_size) < len(texts):
            print(f"Processed {i + batch_size}/{len(texts)} texts...")
    
    print(f"Processed {len(texts)}/{len(texts)} texts.")
    return results

def test_csv(csv_path, model, tokenizer, output_path=None, batch_size=32):
    """
    Test the model on a CSV file with columns: text, labels, subcategory, source
    """
    print(f"\n{'='*70}")
    print("TESTING CSV FILE")
    print(f"{'='*70}")
    print(f"Reading CSV from: {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    print(f"Found {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Verify required column exists
    if 'text' not in df.columns:
        print("ERROR: 'text' column not found in CSV!")
        return
    
    # Get texts
    texts = df['text'].fillna("").astype(str).tolist()
    
    print(f"\nClassifying {len(texts)} texts...")
    
    # Classify
    predictions = classify_batch(texts, model, tokenizer, batch_size=batch_size)
    
    # Add predictions to dataframe
    df['predicted_label'] = [p['predicted_label'] for p in predictions]
    df['predicted_label_name'] = [p['predicted_label_name'] for p in predictions]
    df['confidence'] = [p['confidence'] for p in predictions]
    
    # Add individual probability columns
    for label_name in LABEL_MAPPING.values():
        df[f'prob_{label_name}'] = [p['all_probabilities'][label_name] for p in predictions]
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("PREDICTION SUMMARY")
    print(f"{'='*70}")
    
    prediction_counts = df['predicted_label_name'].value_counts()
    for label, count in prediction_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{label}: {count} ({percentage:.1f}%)")
    
    # If labels column exists, show accuracy
    if 'labels' in df.columns:
        print(f"\n{'='*70}")
        print("ACCURACY METRICS")
        print(f"{'='*70}")
        
        # Assuming the labels column has the true labels (0-4)
        df['true_label'] = df['labels']
        correct = (df['predicted_label'] == df['true_label']).sum()
        accuracy = correct / len(df) * 100
        
        print(f"Overall Accuracy: {accuracy:.2f}% ({correct}/{len(df)})")
        
        # Per-class accuracy
        print("\nPer-class accuracy:")
        for label_id, label_name in LABEL_MAPPING.items():
            mask = df['true_label'] == label_id
            if mask.sum() > 0:
                class_correct = ((df['predicted_label'] == df['true_label']) & mask).sum()
                class_total = mask.sum()
                class_acc = class_correct / class_total * 100
                print(f"  {label_name}: {class_acc:.2f}% ({class_correct}/{class_total})")
        
        # Confusion info
        print(f"\n{'='*70}")
        print("CONFUSION MATRIX")
        print(f"{'='*70}")
        confusion_matrix = pd.crosstab(
            df['true_label'].map(LABEL_MAPPING), 
            df['predicted_label'].map(LABEL_MAPPING),
            rownames=['True'],
            colnames=['Predicted']
        )
        print(confusion_matrix)
    
    # Save results
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results_{timestamp}.csv"
    elif os.path.isdir(output_path):
        # If a directory was provided, create a filename in that directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_path, f"results_{timestamp}.csv")
    
    df.to_csv(output_path, index=False)
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")
    
    # Show some example predictions
    print(f"\n{'='*70}")
    print("SAMPLE PREDICTIONS (first 5 rows)")
    print(f"{'='*70}")
    
    sample_cols = ['text', 'predicted_label_name', 'confidence']
    if 'labels' in df.columns:
        sample_cols.insert(1, 'labels')
    
    for idx, row in df.head().iterrows():
        print(f"\nRow {idx + 1}:")
        print(f"  Text: {row['text'][:100]}...")
        if 'labels' in df.columns:
            true_label = LABEL_MAPPING.get(row['labels'], 'unknown')
            print(f"  True Label: {true_label}")
        print(f"  Predicted: {row['predicted_label_name']}")
        print(f"  Confidence: {row['confidence']:.2%}")
    
    return df

def main():
    """Main function"""
    
    # Model path - UPDATE THIS to where your model is
    MODEL_PATH = "./checkpoint-819"  # Change this to your model path
    
    print("="*70)
    print("CSV HATE SPEECH CLASSIFIER TESTER")
    print("="*70)
    
    # Check if model path exists
    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: Model not found at {MODEL_PATH}")
        print("Please update MODEL_PATH in the script to point to your model directory")
        return
    
    # Load model
    model, tokenizer = load_model(MODEL_PATH)
    
    # Get CSV path from user
    csv_path = input("\nEnter path to CSV file: ").strip()
    
    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        return
    
    # Get output path (optional)
    output_path = input("Enter output CSV path (press Enter for auto-generated name): ").strip()
    if not output_path:
        output_path = None
    
    # Get batch size
    batch_size_input = input("Enter batch size (press Enter for default 32): ").strip()
    batch_size = int(batch_size_input) if batch_size_input else 32
    
    # Run testing
    test_csv(csv_path, model, tokenizer, output_path, batch_size)

if __name__ == "__main__":
    main()