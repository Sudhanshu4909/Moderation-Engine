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

def load_hate_speech_model(model_path="IMSyPP/hate_speech_en"):
    """Load hate speech detection model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading hate speech model from: {model_path}")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
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
            "confidence": round(confidence, 3)
        }
    except Exception as e:
        return {"label": 0, "label_name": "acceptable", "confidence": 0.0, "error": str(e)}

def analyze_single_text(text, model_path="IMSyPP/hate_speech_en"):
    """Analyze a single text for hate speech"""
    
    # Load model
    model, tokenizer, device = load_hate_speech_model(model_path)
    
    # Classify text
    result = classify_hate_speech(text, model, tokenizer, device)
    
    # Display result
    print("\n=== HATE SPEECH ANALYSIS ===")
    print(f"Text: '{text}'")
    print(f"Label: {result['label']} ({result['label_name']})")
    print(f"Confidence: {result['confidence']}")
    if "error" in result:
        print(f"Error: {result['error']}")
    
    return result

def analyze_text_list(texts, model_path="IMSyPP/hate_speech_en", save_results=True):
    """Analyze a list of texts for hate speech"""
    
    # Load model once for all texts
    model, tokenizer, device = load_hate_speech_model(model_path)
    
    results = []
    
    print(f"\nAnalyzing {len(texts)} texts...")
    
    for i, text in enumerate(texts, 1):
        result = classify_hate_speech(text, model, tokenizer, device)
        result["text_id"] = i
        result["text"] = text
        results.append(result)
        
        print(f"{i}. [{result['label_name']}] {text[:50]}..." if len(text) > 50 else f"{i}. [{result['label_name']}] {text}")
    
    # Summary statistics
    print("\n=== SUMMARY ===")
    print(f"Total texts analyzed: {len(results)}")
    acceptable = len([r for r in results if r['label'] == 0])
    inappropriate = len([r for r in results if r['label'] == 1])
    offensive = len([r for r in results if r['label'] == 2])
    violent = len([r for r in results if r['label'] == 3])
    
    print(f"Acceptable: {acceptable}")
    print(f"Inappropriate: {inappropriate}")
    print(f"Offensive: {offensive}")
    print(f"Violent: {violent}")
    
    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as CSV
        csv_filename = f"hate_speech_results_{timestamp}.csv"
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['text_id', 'text', 'label', 'label_name', 'confidence', 'error']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = {
                    'text_id': result.get('text_id', ''),
                    'text': result.get('text', ''),
                    'label': result.get('label', 0),
                    'label_name': result.get('label_name', ''),
                    'confidence': result.get('confidence', 0),
                    'error': result.get('error', '')
                }
                writer.writerow(row)
        
        # Save as JSON
        json_filename = f"hate_speech_results_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(results, jsonfile, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {csv_filename} and {json_filename}")
    
    return results

def analyze_text_file(file_path, model_path="IMSyPP/hate_speech_en", save_results=True):
    """Analyze texts from a file (one text per line)"""
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    print(f"Reading texts from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    if not texts:
        print("No texts found in file")
        return
    
    return analyze_text_list(texts, model_path, save_results)

def interactive_mode(model_path="IMSyPP/hate_speech_en"):
    """Interactive mode for analyzing texts one by one"""
    
    print("\n=== INTERACTIVE HATE SPEECH DETECTION ===")
    print("Enter text to analyze (or 'quit' to exit)")
    
    # Load model once
    model, tokenizer, device = load_hate_speech_model(model_path)
    
    while True:
        print("\n" + "="*50)
        text = input("Enter text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not text:
            print("Please enter some text")
            continue
        
        result = classify_hate_speech(text, model, tokenizer, device)
        
        print(f"\nLabel: {result['label']} ({result['label_name']})")
        print(f"Confidence: {result['confidence']}")
        if "error" in result:
            print(f"Error: {result['error']}")

def test_example():
    """Test with example texts"""
    
    example_texts = [
        "This is a normal conversation about sports",
        "I hate this stupid content",
        "This contains offensive language and threats",
        "i want to lick your pussy",
        "I will fuck you like a dog",
        "Hello, how are you doing today?",
        "The weather is nice today",
        "You are an idiot and should die",
        "I love spending time with my family"
    ]
    
    print("Testing with example texts...")
    analyze_text_list(example_texts, save_results=True)

def main():
    """Main function with options for different processing modes"""
    
    print("Hate Speech Detection System")
    print("="*50)
    print("1. Analyze single text")
    print("2. Analyze list of texts")
    print("3. Analyze texts from file")
    print("4. Interactive mode")
    print("5. Test with examples")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    model_path = "IMSyPP/hate_speech_en"
    
    if choice == "1":
        # Single text mode
        text = input("Enter text to analyze: ").strip()
        if text:
            analyze_single_text(text, model_path)
        else:
            print("No text provided")
    
    elif choice == "2":
        # Multiple texts mode
        print("Enter texts (one per line, empty line to finish):")
        texts = []
        while True:
            text = input().strip()
            if not text:
                break
            texts.append(text)
        
        if texts:
            analyze_text_list(texts, model_path, save_results=True)
        else:
            print("No texts provided")
    
    elif choice == "3":
        # File mode
        file_path = input("Enter file path: ").strip()
        if file_path:
            analyze_text_file(file_path, model_path, save_results=True)
        else:
            print("No file path provided")
    
    elif choice == "4":
        # Interactive mode
        interactive_mode(model_path)
    
    elif choice == "5":
        # Test examples
        test_example()
    
    else:
        print("Invalid choice. Please select 1-5.")

if __name__ == "__main__":
    main()