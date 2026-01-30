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
import json

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

def classify_text(text, model, tokenizer):
    """
    Classify a single text
    Returns: label (int), label_name (str), confidence (float), all_probabilities (dict)
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get prediction
    predicted_label = logits.argmax().item()
    confidence = probabilities.max().item()
    
    # Get all probabilities
    all_probs = {LABEL_MAPPING[i]: round(probabilities[0][i].item(), 4) for i in range(len(LABEL_MAPPING))}
    
    return {
        "text": text,
        "predicted_label": predicted_label,
        "predicted_label_name": LABEL_MAPPING[predicted_label],
        "confidence": round(confidence, 4),
        "all_probabilities": all_probs
    }

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
                "text": text,
                "predicted_label": predicted_label,
                "predicted_label_name": LABEL_MAPPING[predicted_label],
                "confidence": round(confidence, 4),
                "all_probabilities": all_probs
            })
        
        if (i + batch_size) < len(texts):
            print(f"Processed {i + batch_size}/{len(texts)} texts...")
    
    return results

def interactive_mode(model, tokenizer):
    """Interactive mode for testing texts"""
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("Enter text to classify (or 'quit' to exit)")
    print()
    
    while True:
        text = input("Enter text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not text:
            print("Please enter some text")
            continue
        
        result = classify_text(text, model, tokenizer)
        
        print(f"\nPrediction: {result['predicted_label']} - {result['predicted_label_name']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nAll probabilities:")
        for label, prob in result['all_probabilities'].items():
            print(f"  {label}: {prob:.2%}")
        print()

def test_examples(model, tokenizer):
    """Test with example texts"""
    print("\n" + "="*70)
    print("TESTING WITH EXAMPLE TEXTS")
    print("="*70)
    
    examples = [
        "I love spending time with my family and friends",
        "Those people from that community are all the same",
        "I hate those immigrants, they should go back",
        "People from that ethnic group are inferior",
        "That religion is evil and dangerous",
        "Let's have a peaceful discussion about our differences",
        "All Muslims are terrorists",
        "Black people are criminals",
        "Jews control the media",
        "The weather is nice today"
    ]
    
    print("\nClassifying examples...\n")
    
    results = classify_batch(examples, model, tokenizer)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Text: {result['text'][:70]}...")
        print(f"   Prediction: {result['predicted_label_name']} (confidence: {result['confidence']:.2%})")
        print()

def classify_from_file(file_path, model, tokenizer, output_file=None):
    """Classify texts from a file (one text per line)"""
    print(f"\nReading texts from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(texts)} texts")
    print("Classifying...")
    
    results = classify_batch(texts, model, tokenizer)
    
    print(f"\nClassified {len(results)} texts")
    
    # Summary
    label_counts = {}
    for result in results:
        label = result['predicted_label_name']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\nSummary:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} ({count/len(results)*100:.1f}%)")
    
    # Save results
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")
    
    return results

def main():
    """Main function with menu"""
    
    # Model path - UPDATE THIS to where you downloaded your model
    MODEL_PATH = "./checkpoint-819"  # or "./fine_tuned_model" or wherever you put it
    
    print("="*70)
    print("FINE-TUNED HATE SPEECH CLASSIFIER")
    print("="*70)
    
    # Load model
    model, tokenizer = load_model(MODEL_PATH)
    
    print("\n" + "="*70)
    print("MENU")
    print("="*70)
    print("1. Test with example texts")
    print("2. Interactive mode (classify your own texts)")
    print("3. Classify from file")
    print("4. Classify single text")
    print("5. Exit")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == "1":
        test_examples(model, tokenizer)
    
    elif choice == "2":
        interactive_mode(model, tokenizer)
    
    elif choice == "3":
        file_path = input("Enter file path (one text per line): ").strip()
        output_file = input("Enter output file path (or press Enter to skip): ").strip()
        
        if os.path.exists(file_path):
            classify_from_file(file_path, model, tokenizer, output_file if output_file else None)
        else:
            print(f"File not found: {file_path}")
    
    elif choice == "4":
        text = input("Enter text to classify: ").strip()
        if text:
            result = classify_text(text, model, tokenizer)
            print(f"\nText: {result['text']}")
            print(f"Prediction: {result['predicted_label']} - {result['predicted_label_name']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print("\nAll probabilities:")
            for label, prob in result['all_probabilities'].items():
                print(f"  {label}: {prob:.2%}")
    
    elif choice == "5":
        print("Exiting...")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()