"""
Test Hate Speech Detection Model
Load trained model and test on example sentences
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
torch.set_num_threads(1)  # ADD THIS LINE
torch.set_num_interop_threads(1)  

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os

# Configuration
MODEL_PATH = "/Users/apple/Downloads/New Moderation/Test/hate_speech_model"  # Path to your trained model DIRECTORY (not the file!)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = 128

# Label names (same as training)
LABELS = [
    'hate_speech',
    'racist',
    'sexist',
    'homophobic',
    'religious',
    'xenophobic',
    'ableist',
]

# Test sentences
TEST_SENTENCES = [
    # Non-hate examples
    "I love spending time with my fucker and ass friends and family.",
    "Everyone deserves equal rights and opportunities.",
    "The weather is beautiful today, perfect for a walk.",
    "Just finished reading an amazing book about history.",
    "Looking forward to the weekend with my partner.",
    
    # Hate speech examples (racist)
    "Those people should go back to their own country.",
    "That race is inferior and doesn't belong here.",
    "Immigrants are ruining our culture and economy.",
    
    # Hate speech examples (sexist)
    "Women belong in the kitchen, not in leadership.",
    "Females are too emotional to make important decisions.",
    "Men are naturally superior to women in every way.",
    
    # Hate speech examples (homophobic)
    "Gay people are destroying our traditional values.",
    "Being transgender is just mental illness.",
    "LGBTQ people shouldn't be allowed to adopt children.",
    
    # Hate speech examples (religious)
    "All Muslims are terrorists and violent.",
    "Jews control all the banks and media.",
    "Christians are brainwashed idiots.",
    
    # Hate speech examples (xenophobic)
    "Refugees are just criminals coming to steal from us.",
    "Foreigners are taking all our jobs away.",
    "Close the borders and deport all illegals now.",
    
    # Hate speech examples (ableist)
    "Disabled people are a burden on society.",
    "Mentally ill people are dangerous and should be locked up.",
    "People with disabilities shouldn't be allowed to have kids.",
    
    # Ambiguous/borderline cases
    "I disagree with that political policy strongly.",
    "That movie was absolutely terrible and offensive.",
    "This restaurant has the worst service I've ever experienced.",
]


def load_model():
    """Load the trained model and tokenizer"""
    print("🔄 Loading model...")
    print(f"   Model path: {MODEL_PATH}")
    print(f"   Device: {DEVICE}")
    
    # Check if model directory exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ ERROR: Model directory not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")
    
    # Check for required files
    config_file = os.path.join(MODEL_PATH, 'config.json')
    if not os.path.exists(config_file):
        print(f"\n❌ ERROR: config.json not found in {MODEL_PATH}")
        raise FileNotFoundError(f"config.json not found in {MODEL_PATH}")
    
    try:
        print("   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, 
            local_files_only=True,
            use_fast=False
        )
        
        print("   Loading model (this may take a moment)...")
        
        # Load with minimal threading
        with torch.no_grad():
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH, 
                local_files_only=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                use_safetensors=True  # Force safetensors if available
            )
        
        print("   Moving to device...")
        model.to(DEVICE)
        model.eval()
        
        print("   ✓ Model loaded successfully\n")
        return model, tokenizer
    
    except Exception as e:
        print(f"\n❌ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        raise


def predict_hate_speech(text, model, tokenizer, threshold=0.5):
    """
    Predict hate speech categories for a given text
    
    Args:
        text: Input text to classify
        model: Trained model
        tokenizer: Tokenizer
        threshold: Confidence threshold (0-1)
    
    Returns:
        dict with predictions and probabilities
    """
    # Tokenize
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    
    # Predict
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get probabilities
        probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()
    
    # Get predictions above threshold
    predictions = []
    for i, (label, prob) in enumerate(zip(LABELS, probs)):
        if prob > threshold:
            predictions.append({
                'label': label,
                'probability': float(prob),
                'confidence': 'high' if prob > 0.8 else 'medium' if prob > 0.6 else 'low'
            })
    
    return {
        'text': text,
        'is_hate_speech': probs[0] > threshold,
        'predictions': predictions,
        'all_probabilities': {label: float(prob) for label, prob in zip(LABELS, probs)}
    }


def print_prediction(result, verbose=False):
    """Pretty print prediction results"""
    text = result['text']
    is_hate = result['is_hate_speech']
    predictions = result['predictions']
    
    # Truncate long text
    display_text = text if len(text) <= 80 else text[:77] + "..."
    
    print(f"\n{'='*80}")
    print(f"Text: {display_text}")
    print(f"{'='*80}")
    
    if not predictions:
        print("✅ NO HATE SPEECH DETECTED")
    else:
        print("⚠️  HATE SPEECH DETECTED")
        print("\nDetected categories:")
        for pred in predictions:
            emoji = "🔴" if pred['confidence'] == 'high' else "🟡" if pred['confidence'] == 'medium' else "🟢"
            print(f"  {emoji} {pred['label']:15s}: {pred['probability']:.1%} ({pred['confidence']} confidence)")
    
    if verbose:
        print("\nAll probabilities:")
        for label, prob in result['all_probabilities'].items():
            bar = "█" * int(prob * 20)
            print(f"  {label:15s}: {prob:.1%} {bar}")


def batch_predict(texts, model, tokenizer, threshold=0.5):
    """Predict on multiple texts efficiently"""
    results = []
    for text in texts:
        result = predict_hate_speech(text, model, tokenizer, threshold)
        results.append(result)
    return results


def get_statistics(results):
    """Get statistics from batch predictions"""
    total = len(results)
    hate_count = sum(1 for r in results if r['is_hate_speech'])
    
    # Count by category
    category_counts = {label: 0 for label in LABELS}
    for result in results:
        for pred in result['predictions']:
            category_counts[pred['label']] += 1
    
    return {
        'total': total,
        'hate_speech': hate_count,
        'non_hate': total - hate_count,
        'hate_percentage': (hate_count / total * 100) if total > 0 else 0,
        'category_counts': category_counts
    }


def main():
    """Main testing function"""
    print("="*80)
    print("  HATE SPEECH DETECTION MODEL - TESTING")
    print("="*80)
    print()
    
    # Load model
    model, tokenizer = load_model()
    
    # Test on all sentences
    print(f"🧪 Testing {len(TEST_SENTENCES)} sentences...")
    print()
    
    results = batch_predict(TEST_SENTENCES, model, tokenizer, threshold=0.5)
    
    # Print individual results
    for i, result in enumerate(results, 1):
        print(f"\n[{i}/{len(TEST_SENTENCES)}]")
        print_prediction(result, verbose=False)
    
    # Print statistics
    stats = get_statistics(results)
    
    print("\n" + "="*80)
    print("📊 STATISTICS")
    print("="*80)
    print(f"Total texts tested: {stats['total']}")
    print(f"Hate speech detected: {stats['hate_speech']} ({stats['hate_percentage']:.1f}%)")
    print(f"Non-hate speech: {stats['non_hate']}")
    
    print("\nDetections by category:")
    for label, count in stats['category_counts'].items():
        if count > 0:
            print(f"  {label:15s}: {count}")
    
    print("\n" + "="*80)
    print()


def test_custom_text():
    """Interactive mode - test your own text"""
    print("="*80)
    print("  INTERACTIVE MODE")
    print("="*80)
    print()
    
    model, tokenizer = load_model()
    
    print("Enter text to test (or 'quit' to exit):\n")
    
    while True:
        text = input(">>> ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not text:
            continue
        
        result = predict_hate_speech(text, model, tokenizer, threshold=0.5)
        print_prediction(result, verbose=True)
        print()


if __name__ == "__main__":
    # Run main test
    main()
    
    # Uncomment below for interactive mode
    # test_custom_text()
    
    print("\n💡 TIP: To test your own sentences, uncomment the test_custom_text() line")
    print("    or modify the TEST_SENTENCES list at the top of this file.\n")