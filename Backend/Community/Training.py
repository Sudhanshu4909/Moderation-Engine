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

import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import json
from datetime import datetime

def load_and_prepare_data(csv_path):
    """
    Load CSV and prepare data for fine-tuning
    Expected columns: text, label, subcategory, source
    """
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Total samples: {len(df)}")
    print(f"\nColumns: {list(df.columns)}")
    
    # Clean data
    df = df.dropna(subset=['text', 'label'])
    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(int)
    
    # Filter valid subcategories
    valid_subcategories = ['community', 'racial_ethnic', 'religious', 'nationality']
    df_filtered = df[df['subcategory'].isin(valid_subcategories)].copy()
    
    print(f"\nSamples after filtering valid subcategories: {len(df_filtered)}")
    
    # Create label mapping for subcategories
    # 0: non-hate, 1: community hate, 2: racial_ethnic hate, 3: religious hate, 4: nationality hate
    def create_fine_grained_label(row):
        if row['label'] == 0:
            return 0  # non-hate
        elif row['subcategory'] == 'community':
            return 1
        elif row['subcategory'] == 'racial_ethnic':
            return 2
        elif row['subcategory'] == 'religious':
            return 3
        elif row['subcategory'] == 'nationality':
            return 4
        else:
            return 0  # default to non-hate for unknown
    
    df_filtered['fine_grained_label'] = df_filtered.apply(create_fine_grained_label, axis=1)
    
    # Label distribution
    print("\n=== LABEL DISTRIBUTION ===")
    print("\nBinary labels (0=non-hate, 1=hate):")
    print(df_filtered['label'].value_counts().sort_index())
    
    print("\nFine-grained labels:")
    label_names = {0: 'non-hate', 1: 'community', 2: 'racial_ethnic', 3: 'religious', 4: 'nationality'}
    for label, count in df_filtered['fine_grained_label'].value_counts().sort_index().items():
        print(f"{label} ({label_names[label]}): {count}")
    
    return df_filtered

def create_datasets(df, test_size=0.2, val_size=0.1):
    """Split data into train, validation, and test sets"""
    
    # First split: train+val and test
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=42,
        stratify=df['fine_grained_label']
    )
    
    # Second split: train and val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size / (1 - test_size),
        random_state=42,
        stratify=train_val_df['fine_grained_label']
    )
    
    print(f"\n=== DATASET SPLIT ===")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df[['text', 'fine_grained_label']].rename(columns={'fine_grained_label': 'labels'}))
    val_dataset = Dataset.from_pandas(val_df[['text', 'fine_grained_label']].rename(columns={'fine_grained_label': 'labels'}))
    test_dataset = Dataset.from_pandas(test_df[['text', 'fine_grained_label']].rename(columns={'fine_grained_label': 'labels'}))
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    return dataset_dict, test_df

def tokenize_function(examples, tokenizer):
    """Tokenize text data"""
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Overall accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Precision, recall, F1 for each class
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def fine_tune_model(csv_path, base_model_path="IMSyPP/hate_speech_en", output_dir="./fine_tuned_hate_speech_model"):
    """
    Fine-tune the hate speech model on community, racial, and ethnic hate speech data
    """
    
    print("="*70)
    print("HATE SPEECH MODEL FINE-TUNING")
    print("="*70)
    
    # Load and prepare data
    df = load_and_prepare_data(csv_path)
    
    # Create datasets
    datasets, test_df = create_datasets(df)
    
    # Load tokenizer and model
    print(f"\nLoading base model: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Model with 5 labels: 0=non-hate, 1=community, 2=racial_ethnic, 3=religious, 4=nationality
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=5,
        ignore_mismatched_sizes=True
    )
    
    # Tokenize datasets
    print("\nTokenizing datasets...")
    tokenized_datasets = datasets.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=['text']
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments - FIXED for compatibility
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        save_total_limit=2,
        warmup_steps=500,
        report_to="none",  # Disable reporting to avoid issues
        use_cpu=False,  # Use GPU if available
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    try:
        trainer.train()
    except Exception as e:
        print(f"\nError during training: {e}")
        print("\nTrying with no_cuda=True...")
        
        # Recreate training args with CPU only
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,  # Reduced batch size for CPU
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            save_total_limit=2,
            warmup_steps=500,
            report_to="none",
            no_cuda=True,  # Force CPU
        )
        
        # Recreate trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        trainer.train()
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    test_results = trainer.evaluate(tokenized_datasets['test'])
    
    print("\nTest Results:")
    for key, value in test_results.items():
        print(f"{key}: {value:.4f}")
    
    # Generate detailed classification report
    print("\n" + "="*70)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*70)
    
    predictions = trainer.predict(tokenized_datasets['test'])
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    label_names = ['non-hate', 'community', 'racial_ethnic', 'religious', 'nationality']
    report = classification_report(true_labels, pred_labels, target_names=label_names, zero_division=0)
    print("\n" + report)
    
    # Save model
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)
    
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save label mapping
    label_mapping = {
        0: "non-hate",
        1: "community",
        2: "racial_ethnic",
        3: "religious",
        4: "nationality"
    }
    
    with open(f"{output_dir}/label_mapping.json", 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    # Save training info
    training_info = {
        "base_model": base_model_path,
        "training_date": datetime.now().isoformat(),
        "num_train_samples": len(tokenized_datasets['train']),
        "num_val_samples": len(tokenized_datasets['validation']),
        "num_test_samples": len(tokenized_datasets['test']),
        "test_accuracy": float(test_results['eval_accuracy']),
        "test_f1": float(test_results['eval_f1']),
        "label_mapping": label_mapping
    }
    
    with open(f"{output_dir}/training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"\nModel saved to: {output_dir}")
    print(f"Label mapping saved to: {output_dir}/label_mapping.json")
    print(f"Training info saved to: {output_dir}/training_info.json")
    
    print("\n" + "="*70)
    print("FINE-TUNING COMPLETE!")
    print("="*70)
    
    return trainer, test_results

def main():
    """Main function"""
    
    # Configuration
    CSV_PATH = "./community_racial_ethnic_hate_speech.csv"
    BASE_MODEL = "IMSyPP/hate_speech_en"
    OUTPUT_DIR = "./fine_tuned_hate_speech_model"
    
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        return
    
    # Run fine-tuning
    trainer, test_results = fine_tune_model(
        csv_path=CSV_PATH,
        base_model_path=BASE_MODEL,
        output_dir=OUTPUT_DIR
    )
    
    print("\nTo use the fine-tuned model:")
    print(f"model = AutoModelForSequenceClassification.from_pretrained('{OUTPUT_DIR}')")
    print(f"tokenizer = AutoTokenizer.from_pretrained('{OUTPUT_DIR}')")

if __name__ == "__main__":
    main()