import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.set_num_threads(1)

from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

MODEL_PATH = "/Users/apple/Downloads/New Moderation/Test/hate_speech_model"

print("Loading config...")
config = AutoConfig.from_pretrained(MODEL_PATH)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

print("Creating empty model...")
model = AutoModelForSequenceClassification.from_config(config)

print("Loading weights from safetensors...")
state_dict = load_file(f"{MODEL_PATH}/model.safetensors")

print("Loading state dict into model...")
model.load_state_dict(state_dict)

print("SUCCESS! Model loaded!")
model.eval()

# Test it
text = "This is a test"
inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)
print(f"Test output shape: {outputs.logits.shape}")
print("✅ Model is working!")