"""
evaluation.py
=============
Evaluation script for Character-Level Name Generation.
This script loads pre-trained models and computes performance metrics.

Usage:
    python evaluation.py
"""

import os
import pickle
import torch
import random
import numpy as np
from p2_models import CharVanillaRNN, CharBLSTM, CharRNNAttention
from p2_train_eval import (
    generate_names, 
    evaluate_model, 
    qualitative_analysis, 
    comparison_summary,
    load_names
)

# Configuration
SEED = 42
EMBEDDING_DIM = 32
HIDDEN_SIZE = 128
ATTENTION_SIZE = 64
TEMPERATURE = 0.8
NUM_GENERATE = 200

# Set random seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cpu")

    # 1. Load Vocabulary
    vocab_path = os.path.join(script_dir, "vocab_p2.pkl")
    if not os.path.exists(vocab_path):
        print(f"Error: Vocabulary file not found at {vocab_path}")
        print("Please run p2_train_eval.py first to train and save models.")
        return

    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
    
    char2idx = vocab_data['char2idx']
    idx2char = vocab_data['idx2char']
    vocab_size = len(char2idx)
    print(f"Loaded vocabulary with {vocab_size} characters.")

    # 2. Load Training Names (for Novelty computation)
    names_path = os.path.join(script_dir, "TrainingNames.txt")
    if os.path.exists(names_path):
        training_names = load_names(names_path)
    else:
        print(f"Warning: TrainingNames.txt not found at {names_path}. Novelty metrics will be 100%.")
        training_names = []

    # 3. Initialize Models
    models_to_load = {
        "Vanilla RNN": CharVanillaRNN(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE),
        "BLSTM": CharBLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE),
        "RNN + Attention": CharRNNAttention(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE),
    }

    results = {}

    for model_name, model in models_to_load.items():
        # Load weights
        model_file = f"{model_name.replace(' ', '_').replace('+', 'plus').lower()}.pkl"
        model_path = os.path.join(script_dir, model_file)
        
        if not os.path.exists(model_path):
            print(f"Skipping {model_name}: Weights file {model_file} not found.")
            continue

        print(f"\nLoading {model_name} from {model_file}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Generate names
        generated = generate_names(
            model, model_name, char2idx, idx2char, device,
            num_names=NUM_GENERATE, temperature=TEMPERATURE
        )

        # Quantitative Evaluation
        metrics = evaluate_model(generated, training_names, model_name)
        results[model_name] = metrics

        # Qualitative Analysis
        qualitative_analysis(generated, model_name)

    # 4. Cross-model comparison
    if results:
        comparison_summary(results)

if __name__ == "__main__":
    main()
