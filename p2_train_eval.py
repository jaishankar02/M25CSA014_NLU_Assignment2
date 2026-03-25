"""
p2_train_eval.py
================
Training and Evaluation script for Character-Level Name Generation.

This script:
    1. Loads the TrainingNames.txt dataset and builds a character vocabulary
    2. Creates training sequences with teacher forcing
    3. Trains all three models (Vanilla RNN, BLSTM, RNN with Attention)
    4. Generates new names from each trained model
    5. Computes quantitative metrics:
       - Novelty Rate: % of generated names NOT in the training set
       - Diversity: # unique generated names / total generated names
    6. Performs qualitative analysis: prints samples, discusses realism

Usage:
    python p2_train_eval.py

Note: All models use teacher forcing during training and autoregressive
      (character-by-character) generation during inference.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from collections import Counter

# Import our from-scratch models
from p2_models import CharVanillaRNN, CharBLSTM, CharRNNAttention, print_model_summary

# ============================================================
# Configuration
# ============================================================
SEED = 42
EMBEDDING_DIM = 32       # Character embedding dimension
HIDDEN_SIZE = 128        # RNN hidden state size
ATTENTION_SIZE = 64      # Attention projection size (for model 3)
NUM_EPOCHS = 100         # Training epochs
BATCH_SIZE = 64          # Batch size
LEARNING_RATE = 0.003    # Learning rate
TEMPERATURE = 0.8        # Sampling temperature for generation
NUM_GENERATE = 200       # Number of names to generate per model
MAX_GEN_LEN = 30         # Maximum length for generated names

# Special tokens
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'     # Start of sequence
EOS_TOKEN = '<EOS>'     # End of sequence

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Data Preparation
# ============================================================

def load_names(filepath):
    """
    Load names from the training file.
    Each line contains one name (first + surname).
    
    Returns:
        names (list of str): List of names.
    """
    names = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            if name:
                names.append(name)
    return names


def build_char_vocab(names):
    """
    Build a character-level vocabulary from the training names.

    Returns:
        char2idx (dict): Character to index mapping.
        idx2char (dict): Index to character mapping.
        vocab_size (int): Total vocabulary size (chars + special tokens).
    """
    # Collect all unique characters
    all_chars = set()
    for name in names:
        all_chars.update(name)

    # Sort for deterministic ordering
    chars = sorted(all_chars)

    # Create mappings with special tokens
    char2idx = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2}
    for i, c in enumerate(chars, start=3):
        char2idx[c] = i

    idx2char = {v: k for k, v in char2idx.items()}
    vocab_size = len(char2idx)

    print(f"  Character vocabulary size: {vocab_size}")
    print(f"  Characters: {chars}")

    return char2idx, idx2char, vocab_size


def encode_name(name, char2idx, add_sos=True, add_eos=True):
    """
    Encode a name string into a list of character indices.

    Args:
        name (str): The name to encode.
        char2idx (dict): Character-to-index mapping.
        add_sos (bool): Whether to prepend SOS token.
        add_eos (bool): Whether to append EOS token.

    Returns:
        encoded (list of int): Encoded character indices.
    """
    encoded = []
    if add_sos:
        encoded.append(char2idx[SOS_TOKEN])
    for c in name:
        encoded.append(char2idx.get(c, char2idx[PAD_TOKEN]))
    if add_eos:
        encoded.append(char2idx[EOS_TOKEN])
    return encoded


def create_batches(names, char2idx, batch_size):
    """
    Create padded batches for training.

    Each training example is:
        Input:  [SOS, c1, c2, ..., cn]
        Target: [c1, c2, ..., cn, EOS]

    Returns:
        batches: list of (input_tensor, target_tensor) pairs.
    """
    # Encode all names
    encoded = [encode_name(name, char2idx) for name in names]

    # Shuffle
    random.shuffle(encoded)

    batches = []
    for i in range(0, len(encoded), batch_size):
        batch = encoded[i:i + batch_size]

        # Find max length in this batch
        max_len = max(len(seq) for seq in batch)

        # Pad sequences
        inputs = []
        targets = []
        for seq in batch:
            # Input: all tokens except last (SOS to last char)
            inp = seq[:-1]
            # Target: all tokens except first (first char to EOS)
            tgt = seq[1:]

            # Pad
            pad_len = (max_len - 1) - len(inp)
            inp = inp + [char2idx[PAD_TOKEN]] * pad_len
            tgt = tgt + [char2idx[PAD_TOKEN]] * pad_len

            inputs.append(inp)
            targets.append(tgt)

        input_tensor = torch.tensor(inputs, dtype=torch.long)
        target_tensor = torch.tensor(targets, dtype=torch.long)
        batches.append((input_tensor, target_tensor))

    return batches


# ============================================================
# Training
# ============================================================

def train_model(model, model_name, names, char2idx, idx2char,
                num_epochs, batch_size, lr, device):
    """
    Train a character-level model using teacher forcing.

    Args:
        model: The PyTorch model to train.
        model_name: Name string for display.
        names: List of training name strings.
        char2idx, idx2char: Vocabulary mappings.
        num_epochs, batch_size, lr: Training hyperparameters.
        device: torch device.

    Returns:
        model: Trained model.
    """
    print(f"\n{'=' * 60}")
    print(f"TRAINING: {model_name}")
    print(f"{'=' * 60}")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx[PAD_TOKEN])

    # Print architecture summary
    print_model_summary(model, model_name)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        # Create fresh batches each epoch (with reshuffling)
        batches = create_batches(names, char2idx, batch_size)

        for input_batch, target_batch in batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            # Forward pass
            logits, _ = model(input_batch)

            # Reshape for cross-entropy loss
            # logits: (batch_size, seq_len, vocab_size)
            # target: (batch_size, seq_len)
            loss = criterion(logits.view(-1, model.vocab_size), target_batch.view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # Log periodically
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs} — Loss: {avg_loss:.4f}")

    print(f"  Training complete!")
    return model


# ============================================================
# Generation
# ============================================================

def generate_name(model, char2idx, idx2char, device,
                  temperature=0.8, max_len=30, use_forward_only=False):
    """
    Generate a single name autoregressively.

    Starts with the SOS token and samples characters one at a time
    until EOS is produced or max_len is reached.

    Args:
        model: Trained model.
        char2idx, idx2char: Vocabulary mappings.
        device: torch device.
        temperature: Sampling temperature (lower = more conservative).
        max_len: Maximum number of characters to generate.
        use_forward_only: If True, use forward-only mode (for BLSTM).

    Returns:
        generated_name (str): The generated name.
    """
    model.eval()
    with torch.no_grad():
        # Start with SOS token
        current_char = torch.tensor([[char2idx[SOS_TOKEN]]], dtype=torch.long, device=device)
        hidden = None
        generated_chars = []

        for _ in range(max_len):
            # Forward pass for one character
            if use_forward_only and hasattr(model, 'generate_forward_only'):
                logits, hidden = model.generate_forward_only(current_char, hidden)
            else:
                if isinstance(model, CharVanillaRNN):
                    logits, hidden = model(current_char, hidden)
                elif isinstance(model, CharRNNAttention):
                    logits, hidden = model(current_char, hidden)
                else:
                    logits, hidden = model(current_char, hidden)

            # Get logits for the last time step
            logits = logits[:, -1, :] / temperature

            # Sample from the distribution
            probs = torch.softmax(logits, dim=-1)
            next_char_idx = torch.multinomial(probs, num_samples=1).item()

            # Check for EOS
            if next_char_idx == char2idx[EOS_TOKEN]:
                break

            # Skip PAD and SOS tokens
            if next_char_idx == char2idx[PAD_TOKEN] or next_char_idx == char2idx[SOS_TOKEN]:
                continue

            generated_chars.append(idx2char[next_char_idx])
            current_char = torch.tensor([[next_char_idx]], dtype=torch.long, device=device)

            # Reset hidden for attention model to avoid accumulating too many states
            if isinstance(model, CharRNNAttention):
                # For attention model, we need to rebuild the sequence each time
                # Build full sequence so far and re-process
                full_seq = [char2idx[SOS_TOKEN]] + [char2idx.get(c, 0) for c in generated_chars]
                full_tensor = torch.tensor([full_seq], dtype=torch.long, device=device)
                logits_full, hidden = model(full_tensor)
                # hidden is now the final state after processing the full sequence

    return ''.join(generated_chars)


def generate_names(model, model_name, char2idx, idx2char, device,
                   num_names=200, temperature=0.8):
    """
    Generate multiple names from a trained model.

    Args:
        model: Trained model.
        model_name: Model name for display.
        char2idx, idx2char: Vocabulary mappings.
        device: torch device.
        num_names: Number of names to generate.
        temperature: Sampling temperature.

    Returns:
        generated (list of str): List of generated names.
    """
    print(f"\nGenerating {num_names} names from {model_name}...")

    use_forward = isinstance(model, CharBLSTM)
    generated = []
    for _ in range(num_names):
        name = generate_name(model, char2idx, idx2char, device,
                             temperature=temperature, max_len=MAX_GEN_LEN,
                             use_forward_only=use_forward)
        if name:  # Only keep non-empty names
            generated.append(name)

    print(f"  Successfully generated {len(generated)} names")
    return generated


# ============================================================
# Quantitative Evaluation
# ============================================================

def evaluate_model(generated_names, training_names, model_name):
    """
    Compute quantitative metrics for generated names.

    Metrics:
        1. Novelty Rate: Percentage of generated names that do NOT appear
           in the training set. Higher is better — indicates the model is
           creating new names rather than memorizing.

        2. Diversity: Number of unique generated names divided by total
           generated names. Higher is better — indicates the model produces
           varied outputs rather than repeating the same names.

    Args:
        generated_names: List of generated name strings.
        training_names: List of training name strings.
        model_name: Name of the model for display.
    """
    print(f"\n{'=' * 60}")
    print(f"QUANTITATIVE EVALUATION: {model_name}")
    print(f"{'=' * 60}")

    # Normalize names for comparison (lowercase)
    training_set = set(name.lower().strip() for name in training_names)
    generated_lower = [name.lower().strip() for name in generated_names]

    total_generated = len(generated_names)
    if total_generated == 0:
        print("  No names generated — skipping evaluation.")
        return

    # Novelty Rate
    novel_names = [name for name in generated_lower if name not in training_set]
    novelty_rate = len(novel_names) / total_generated * 100

    # Diversity
    unique_names = set(generated_lower)
    diversity = len(unique_names) / total_generated

    print(f"  Total names generated:    {total_generated}")
    print(f"  Novel names (not in train): {len(novel_names)} / {total_generated}")
    print(f"  Novelty Rate:             {novelty_rate:.1f}%")
    print(f"  Unique names:             {len(unique_names)} / {total_generated}")
    print(f"  Diversity:                {diversity:.4f}")
    print()

    return {
        'total': total_generated,
        'novel': len(novel_names),
        'novelty_rate': novelty_rate,
        'unique': len(unique_names),
        'diversity': diversity,
    }


# ============================================================
# Qualitative Analysis
# ============================================================

def qualitative_analysis(generated_names, model_name):
    """
    Perform qualitative analysis on generated names.
    Prints representative samples and discusses patterns.

    Args:
        generated_names: List of generated name strings.
        model_name: Name of the model.
    """
    print(f"\n{'=' * 60}")
    print(f"QUALITATIVE ANALYSIS: {model_name}")
    print(f"{'=' * 60}")

    if not generated_names:
        print("  No names to analyze.")
        return

    # Print 20 representative samples
    print(f"\n  Representative Generated Names (20 samples):")
    print(f"  {'-' * 40}")
    sample_size = min(20, len(generated_names))
    samples = random.sample(generated_names, sample_size)
    for i, name in enumerate(samples, 1):
        print(f"    {i:2d}. {name}")

    # Length statistics
    lengths = [len(name) for name in generated_names]
    avg_len = sum(lengths) / len(lengths)
    print(f"\n  Length Statistics:")
    print(f"    Average length:  {avg_len:.1f} characters")
    print(f"    Min length:      {min(lengths)} characters")
    print(f"    Max length:      {max(lengths)} characters")

    # Check for common patterns / failure modes
    print(f"\n  Common Failure Modes:")
    too_short = [n for n in generated_names if len(n) < 3]
    too_long = [n for n in generated_names if len(n) > 25]
    has_space = [n for n in generated_names if ' ' in n]

    print(f"    Names too short (<3 chars): {len(too_short)}")
    if too_short:
        print(f"      Examples: {too_short[:5]}")
    print(f"    Names too long (>25 chars): {len(too_long)}")
    if too_long:
        print(f"      Examples: {too_long[:3]}")
    print(f"    Names with space (full names): {len(has_space)}")

    # Character frequency analysis
    all_chars = Counter(''.join(generated_names).lower())
    print(f"\n  Top 10 Most Frequent Characters:")
    for char, count in all_chars.most_common(10):
        display_char = repr(char) if char == ' ' else char
        print(f"    '{display_char}': {count}")

    print()


# ============================================================
# Comparison Summary
# ============================================================

def comparison_summary(results):
    """
    Print a comparison table of all models.

    Args:
        results: dict of {model_name: metrics_dict}
    """
    print(f"\n{'=' * 60}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'=' * 60}")
    print(f"\n  {'Model':<25} {'Novelty %':>10} {'Diversity':>10} {'Generated':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    for name, metrics in results.items():
        if metrics:
            print(f"  {name:<25} {metrics['novelty_rate']:>9.1f}% {metrics['diversity']:>10.4f} {metrics['total']:>10}")

    print(f"\nInterpretation:")
    print(f"  - Higher Novelty = model creates more original names (less memorization)")
    print(f"  - Higher Diversity = model generates more varied names (less repetition)")
    print(f"  - The BLSTM typically captures longer-range dependencies due to bidirectional context")
    print(f"  - The Attention model can focus on relevant parts of the sequence for generation")
    print()


# ============================================================
# Main
# ============================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    names_path = os.path.join(script_dir, "TrainingNames.txt")
    device = torch.device("cpu")  # Use CPU since we're working with small models

    # Load dataset
    print("Loading training names...")
    names = load_names(names_path)
    print(f"  Loaded {len(names)} names")

    # Build character vocabulary
    print("\nBuilding character vocabulary...")
    char2idx, idx2char, vocab_size = build_char_vocab(names)

    # Save vocabulary
    vocab_save_path = os.path.join(script_dir, "vocab_p2.pkl")
    with open(vocab_save_path, 'wb') as f:
        pickle.dump({'char2idx': char2idx, 'idx2char': idx2char}, f)
    print(f"  Vocabulary saved to: {vocab_save_path}")

    # ----------------------------------------------------------------
    # Initialize all three models
    # ----------------------------------------------------------------
    models = {
        "Vanilla RNN": CharVanillaRNN(
            vocab_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            dropout=0.1,
        ),
        "BLSTM": CharBLSTM(
            vocab_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            dropout=0.1,
        ),
        "RNN + Attention": CharRNNAttention(
            vocab_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            attention_size=ATTENTION_SIZE,
            dropout=0.1,
        ),
    }

    # Print learning rate and other shared hyperparameters
    print(f"\n{'=' * 60}")
    print("SHARED TRAINING HYPERPARAMETERS")
    print(f"{'=' * 60}")
    print(f"  Embedding dimension:  {EMBEDDING_DIM}")
    print(f"  Hidden size:          {HIDDEN_SIZE}")
    print(f"  Learning rate:        {LEARNING_RATE}")
    print(f"  Batch size:           {BATCH_SIZE}")
    print(f"  Epochs:               {NUM_EPOCHS}")
    print(f"  Temperature (gen):    {TEMPERATURE}")
    print(f"  Device:               {device}")

    # ----------------------------------------------------------------
    # Train and evaluate each model
    # ----------------------------------------------------------------
    all_results = {}
    all_generated = {}

    for model_name, model in models.items():
        # Train
        trained_model = train_model(
            model, model_name, names, char2idx, idx2char,
            NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, device
        )

        # Generate names
        generated = generate_names(
            trained_model, model_name, char2idx, idx2char, device,
            num_names=NUM_GENERATE, temperature=TEMPERATURE
        )
        all_generated[model_name] = generated

        # Quantitative evaluation
        metrics = evaluate_model(generated, names, model_name)
        all_results[model_name] = metrics

        # Qualitative analysis
        qualitative_analysis(generated, model_name)

        # Save generated names to file
        gen_file = os.path.join(script_dir, f"generated_{model_name.replace(' ', '_').replace('+', 'plus').lower()}.txt")
        with open(gen_file, 'w', encoding='utf-8') as f:
            for name in generated:
                f.write(name + '\n')
        print(f"  Generated names saved to: {gen_file}")

        # Save model state dict
        model_save_path = os.path.join(script_dir, f"{model_name.replace(' ', '_').replace('+', 'plus').lower()}.pkl")
        torch.save(trained_model.state_dict(), model_save_path)
        print(f"  Model saved to: {model_save_path}")

    # ----------------------------------------------------------------
    # Cross-model comparison
    # ----------------------------------------------------------------
    comparison_summary(all_results)

    # Save summary metrics
    print("\nAll models trained and evaluated successfully!")
    print(f"Check the generated_*.txt files for full lists of generated names.")


if __name__ == "__main__":
    main()
