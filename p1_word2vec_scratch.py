"""
p1_word2vec_scratch.py
======================
Word2Vec implementation FROM SCRATCH using PyTorch.

This script implements both CBOW and Skip-gram models with Negative Sampling,
trains them on the IIT Jodhpur cleaned corpus, and performs:
    - Semantic analysis: Top-5 nearest neighbors for selected words
    - Analogy experiments (e.g., UG:BTech :: PG:?)
    - PCA and t-SNE visualizations of learned embeddings

Architecture Details:
    CBOW (Continuous Bag of Words):
        - Input: one-hot encoded context words (window_size * 2 words)
        - Embedding layer: maps each context word to a dense vector
        - The context embeddings are averaged to predict the center word
        - Output: score for each word in vocabulary via negative sampling

    Skip-gram:
        - Input: one-hot encoded center word
        - Embedding layer: maps center word to a dense vector
        - Output: predict each context word individually
        - Uses negative sampling for efficient training

    Negative Sampling:
        - Instead of computing softmax over the entire vocabulary,
          we randomly sample K negative (incorrect) words and use
          binary logistic regression to distinguish true context words
          from noise samples.

Usage:
    python p1_word2vec_scratch.py
"""

import os
import random
import math
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Visualization imports
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ============================================================
# HYPERPARAMETERS — experiment with these as required
# ============================================================
EMBEDDING_DIM = 100       # Dimensionality of word embeddings
WINDOW_SIZE = 5           # Context window size (words on each side)
NUM_NEGATIVE = 5          # Number of negative samples per positive pair
MIN_COUNT = 2             # Minimum word frequency to include in vocabulary
LEARNING_RATE = 0.005     # Learning rate for optimization
NUM_EPOCHS = 15           # Number of training epochs
BATCH_SIZE = 512          # Batch size for training
SUBSAMPLE_THRESHOLD = 1e-3  # Subsampling threshold for frequent words


# ============================================================
# Data Preparation
# ============================================================

def load_corpus(filepath):
    """
    Load the cleaned corpus. Each line is a space-separated sentence.
    Returns a list of sentences, where each sentence is a list of tokens.
    """
    sentences = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                sentences.append(tokens)
    return sentences


def build_vocab(sentences, min_count=2):
    """
    Build vocabulary from sentences.
    Only includes words that appear at least min_count times.
    
    Returns:
        word2idx (dict): Mapping from word to index.
        idx2word (dict): Mapping from index to word.
        word_counts (Counter): Frequency of each word in vocabulary.
    """
    # Count all word frequencies
    counter = Counter()
    for sent in sentences:
        counter.update(sent)

    # Filter by minimum count
    filtered = {w: c for w, c in counter.items() if c >= min_count}

    # Create mappings
    word2idx = {}
    idx2word = {}
    for i, (word, _) in enumerate(sorted(filtered.items(), key=lambda x: -x[1])):
        word2idx[word] = i
        idx2word[i] = word

    word_counts = Counter({w: c for w, c in filtered.items()})

    print(f"  Vocabulary size (min_count={min_count}): {len(word2idx)}")
    return word2idx, idx2word, word_counts


def compute_discard_prob(word_counts, threshold=1e-3):
    """
    Compute subsampling discard probabilities for frequent words.
    This helps reduce the impact of very frequent words like 'the', 'and', etc.

    The probability of keeping a word w is:
        P(keep w) = sqrt(threshold / freq(w))
    
    Args:
        word_counts: Counter with word frequencies.
        threshold: Subsampling threshold.
    
    Returns:
        discard_prob (dict): Probability of discarding each word.
    """
    total = sum(word_counts.values())
    discard_prob = {}
    for word, count in word_counts.items():
        freq = count / total
        # Probability of keeping the word
        keep_prob = min(1.0, math.sqrt(threshold / freq) + threshold / freq)
        discard_prob[word] = 1.0 - keep_prob
    return discard_prob


def build_noise_distribution(word_counts, power=0.75):
    """
    Build the noise distribution for negative sampling.
    Each word's probability is proportional to count^0.75 (as in the original Word2Vec paper).

    Args:
        word_counts: Counter with word frequencies.
        power: Exponent to raise counts to (0.75 as in original paper).

    Returns:
        noise_dist (torch.Tensor): Probability distribution over vocabulary.
    """
    counts = np.array([word_counts[w] for w in sorted(word_counts.keys(),
                       key=lambda x: -word_counts[x])], dtype=np.float64)
    powered = np.power(counts, power)
    noise_dist = torch.from_numpy(powered / powered.sum()).float()
    return noise_dist


# ============================================================
# Dataset Classes
# ============================================================

class SkipGramDataset(Dataset):
    """
    Dataset for Skip-gram model.
    For each center word, generates (center, context) positive pairs.
    Negative samples are drawn during training.
    """
    def __init__(self, sentences, word2idx, window_size, discard_prob):
        self.pairs = []  # List of (center_idx, context_idx)
        
        # Generate all (center, context) pairs with subsampling
        for sent in sentences:
            # Filter sentence through vocabulary and subsampling
            indices = []
            for w in sent:
                if w in word2idx:
                    # Subsample: randomly discard frequent words
                    if random.random() >= discard_prob.get(w, 0):
                        indices.append(word2idx[w])

            # Generate skip-gram pairs
            for i, center in enumerate(indices):
                # Random window size for each position (as in original Word2Vec)
                actual_window = random.randint(1, window_size)
                start = max(0, i - actual_window)
                end = min(len(indices), i + actual_window + 1)
                for j in range(start, end):
                    if j != i:
                        self.pairs.append((center, indices[j]))

        print(f"  Skip-gram training pairs: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long)


class CBOWDataset(Dataset):
    """
    Dataset for CBOW model.
    For each center word, the context is the surrounding words within the window.
    """
    def __init__(self, sentences, word2idx, window_size, discard_prob):
        self.data = []  # List of (context_indices, center_idx)
        
        for sent in sentences:
            # Filter through vocab and subsampling
            indices = []
            for w in sent:
                if w in word2idx:
                    if random.random() >= discard_prob.get(w, 0):
                        indices.append(word2idx[w])

            # Generate CBOW samples
            for i in range(len(indices)):
                actual_window = random.randint(1, window_size)
                start = max(0, i - actual_window)
                end = min(len(indices), i + actual_window + 1)
                context = [indices[j] for j in range(start, end) if j != i]
                if context:
                    self.data.append((context, indices[i]))

        print(f"  CBOW training samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, center = self.data[idx]
        return context, center


def cbow_collate_fn(batch):
    """
    Custom collate function for CBOW that handles variable-length context windows.
    Pads context sequences to the maximum length in the batch.
    """
    contexts, centers = zip(*batch)
    max_len = max(len(c) for c in contexts)
    
    # Pad context sequences and create masks
    padded_contexts = []
    masks = []
    for ctx in contexts:
        pad_len = max_len - len(ctx)
        padded_contexts.append(ctx + [0] * pad_len)
        masks.append([1.0] * len(ctx) + [0.0] * pad_len)

    return (torch.tensor(padded_contexts, dtype=torch.long),
            torch.tensor(masks, dtype=torch.float),
            torch.tensor(centers, dtype=torch.long))


# ============================================================
# Model Definitions
# ============================================================

class SkipGramNegSampling(nn.Module):
    """
    Skip-gram model with Negative Sampling.

    Architecture:
        - Input embedding matrix (V x D): maps center word to dense vector
        - Output embedding matrix (V x D): maps context/negative word to dense vector
        - Score = dot product between center embedding and context embedding
        - Loss: Binary cross-entropy (positive pairs = 1, negative pairs = 0)

    Parameters:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimensionality of the word embeddings.
    """
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramNegSampling, self).__init__()
        # Center word embeddings (the final word vectors we use)
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Context word embeddings (used during training only)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Initialize weights with small random values (as in original Word2Vec)
        initrange = 0.5 / embedding_dim
        self.center_embeddings.weight.data.uniform_(-initrange, initrange)
        self.context_embeddings.weight.data.zero_()

    def forward(self, center_words, context_words, negative_words):
        """
        Forward pass computing the loss for a batch.

        Args:
            center_words: (batch_size,) — indices of center words
            context_words: (batch_size,) — indices of positive context words
            negative_words: (batch_size, num_neg) — indices of negative samples

        Returns:
            loss: scalar tensor — average negative sampling loss
        """
        # Get embeddings: (batch_size, embedding_dim)
        center_emb = self.center_embeddings(center_words)
        context_emb = self.context_embeddings(context_words)
        neg_emb = self.context_embeddings(negative_words)

        # Positive score: dot product between center and context
        # (batch_size,)
        pos_score = torch.sum(center_emb * context_emb, dim=1)
        pos_loss = torch.nn.functional.logsigmoid(pos_score)

        # Negative score: dot product between center and each negative sample
        # center_emb: (batch_size, 1, embedding_dim)
        # neg_emb: (batch_size, num_neg, embedding_dim)
        neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze(2)
        neg_loss = torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)

        # Total loss: maximize log-likelihood => minimize negative of sum
        loss = -(pos_loss + neg_loss).mean()
        return loss


class CBOWNegSampling(nn.Module):
    """
    CBOW (Continuous Bag of Words) model with Negative Sampling.

    Architecture:
        - Input embedding matrix (V x D): maps each context word to a dense vector
        - Context representation: average of all context word embeddings
        - Output embedding matrix (V x D): maps center word to a dense vector
        - Score = dot product between averaged context and center embedding
        - Loss: Binary cross-entropy via negative sampling

    Parameters:
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimensionality of word embeddings.
    """
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWNegSampling, self).__init__()
        # Context word embeddings (input side)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Center word embeddings (output side, used for scoring)
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Initialization
        initrange = 0.5 / embedding_dim
        self.context_embeddings.weight.data.uniform_(-initrange, initrange)
        self.center_embeddings.weight.data.zero_()

    def forward(self, context_words, context_mask, center_words, negative_words):
        """
        Forward pass for CBOW with negative sampling.

        Args:
            context_words: (batch_size, max_context_len) — padded context word indices
            context_mask: (batch_size, max_context_len) — 1 for real tokens, 0 for padding
            center_words: (batch_size,) — center word indices
            negative_words: (batch_size, num_neg) — negative sample indices

        Returns:
            loss: scalar tensor
        """
        # Get context embeddings: (batch_size, max_context_len, embedding_dim)
        ctx_emb = self.context_embeddings(context_words)

        # Apply mask and average: (batch_size, embedding_dim)
        mask = context_mask.unsqueeze(2)  # (batch_size, max_context_len, 1)
        ctx_emb = (ctx_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Center embeddings: (batch_size, embedding_dim)
        center_emb = self.center_embeddings(center_words)
        neg_emb = self.center_embeddings(negative_words)

        # Positive score
        pos_score = torch.sum(ctx_emb * center_emb, dim=1)
        pos_loss = torch.nn.functional.logsigmoid(pos_score)

        # Negative scores
        neg_score = torch.bmm(neg_emb, ctx_emb.unsqueeze(2)).squeeze(2)
        neg_loss = torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)

        loss = -(pos_loss + neg_loss).mean()
        return loss


# ============================================================
# Training Functions
# ============================================================

def train_skipgram(sentences, word2idx, idx2word, word_counts, noise_dist,
                   embedding_dim, window_size, num_negative, num_epochs,
                   batch_size, lr):
    """
    Train the Skip-gram model with negative sampling.

    Returns:
        embeddings (np.ndarray): Trained word embeddings matrix (vocab_size x embedding_dim).
    """
    print("\n" + "=" * 60)
    print("Training Skip-gram Model (From Scratch)")
    print("=" * 60)
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Window size: {window_size}")
    print(f"  Negative samples: {num_negative}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")

    vocab_size = len(word2idx)

    # Compute subsampling probabilities
    discard_prob = compute_discard_prob(word_counts, SUBSAMPLE_THRESHOLD)

    # Create dataset
    dataset = SkipGramDataset(sentences, word2idx, window_size, discard_prob)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Initialize model
    model = SkipGramNegSampling(vocab_size, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for center, context in dataloader:
            # Draw negative samples from noise distribution
            neg_samples = torch.multinomial(noise_dist, center.size(0) * num_negative,
                                            replacement=True)
            neg_samples = neg_samples.view(center.size(0), num_negative)

            # Forward pass
            loss = model(center, context, neg_samples)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{num_epochs} — Loss: {avg_loss:.4f}")

    # Extract learned embeddings (from center embedding matrix)
    embeddings = model.center_embeddings.weight.data.numpy()
    print(f"  Training complete. Embedding shape: {embeddings.shape}")
    return embeddings


def train_cbow(sentences, word2idx, idx2word, word_counts, noise_dist,
               embedding_dim, window_size, num_negative, num_epochs,
               batch_size, lr):
    """
    Train the CBOW model with negative sampling.

    Returns:
        embeddings (np.ndarray): Trained word embeddings matrix (vocab_size x embedding_dim).
    """
    print("\n" + "=" * 60)
    print("Training CBOW Model (From Scratch)")
    print("=" * 60)
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Window size: {window_size}")
    print(f"  Negative samples: {num_negative}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")

    vocab_size = len(word2idx)

    # Compute subsampling probabilities
    discard_prob = compute_discard_prob(word_counts, SUBSAMPLE_THRESHOLD)

    # Create dataset
    dataset = CBOWDataset(sentences, word2idx, window_size, discard_prob)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           collate_fn=cbow_collate_fn, num_workers=0)

    # Initialize model
    model = CBOWNegSampling(vocab_size, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for context_words, context_mask, center_words in dataloader:
            # Draw negative samples
            neg_samples = torch.multinomial(noise_dist, center_words.size(0) * num_negative,
                                            replacement=True)
            neg_samples = neg_samples.view(center_words.size(0), num_negative)

            # Forward pass
            loss = model(context_words, context_mask, center_words, neg_samples)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{num_epochs} — Loss: {avg_loss:.4f}")

    # Extract embeddings from context embedding matrix (CBOW convention)
    embeddings = model.context_embeddings.weight.data.numpy()
    print(f"  Training complete. Embedding shape: {embeddings.shape}")
    return embeddings


# ============================================================
# Evaluation Functions
# ============================================================

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def find_nearest_neighbors(word, embeddings, word2idx, idx2word, top_k=5):
    """
    Find the top-K nearest neighbors of a word using cosine similarity.

    Args:
        word (str): The query word.
        embeddings (np.ndarray): Word embedding matrix.
        word2idx (dict): Word to index mapping.
        idx2word (dict): Index to word mapping.
        top_k (int): Number of neighbors to return.

    Returns:
        neighbors (list of tuples): [(word, similarity), ...]
    """
    if word not in word2idx:
        print(f"  '{word}' not found in vocabulary.")
        return []

    word_idx = word2idx[word]
    word_vec = embeddings[word_idx]

    # Compute cosine similarity with all other words
    # Normalize all embeddings for efficient computation
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    normalized = embeddings / norms
    word_normalized = word_vec / max(np.linalg.norm(word_vec), 1e-10)

    similarities = normalized @ word_normalized

    # Get top-K (excluding the word itself)
    top_indices = np.argsort(similarities)[::-1]
    neighbors = []
    for idx in top_indices:
        if idx != word_idx:
            neighbors.append((idx2word[idx], similarities[idx]))
        if len(neighbors) == top_k:
            break

    return neighbors


def analogy(word_a, word_b, word_c, embeddings, word2idx, idx2word, top_k=3):
    """
    Perform word analogy: A is to B as C is to ?
    Uses the vector offset method: result ≈ embedding(B) - embedding(A) + embedding(C)

    Args:
        word_a, word_b, word_c: Words forming the analogy A:B :: C:?
        embeddings: Word embedding matrix.
        word2idx, idx2word: Vocabulary mappings.
        top_k: Number of results to return.

    Returns:
        results (list of tuples): [(word, similarity), ...]
    """
    for w in [word_a, word_b, word_c]:
        if w not in word2idx:
            print(f"  '{w}' not in vocabulary, cannot compute analogy.")
            return []

    # Compute the analogy vector
    vec = embeddings[word2idx[word_b]] - embeddings[word2idx[word_a]] + embeddings[word2idx[word_c]]

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms
    vec_normalized = vec / max(np.linalg.norm(vec), 1e-10)

    similarities = normalized @ vec_normalized

    # Exclude input words
    exclude = {word2idx[word_a], word2idx[word_b], word2idx[word_c]}
    top_indices = np.argsort(similarities)[::-1]
    results = []
    for idx in top_indices:
        if idx not in exclude:
            results.append((idx2word[idx], similarities[idx]))
        if len(results) == top_k:
            break

    return results


def semantic_analysis(embeddings, word2idx, idx2word, model_name):
    """
    Perform semantic analysis: nearest neighbors and analogies.
    Prints results in a formatted manner for the report.
    """
    print(f"\n{'=' * 60}")
    print(f"SEMANTIC ANALYSIS — {model_name}")
    print(f"{'=' * 60}")

    # Task 3.1: Top 5 nearest neighbors for specified words
    query_words = ["research", "student", "phd", "exam"]
    # Add 'evan' only if it exists, otherwise try alternatives
    if "evan" in word2idx:
        query_words.insert(3, "evan")
    elif "evaluation" in word2idx:
        query_words.insert(3, "evaluation")

    print("\nTop 5 Nearest Neighbors (Cosine Similarity):")
    print("-" * 50)
    for word in query_words:
        neighbors = find_nearest_neighbors(word, embeddings, word2idx, idx2word, top_k=5)
        if neighbors:
            print(f"\n  '{word}':")
            for rank, (neighbor, sim) in enumerate(neighbors, 1):
                print(f"    {rank}. {neighbor:20s} (similarity: {sim:.4f})")

    # Task 3.2: Analogy experiments
    print(f"\nAnalogy Experiments:")
    print("-" * 50)

    # Define analogy tests relevant to the IIT Jodhpur academic domain
    analogies = [
        # Academic level analogies
        ("undergraduate", "btech", "postgraduate", "UG : BTech :: PG : ?"),
        ("student", "learn", "professor", "student : learn :: professor : ?"),
        ("department", "faculty", "institute", "department : faculty :: institute : ?"),
        # Additional analogies with words more likely in the corpus
        ("research", "paper", "teaching", "research : paper :: teaching : ?"),
        ("computer", "science", "electrical", "computer : science :: electrical : ?"),
    ]

    for word_a, word_b, word_c, desc in analogies:
        results = analogy(word_a, word_b, word_c, embeddings, word2idx, idx2word, top_k=3)
        print(f"\n  {desc}")
        if results:
            for rank, (word, sim) in enumerate(results, 1):
                print(f"    {rank}. {word:20s} (similarity: {sim:.4f})")
            
            # Brief interpretation
            top_word = results[0][0]
            print(f"    → The model predicts '{top_word}' as the top answer.")
        else:
            print("    → Could not compute (words missing from vocabulary)")

    print()


# ============================================================
# Visualization Functions
# ============================================================

def visualize_embeddings(embeddings_dict, word2idx, idx2word, output_dir):
    """
    Create PCA and t-SNE visualizations of word embeddings.
    Compares CBOW and Skip-gram clustering behavior.

    Args:
        embeddings_dict: Dict {"CBOW": cbow_emb, "Skip-gram": sg_emb}
        word2idx, idx2word: Vocabulary mappings.
        output_dir: Directory to save visualization images.
    """
    # Select top N most frequent words for visualization (cleaner plots)
    top_n = min(100, len(word2idx))
    selected_indices = list(range(top_n))  # Already sorted by frequency
    selected_words = [idx2word[i] for i in selected_indices]

    for method_name, reduce_func, filename in [
        ("PCA", lambda X: PCA(n_components=2, random_state=42).fit_transform(X), "pca_embeddings.png"),
        ("t-SNE", lambda X: TSNE(n_components=2, random_state=42, perplexity=min(30, top_n-1)).fit_transform(X), "tsne_embeddings.png"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(f"{method_name} Visualization of Word Embeddings", fontsize=16, fontweight="bold")

        for ax_idx, (model_name, embeddings) in enumerate(embeddings_dict.items()):
            subset = embeddings[selected_indices]
            reduced = reduce_func(subset)

            ax = axes[ax_idx]
            ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, s=20, c="steelblue")

            # Label a subset of points to avoid clutter
            label_every = max(1, top_n // 25)
            for i in range(0, len(selected_words), label_every):
                ax.annotate(selected_words[i], (reduced[i, 0], reduced[i, 1]),
                           fontsize=7, alpha=0.8, ha="center", va="bottom")

            ax.set_title(f"{model_name}", fontsize=13)
            ax.set_xlabel(f"{method_name} Component 1")
            ax.set_ylabel(f"{method_name} Component 2")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  {method_name} visualization saved to: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.join(script_dir, "cleaned_corpus.txt")

    if not os.path.exists(corpus_path):
        print("ERROR: cleaned_corpus.txt not found. Run scrape_iitj.py first.")
        return

    # Load corpus
    print("Loading corpus...")
    sentences = load_corpus(corpus_path)
    print(f"  Loaded {len(sentences)} sentences")

    # Build vocabulary
    print("Building vocabulary...")
    word2idx, idx2word, word_counts = build_vocab(sentences, min_count=MIN_COUNT)

    # Build noise distribution for negative sampling
    noise_dist = build_noise_distribution(word_counts)

    # Print hyperparameters summary
    print("\n" + "=" * 60)
    print("HYPERPARAMETER CONFIGURATION")
    print("=" * 60)
    print(f"  (i)   Embedding dimension: {EMBEDDING_DIM}")
    print(f"  (ii)  Context window size: {WINDOW_SIZE}")
    print(f"  (iii) Number of negative samples: {NUM_NEGATIVE}")
    print(f"  Other: lr={LEARNING_RATE}, epochs={NUM_EPOCHS}, batch={BATCH_SIZE}")

    # ---- Train Skip-gram ----
    sg_embeddings = train_skipgram(
        sentences, word2idx, idx2word, word_counts, noise_dist,
        EMBEDDING_DIM, WINDOW_SIZE, NUM_NEGATIVE, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE
    )

    # ---- Train CBOW ----
    cbow_embeddings = train_cbow(
        sentences, word2idx, idx2word, word_counts, noise_dist,
        EMBEDDING_DIM, WINDOW_SIZE, NUM_NEGATIVE, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE
    )

    # ---- Semantic Analysis ----
    semantic_analysis(sg_embeddings, word2idx, idx2word, "Skip-gram (From Scratch)")
    semantic_analysis(cbow_embeddings, word2idx, idx2word, "CBOW (From Scratch)")

    # ---- Visualizations ----
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    embeddings_dict = {"CBOW (Scratch)": cbow_embeddings, "Skip-gram (Scratch)": sg_embeddings}
    visualize_embeddings(embeddings_dict, word2idx, idx2word, script_dir)

    # Save embeddings for further use
    np.save(os.path.join(script_dir, "sg_embeddings.npy"), sg_embeddings)
    np.save(os.path.join(script_dir, "cbow_embeddings.npy"), cbow_embeddings)

    # Save vocabulary
    import json
    with open(os.path.join(script_dir, "vocab.json"), "w") as f:
        json.dump(word2idx, f)

    print("\nEmbeddings and vocabulary saved.")
    print("All tasks completed successfully!")


if __name__ == "__main__":
    main()
