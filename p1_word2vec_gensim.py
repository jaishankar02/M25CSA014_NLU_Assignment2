"""
p1_word2vec_gensim.py
=====================
Word2Vec training using the Gensim library for comparison with from-scratch models.

This script trains CBOW and Skip-gram models using Gensim's Word2Vec implementation,
then performs the same semantic analysis (nearest neighbors, analogies) and generates
comparison visualizations.

Purpose:
    Compare the quality of embeddings learned by the from-scratch PyTorch implementation
    with Gensim's optimized C implementation.

Usage:
    python p1_word2vec_gensim.py
"""

import os
import json
import numpy as np
from gensim.models import Word2Vec

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Hyperparameters (same as from-scratch for fair comparison)
EMBEDDING_DIM = 100
WINDOW_SIZE = 5
NUM_NEGATIVE = 5
MIN_COUNT = 2
NUM_EPOCHS = 15


def load_corpus(filepath):
    """Load corpus as list of token lists (one per line)."""
    sentences = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                sentences.append(tokens)
    return sentences


def train_gensim_models(sentences):
    """
    Train both CBOW and Skip-gram models using Gensim.
    
    Gensim parameters mapped to our hyperparameters:
        - sg=0 for CBOW, sg=1 for Skip-gram
        - vector_size = EMBEDDING_DIM
        - window = WINDOW_SIZE
        - negative = NUM_NEGATIVE
        - min_count = MIN_COUNT
        - epochs = NUM_EPOCHS

    Returns:
        cbow_model: Trained Gensim CBOW model.
        sg_model: Trained Gensim Skip-gram model.
    """
    print("=" * 60)
    print("Training Gensim CBOW Model")
    print("=" * 60)
    cbow_model = Word2Vec(
        sentences=sentences,
        vector_size=EMBEDDING_DIM,
        window=WINDOW_SIZE,
        min_count=MIN_COUNT,
        sg=0,               # CBOW
        negative=NUM_NEGATIVE,
        epochs=NUM_EPOCHS,
        seed=42,
        workers=4,
    )
    print(f"  Vocabulary size: {len(cbow_model.wv)}")
    print(f"  Embedding shape: ({len(cbow_model.wv)}, {EMBEDDING_DIM})")

    print("\n" + "=" * 60)
    print("Training Gensim Skip-gram Model")
    print("=" * 60)
    sg_model = Word2Vec(
        sentences=sentences,
        vector_size=EMBEDDING_DIM,
        window=WINDOW_SIZE,
        min_count=MIN_COUNT,
        sg=1,               # Skip-gram
        negative=NUM_NEGATIVE,
        epochs=NUM_EPOCHS,
        seed=42,
        workers=4,
    )
    print(f"  Vocabulary size: {len(sg_model.wv)}")
    print(f"  Embedding shape: ({len(sg_model.wv)}, {EMBEDDING_DIM})")

    return cbow_model, sg_model


def semantic_analysis_gensim(model, model_name):
    """
    Perform semantic analysis using a Gensim Word2Vec model.
    Reports nearest neighbors and analogy results.
    """
    print(f"\n{'=' * 60}")
    print(f"SEMANTIC ANALYSIS — {model_name} (Gensim)")
    print(f"{'=' * 60}")

    # Top 5 nearest neighbors
    query_words = ["research", "student", "phd", "exam"]
    # Try to add 'evan' or 'evaluation'
    if "evan" in model.wv:
        query_words.insert(3, "evan")
    elif "evaluation" in model.wv:
        query_words.insert(3, "evaluation")

    print("\nTop 5 Nearest Neighbors (Cosine Similarity):")
    print("-" * 50)
    for word in query_words:
        if word in model.wv:
            neighbors = model.wv.most_similar(word, topn=5)
            print(f"\n  '{word}':")
            for rank, (neighbor, sim) in enumerate(neighbors, 1):
                print(f"    {rank}. {neighbor:20s} (similarity: {sim:.4f})")
        else:
            print(f"\n  '{word}': NOT in vocabulary")

    # Analogy experiments
    print(f"\nAnalogy Experiments:")
    print("-" * 50)

    analogies = [
        ("undergraduate", "btech", "postgraduate", "UG : BTech :: PG : ?"),
        ("student", "learn", "professor", "student : learn :: professor : ?"),
        ("department", "faculty", "institute", "department : faculty :: institute : ?"),
        ("research", "paper", "teaching", "research : paper :: teaching : ?"),
        ("computer", "science", "electrical", "computer : science :: electrical : ?"),
    ]

    for word_a, word_b, word_c, desc in analogies:
        try:
            # Gensim analogy: positive=[word_b, word_c], negative=[word_a]
            results = model.wv.most_similar(positive=[word_b, word_c], negative=[word_a], topn=3)
            print(f"\n  {desc}")
            for rank, (word, sim) in enumerate(results, 1):
                print(f"    {rank}. {word:20s} (similarity: {sim:.4f})")
            print(f"    → The model predicts '{results[0][0]}' as the top answer.")
        except KeyError as e:
            print(f"\n  {desc}")
            print(f"    → Cannot compute: {e}")

    print()


def comparison_visualization(scratch_embeddings, gensim_models, scratch_word2idx, output_dir):
    """
    Create comparison visualizations between scratch and gensim embeddings.
    Shows PCA and t-SNE for all four model variants side by side.
    """
    # Use a common set of words present in both scratch and gensim vocabularies
    # Load scratch vocab
    scratch_words = set(scratch_word2idx.keys())
    
    # Get gensim vocab (from CBOW model — should be same for both)
    cbow_model, sg_model = gensim_models
    gensim_words = set(cbow_model.wv.key_to_index.keys())
    
    # Common words, sorted by frequency in gensim
    common_words = sorted(
        scratch_words & gensim_words,
        key=lambda w: cbow_model.wv.key_to_index.get(w, float('inf'))
    )[:80]  # Top 80 common words

    if len(common_words) < 10:
        print("  Not enough common words for meaningful comparison visualization.")
        return

    # Build embedding matrices for each model
    models_data = {}
    
    # Scratch models
    for name, emb_file in [("CBOW (Scratch)", "cbow_embeddings.npy"),
                            ("Skip-gram (Scratch)", "sg_embeddings.npy")]:
        emb = np.load(os.path.join(output_dir, emb_file))
        subset = np.array([emb[scratch_word2idx[w]] for w in common_words])
        models_data[name] = subset

    # Gensim models
    for name, model in [("CBOW (Gensim)", cbow_model), ("Skip-gram (Gensim)", sg_model)]:
        subset = np.array([model.wv[w] for w in common_words])
        models_data[name] = subset

    # Create comparison plot
    for method_name, reduce_func, filename in [
        ("PCA", lambda X: PCA(n_components=2, random_state=42).fit_transform(X),
         "comparison_pca.png"),
        ("t-SNE", lambda X: TSNE(n_components=2, random_state=42,
                                  perplexity=min(30, len(common_words)-1)).fit_transform(X),
         "comparison_tsne.png"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        fig.suptitle(f"{method_name} Comparison: From-Scratch vs Gensim", fontsize=16, fontweight="bold")

        for ax_idx, (model_name, subset) in enumerate(models_data.items()):
            row, col = ax_idx // 2, ax_idx % 2
            ax = axes[row][col]
            reduced = reduce_func(subset)

            ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, s=25, c="steelblue")

            # Label every few words
            label_every = max(1, len(common_words) // 20)
            for i in range(0, len(common_words), label_every):
                ax.annotate(common_words[i], (reduced[i, 0], reduced[i, 1]),
                           fontsize=7, alpha=0.8, ha="center", va="bottom")

            ax.set_title(model_name, fontsize=13)
            ax.set_xlabel(f"{method_name} Component 1")
            ax.set_ylabel(f"{method_name} Component 2")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  {method_name} comparison saved to: {save_path}")


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

    # Train Gensim models
    cbow_model, sg_model = train_gensim_models(sentences)

    # Semantic analysis
    semantic_analysis_gensim(cbow_model, "CBOW")
    semantic_analysis_gensim(sg_model, "Skip-gram")

    # Save Gensim models
    cbow_model.save(os.path.join(script_dir, "gensim_cbow.model"))
    sg_model.save(os.path.join(script_dir, "gensim_sg.model"))
    print("Gensim models saved.")

    # Comparison visualization (only if scratch embeddings exist)
    vocab_path = os.path.join(script_dir, "vocab.json")
    sg_emb_path = os.path.join(script_dir, "sg_embeddings.npy")
    if os.path.exists(vocab_path) and os.path.exists(sg_emb_path):
        print("\nGenerating comparison visualizations...")
        with open(vocab_path, "r") as f:
            scratch_word2idx = json.load(f)
        comparison_visualization(None, (cbow_model, sg_model), scratch_word2idx, script_dir)
    else:
        print("\nSkipping comparison visualization (run p1_word2vec_scratch.py first).")

    print("\nAll Gensim tasks completed!")


if __name__ == "__main__":
    main()
