"""
p1_stats_wordcloud.py
=====================
Compute dataset statistics and generate a Word Cloud from the cleaned corpus.

This script reads the 'cleaned_corpus.txt' file produced by scrape_iitj.py and:
    1. Reports dataset statistics:
       - Total number of documents (lines/sentences)
       - Total number of tokens
       - Vocabulary size (unique tokens)
       - Most frequent words
    2. Generates and saves a Word Cloud image (wordcloud.png)

Usage:
    python p1_stats_wordcloud.py
"""

import os
from collections import Counter
from wordcloud import WordCloud
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt


def load_corpus(filepath):
    """
    Load the cleaned corpus from a text file.
    Each line is one document containing space-separated tokens.
    
    Args:
        filepath (str): Path to the cleaned corpus text file.
    
    Returns:
        documents (list of list of str): Each document is a list of tokens.
    """
    documents = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                documents.append(tokens)
    return documents


def compute_statistics(documents):
    """
    Compute and print dataset statistics.

    Args:
        documents (list of list of str): The tokenized corpus.

    Returns:
        token_counts (Counter): Frequency counts of all tokens.
    """
    total_documents = len(documents)
    all_tokens = [token for doc in documents for token in doc]
    total_tokens = len(all_tokens)
    vocab = set(all_tokens)
    vocab_size = len(vocab)
    token_counts = Counter(all_tokens)

    # Print formatted statistics
    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"  Total number of documents (sentences): {total_documents}")
    print(f"  Total number of tokens:                {total_tokens}")
    print(f"  Vocabulary size (unique tokens):        {vocab_size}")
    print(f"  Average tokens per document:           {total_tokens / max(total_documents, 1):.1f}")
    print()

    # Show top 30 most frequent words
    print("Top 30 Most Frequent Words:")
    print("-" * 40)
    for i, (word, count) in enumerate(token_counts.most_common(30), 1):
        print(f"  {i:3d}. {word:20s} — {count}")
    print()

    return token_counts


def generate_wordcloud(token_counts, output_path):
    """
    Generate and save a Word Cloud visualization.

    Args:
        token_counts (Counter): Word frequency counts.
        output_path (str): File path to save the word cloud image.
    """
    # Create word cloud object with custom styling
    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        max_words=200,
        colormap="viridis",          # Use a visually appealing colormap
        contour_width=1,
        contour_color="steelblue",
        random_state=42,
    )

    # Generate from frequency dict
    wc.generate_from_frequencies(token_counts)

    # Plot and save
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud — IIT Jodhpur Corpus", fontsize=18, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Word Cloud saved to: {output_path}")


def main():
    """
    Main function: load corpus, compute stats, generate word cloud.
    """
    # Determine the directory where this script lives
    script_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.join(script_dir, "cleaned_corpus.txt")

    # Check if the corpus file exists
    if not os.path.exists(corpus_path):
        print(f"ERROR: Corpus file not found at {corpus_path}")
        print("Please run scrape_iitj.py first to generate the cleaned corpus.")
        return

    # Load and analyze
    documents = load_corpus(corpus_path)
    token_counts = compute_statistics(documents)

    # Generate word cloud
    wordcloud_path = os.path.join(script_dir, "wordcloud.png")
    generate_wordcloud(token_counts, wordcloud_path)

    print("Done! Statistics and Word Cloud generated successfully.")


if __name__ == "__main__":
    main()
