"""
scrape_iitj.py
==============
Script to collect textual data from IIT Jodhpur official website pages,
preprocess it, and create a cleaned corpus for Word2Vec training.

Data Sources (at least 3 IIT Jodhpur pages):
    1. IIT Jodhpur main / about page
    2. Academic programs page
    3. Research / departments page
    4. Faculty profiles page (if available)

Preprocessing Steps:
    (i)   Removal of boilerplate text and formatting artifacts (HTML tags, scripts, styles)
    (ii)  Tokenization using simple whitespace + regex based tokenizer
    (iii) Lower-casing all tokens
    (iv)  Removal of excessive punctuation and non-textual content (URLs, emails, numbers-only tokens)

Output:
    cleaned_corpus.txt  — one sentence per line, tokens separated by spaces
"""

import requests
from bs4 import BeautifulSoup
import re
import os

# ============================================================
# Configuration: IIT Jodhpur URLs to scrape
# ============================================================
URLS = [
    # Main / About page
    "https://iitj.ac.in/",
    # Academics page
    "https://iitj.ac.in/es/en/engineering-science",
    # Department listing page
    "https://iitj.ac.in/department/index.php?id=dept_structure",
    # Research highlights
    "https://iitj.ac.in/office-of-research-development/en/office-of-research-and-development",
    # CSE department page
    "https://cse.iitj.ac.in/",
    # EE department
    "https://ee.iitj.ac.in/",
    
    "https://iitj.ac.in/office-of-students/en/office-of-students",
    # Academic regulations
    "https://iitj.ac.in/office-of-executive-education/en/office-of-executive-education",
    # Faculty page
    "https://iitj.ac.in/main/en/visiting-faculty-members",
    
    "https://iitj.ac.in/dia/en/dia",
]

# Headers to mimic a browser request
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# ============================================================
# Boilerplate phrases to remove (common across IIT J pages)
# ============================================================
BOILERPLATE_PHRASES = [
    "skip to main content",
    "skip to navigation",
    "copyright",
    "all rights reserved",
    "designed and developed",
    "powered by",
    "follow us on",
    "connect with us",
    "privacy policy",
    "terms of use",
    "disclaimer",
    "site map",
    "back to top",
]


def fetch_page(url):
    """
    Fetch HTML content from a given URL.
    Returns the raw HTML text or None if the request fails.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=15, verify=False)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"  [WARNING] Could not fetch {url}: {e}")
        return None


def extract_text_from_html(html_content):
    """
    Parse HTML and extract meaningful text content.
    Removes scripts, styles, navigation, headers, footers, and other non-content elements.
    Returns a list of text paragraphs.
    """
    soup = BeautifulSoup(html_content, "lxml")

    # Remove non-content elements: scripts, styles, nav, footer, header, forms
    for tag in soup(["script", "style", "nav", "footer", "header", "form",
                     "noscript", "iframe", "meta", "link"]):
        tag.decompose()

    # Also remove common boilerplate containers by class/id patterns
    for element in soup.find_all(attrs={"class": re.compile(
            r"(footer|header|nav|menu|sidebar|breadcrumb|social|cookie)", re.I)}):
        element.decompose()

    for element in soup.find_all(attrs={"id": re.compile(
            r"(footer|header|nav|menu|sidebar|breadcrumb|social|cookie)", re.I)}):
        element.decompose()

    # Extract text from remaining elements
    paragraphs = []
    for element in soup.find_all(["p", "li", "td", "h1", "h2", "h3", "h4", "h5", "h6",
                                   "span", "div", "article", "section"]):
        text = element.get_text(separator=" ", strip=True)
        if text and len(text) > 10:  # Filter out very short fragments
            paragraphs.append(text)

    return paragraphs


def preprocess_text(text):
    """
    Preprocess a single text string:
      - Convert to lowercase
      - Remove URLs such as http://... or www...
      - Remove email addresses
      - Remove special characters and excessive punctuation
      - Remove number-only tokens
      - Tokenize by splitting on whitespace
      - Filter out tokens shorter than 2 characters
    Returns a list of clean tokens.
    """
    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", "", text)

    # Remove HTML entities like &amp; &nbsp; etc.
    text = re.sub(r"&\w+;", " ", text)

    # Remove special characters, keeping only alphanumeric and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    tokens = text.split()

    # Remove tokens that are purely numeric or too short
    tokens = [t for t in tokens if len(t) >= 2 and not t.isdigit()]

    return tokens


def is_boilerplate(text):
    """
    Check if a text string is likely boilerplate content.
    Returns True if any boilerplate phrase is found in the text.
    """
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in BOILERPLATE_PHRASES)


def main():
    """
    Main function to orchestrate the scraping and preprocessing pipeline.
    Collects text from IIT Jodhpur pages, cleans it, and saves to cleaned_corpus.txt.
    """
    print("=" * 60)
    print("IIT Jodhpur Data Collection & Preprocessing")
    print("=" * 60)

    all_documents = []       # Each 'document' is a list of tokens from one paragraph
    raw_paragraphs = []      # Keep track of raw paragraphs before cleaning
    successful_urls = []

    for i, url in enumerate(URLS):
        print(f"\n[{i+1}/{len(URLS)}] Fetching: {url}")
        html = fetch_page(url)
        if html is None:
            continue

        successful_urls.append(url)
        paragraphs = extract_text_from_html(html)
        print(f"  Extracted {len(paragraphs)} text segments")

        page_docs = 0
        for para in paragraphs:
            # Skip boilerplate content
            if is_boilerplate(para):
                continue

            tokens = preprocess_text(para)
            if len(tokens) >= 3:  # Only keep sentences with at least 3 meaningful tokens
                all_documents.append(tokens)
                raw_paragraphs.append(para)
                page_docs += 1

        print(f"  Kept {page_docs} cleaned text segments")

    # --------------------------------------------------------
    # Save the cleaned corpus — one sentence per line
    # --------------------------------------------------------
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleaned_corpus.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in all_documents:
            f.write(" ".join(doc) + "\n")

    print(f"\n{'=' * 60}")
    print(f"Data Collection Complete!")
    print(f"{'=' * 60}")
    print(f"  URLs successfully scraped: {len(successful_urls)}/{len(URLS)}")
    print(f"  Total documents (sentences): {len(all_documents)}")
    total_tokens = sum(len(doc) for doc in all_documents)
    print(f"  Total tokens: {total_tokens}")
    vocab = set(token for doc in all_documents for token in doc)
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Corpus saved to: {output_path}")

    # Print which URLs were successfully scraped
    print(f"\nSuccessfully scraped URLs:")
    for url in successful_urls:
        print(f"  - {url}")


if __name__ == "__main__":
    main()
