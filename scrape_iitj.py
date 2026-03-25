"""
scrape_iitj.py
==============
Script to collect textual data from IIT Jodhpur official website pages (HTML and PDFs),
preprocess it, and create a cleaned corpus for Word2Vec training.
"""

import requests
from bs4 import BeautifulSoup
import re
import os
import fitz  # PyMuPDF

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
    "https://iitj.ac.in/office-of-academics/en/academic-regulations",
    "https://iitj.ac.in/PageImages/Gallery/03-2025/1_Academic_Regulations_Final_03_09_2019.pdf",
    "https://iitj.ac.in/PageImages/Gallery/03-2025/2_Academic_Regulations_Final_03_09_2019.pdf",
    "https://iitj.ac.in/PageImages/Gallery/03-2025/2.1_notification_26102020.pdf",
    "https://iitj.ac.in/PageImages/Gallery/03-2025/3_Academic_Regulations_Final_03_09_2019.pdf",
    ""
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

BOILERPLATE_PHRASES = [
    "skip to main content", "skip to navigation", "copyright", "all rights reserved",
    "designed and developed", "powered by", "follow us on", "connect with us",
    "privacy policy", "terms of use", "disclaimer", "site map", "back to top",
]

def fetch_page(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=15, verify=False)
        response.raise_for_status()
        return response.content if url.endswith('.pdf') else response.text
    except Exception as e:
        print(f"  [WARNING] Could not fetch {url}: {e}")
        return None

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "form", "noscript", "iframe", "meta", "link"]):
        tag.decompose()
    for element in soup.find_all(attrs={"class": re.compile(r"(footer|header|nav|menu|sidebar|breadcrumb|social|cookie)", re.I)}):
        element.decompose()
    for element in soup.find_all(attrs={"id": re.compile(r"(footer|header|nav|menu|sidebar|breadcrumb|social|cookie)", re.I)}):
        element.decompose()

    paragraphs = []
    for element in soup.find_all(["p", "li", "td", "h1", "h2", "h3", "h4", "h5", "h6", "span", "div", "article", "section"]):
        text = element.get_text(separator=" ", strip=True)
        if text and len(text) > 10:
            paragraphs.append(text)
    return paragraphs

def extract_text_from_pdf(pdf_bytes):
    paragraphs = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            text = page.get_text("text")
            for line in text.split('\n'):
                line = line.strip()
                if len(line) > 10:
                    paragraphs.append(line)
    except Exception as e:
        print(f"  [WARNING] Error parsing PDF: {e}")
    return paragraphs

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if len(t) >= 2 and not t.isdigit()]
    return tokens

def is_boilerplate(text):
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in BOILERPLATE_PHRASES)

def main():
    print("=" * 60)
    print("IIT Jodhpur Data Collection & Preprocessing")
    print("=" * 60)

    all_documents = []
    successful_urls = []

    for i, url in enumerate(URLS):
        if not url.strip():
            continue
            
        print(f"\n[{i+1}/{len(URLS)}] Fetching: {url}")
        content = fetch_page(url)
        if content is None:
            continue

        successful_urls.append(url)
        if url.endswith('.pdf'):
            paragraphs = extract_text_from_pdf(content)
        else:
            paragraphs = extract_text_from_html(content)
            
        print(f"  Extracted {len(paragraphs)} text segments")

        page_docs = 0
        for para in paragraphs:
            if is_boilerplate(para):
                continue
            tokens = preprocess_text(para)
            if len(tokens) >= 3:
                all_documents.append(tokens)
                page_docs += 1

        print(f"  Kept {page_docs} cleaned text segments")

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleaned_corpus.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in all_documents:
            f.write(" ".join(doc) + "\n")

    print(f"\n{'=' * 60}")
    print(f"Data Collection Complete!")
    print(f"{'=' * 60}")
    print(f"  URLs successfully scraped: {len(successful_urls)}/{len(URLS)}")
    print(f"  Total documents (sentences): {len(all_documents)}")
    print(f"  Total tokens: {sum(len(doc) for doc in all_documents)}")
    vocab = set(token for doc in all_documents for token in doc)
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Corpus saved to: {output_path}")
    print(f"\nSuccessfully scraped URLs:")
    for url in successful_urls:
        print(f"  - {url}")

if __name__ == "__main__":
    main()
