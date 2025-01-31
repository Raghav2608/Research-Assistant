import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from src.data_processing.entry_processor import EntryProcessor
from src.data_ingestion.arxiv.utils import fetch_arxiv_papers, parse_papers
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

def run_eda(entries):
    """
    Perform Exploratory Data Analysis on dynamically fetched and processed data.

    Args:
        entries (list[dict]): List of processed paper entries.
    """
    # Analyses text lengths
    lengths = [len(entry["content"]) for entry in entries]
    plt.hist(lengths, bins=30, edgecolor="black", alpha=0.7)
    plt.title("Text Length Distribution")
    plt.xlabel("Length of Content")
    plt.ylabel("Frequency")
    plt.show()

    # Generates word cloud for content
    all_text = " ".join([entry["content"] for entry in entries if entry["content"]])
    wordcloud = WordCloud(width=800, height=400).generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Content")
    plt.show()

    # Token count analysis
    token_counts = [len(word_tokenize(entry["content"])) for entry in entries if entry["content"]]
    print("\nToken Count Summary:")
    print(f"Average tokens per entry: {sum(token_counts)/len(token_counts):.2f}")
    print(f"Max tokens: {max(token_counts)}")
    print(f"Min tokens: {min(token_counts)}")

    # Example: List most common words
    all_words = word_tokenize(all_text)
    word_freq = nltk.FreqDist(all_words)
    print("\nMost Common Words:")
    for word, freq in word_freq.most_common(10):
        print(f"{word}: {freq}")

if __name__ == "__main__":
    # Step 1: Parameters for fetching papers
    search_query = "all:transformer"  # Example query
    start = 0
    max_results = 10  # 10 papers for analysis

    # Fetch and parse papers dynamically
    xml_papers = fetch_arxiv_papers(search_query, start, max_results)
    entries = parse_papers(xml_papers)

    # Process each entry using EntryProcessor
    entry_processor = EntryProcessor()
    processed_entries = [entry_processor(entry) for entry in entries]

    # Perform EDA on processed entries
    run_eda(processed_entries)