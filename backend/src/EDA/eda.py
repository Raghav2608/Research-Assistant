import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.util import ngrams
from collections import Counter
import nltk
import string

from backend.src.data_processing.entry_processor import EntryProcessor
from backend.src.data_ingestion.arxiv.utils import parse_papers, fetch_arxiv_papers

nltk.download('punkt')
nltk.download('stopwords')

# English stopwords from NLTK
english_stopwords = set(stopwords.words('english'))
combined_stopwords = english_stopwords.union(STOPWORDS)

# Add domain-specific stopwords
domain_stopwords = {"model", "learning", "data", "results", "study", "method", "paper", "research"}
combined_stopwords.update(domain_stopwords)

def run_eda(entries):
    contents = [entry["content"] for entry in entries if entry["content"]]

    # text length
    lengths = [len(content) for content in contents]
    plt.hist(lengths, bins=30, edgecolor="black", alpha=0.7)
    plt.title("Text Length Distribution")
    plt.xlabel("Length of Content")
    plt.ylabel("Frequency")
    plt.show()

    # word cloud
    all_text = " ".join(contents)
    wordcloud = WordCloud(width=800, height=400, stopwords=combined_stopwords).generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Content")
    plt.show()

    # token count
    token_counts = [len(word_tokenize(content)) for content in contents]
    print("\nToken Count Summary:")
    print(f"Average tokens per entry: {sum(token_counts)/len(token_counts):.2f}")
    print(f"Max tokens: {max(token_counts)}")
    print(f"Min tokens: {min(token_counts)}")

    # most common words
    all_words = word_tokenize(all_text)
    filtered_words = [word.lower() for word in all_words if word.lower() not in combined_stopwords and word.isalpha()]
    word_freq = FreqDist(filtered_words)
    print("\nMost Common Words (excluding stopwords):")
    for word, freq in word_freq.most_common(10):
        print(f"{word}: {freq}")

    bigrams = list(ngrams(filtered_words, 2))
    bigram_freq = Counter(bigrams)
    print("\nMost Common Bigrams:")
    for bigram, freq in bigram_freq.most_common(10):
        print(f"{' '.join(bigram)}: {freq}")

    # outliers
    outlier_threshold = 2  
    mean_length = sum(lengths) / len(lengths)
    std_length = (sum((x - mean_length) ** 2 for x in lengths) / len(lengths)) ** 0.5
    outliers = [content for content, length in zip(contents, lengths) if abs(length - mean_length) > outlier_threshold * std_length]
    print(f"\nNumber of outliers: {len(outliers)}")

if __name__ == "__main__":
    # Parameters for fetching papers
    search_query = "all:transformer"  
    start = 0
    max_results = 10  

    # fetch and parse papers
    xml_papers = fetch_arxiv_papers(search_query, start, max_results)
    entries = parse_papers(xml_papers)

    # process each entry using EntryProcessor
    entry_processor = EntryProcessor()
    processed_entries = [entry_processor(entry) for entry in entries]

    run_eda(processed_entries)