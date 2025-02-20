from sentence_transformers import SentenceTransformer
from typing import List

class KeywordExtractor:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def __call__(self, sentences:List[str]):
        embeddings = self.model.encode(sentences)
        return embeddings
    

if __name__ == "__main__":
    kw_extractor = KeywordExtractor()
    sentences = ["What are the latest advancements in deep learning"]
    out = kw_extractor(sentences)
    print("Output", out)