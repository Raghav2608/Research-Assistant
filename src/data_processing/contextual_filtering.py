import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from src.chunking.chunker import Chunker

class ContextualFilter:
    """
    Class for filtering out words that are not relevant to the context of the text.
    """
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()
        self.chunker = Chunker(chunk_size=512)

        self.context_window = 10
        self.similarity_threshold = 0.1
    
    def get_similarity(self, embedding1:torch.Tensor, embedding2:torch.Tensor) -> float:
        """
        Calculates the cosine similarity between two embeddings.

        Args:
            embedding1 (torch.Tensor): The first embedding.
            embedding2 (torch.Tensor): The second embedding.
        """
        return cosine_similarity(embedding1, embedding2)

    def __call__(self, text:str) -> str:
        """
        Keeps words that are relevant to the context of the text.
        - Compares the similarity of the word's embedding with the embeddings of the words before and after it.
        - If the similarity is above a certain threshold, the word is kept.

        Args:
            text (str): The text to filter.
        """
        # Tokenize the text
        inputs = self.chunker.get_chunks(text, return_as_text=False)[0] # Take first chunk (FOR NOW)

        # Get the embeddings from the model
        output = self.model(**inputs)
        embeddings = output.last_hidden_state
        # print(embeddings.shape)
        
        keep_words = []
        for i, token in enumerate(inputs["input_ids"][0]):
            word = self.tokenizer.decode([token])
            if i == 0 or i == len(inputs["input_ids"][0]) - 1:
                keep_words.append(word)
                continue # Last and first tokens should be kept
            
            if i < self.context_window:
                before_embedding = embeddings[0][:i].mean(dim=0).unsqueeze(0)
            else:
                last_x_words = max(i-self.context_window, 0)
                before_embedding = embeddings[0][last_x_words:i].mean(dim=0).unsqueeze(0)

            if (i + self.context_window) >= len(inputs["input_ids"][0]):
                after_embedding = embeddings[0][i+1:].mean(dim=0).unsqueeze(0)
            else:
                next_x_words = min(i+self.context_window + 1, embeddings.size(1))
                after_embedding = embeddings[0][i+1:next_x_words].mean(dim=0).unsqueeze(0)

            current_embedding = embeddings[0][i].unsqueeze(0).detach().numpy()
            before_embedding = before_embedding.detach().numpy()
            after_embedding = after_embedding.detach().numpy()
            # print("Current embedding shape:", current_embedding.shape, np.isnan(current_embedding).any())
            # print("Before embedding shape:", before_embedding.shape, np.isnan(before_embedding).any())
            # print("After embedding shape:", after_embedding.shape, np.isnan(after_embedding).any())

            before_similarity = self.get_similarity(current_embedding, before_embedding)
            after_similarity = self.get_similarity(current_embedding, after_embedding)
            
            print("Word:", word)
            print("Similarity with context before:", before_similarity)
            print("Similarity with context after:", after_similarity)

            if before_similarity > self.similarity_threshold and after_similarity > self.similarity_threshold:
                keep_words.append(word)
        return " ".join(keep_words)