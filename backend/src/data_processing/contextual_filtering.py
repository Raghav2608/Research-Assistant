import torch

from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple

from backend.src.chunking.chunker import Chunker

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
        self.similarity_threshold = 0.5
    
    def get_similarity(self, embedding1:torch.Tensor, embedding2:torch.Tensor) -> float:
        """
        Calculates the cosine similarity between two embeddings.

        Args:
            embedding1 (torch.Tensor): The first embedding.
            embedding2 (torch.Tensor): The second embedding.
        """
        return cosine_similarity(embedding1, embedding2)
    
    def get_embeddings_and_tokens_from_chunks(
                                            self, 
                                            chunks:List[Dict[str, torch.Tensor]]
                                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts the embeddings and tokens from the given chunks.

        Args:
            chunks (List[Dict[str, torch.Tensor]]): The chunks containing the 
                                        
        """
        all_embeddings = []
        all_tokens = []
        for inputs in chunks:
            # Get the embeddings from the model
            output = self.model(**inputs)
            embeddings = output.last_hidden_state
            # print(embeddings.shape)
            all_embeddings.append(embeddings)
            all_tokens.append(inputs["input_ids"])
        all_embeddings = torch.cat(all_embeddings, dim=1) # [1, num_tokens, 768]
        all_tokens = torch.cat(all_tokens, dim=1) # [1, num_tokens]
        # print(all_embeddings.shape, all_tokens.shape)
        return all_embeddings, all_tokens

    def filter_based_on_context(self, all_tokens:torch.Tensor, all_embeddings:torch.Tensor) -> str:
        """
        Filters tokens based on their contextual embeddings.
        - Processes a sequence of tokens and their corresponding embeddings,
          retaining only those tokens that have a high similarity with their 
          surrounding context.
        - The first and last tokens are always kept.

        Args:
            all_tokens (torch.Tensor): The tokenized text.
            all_embeddings (torch.Tensor): The embeddings corresponding to the tokens.
        """
        keep_words = []
        num_tokens = all_tokens.size(1)
        for i in range(num_tokens):
            token = all_tokens[0][i].item()
            word = self.tokenizer.decode([token])

            # Last and first tokens should not be kept (special tokens) i.e., [CLS] and [SEP]
            if i == 0 or (i == num_tokens - 1):
                continue
            
            if i < self.context_window:
                before_embedding = all_embeddings[0][:i].mean(dim=0).unsqueeze(0)
            else:
                last_x_words = max(i-self.context_window, 0)
                before_embedding = all_embeddings[0][last_x_words:i].mean(dim=0).unsqueeze(0)

            if (i + self.context_window) >= num_tokens:
                after_embedding = all_embeddings[0][i+1:].mean(dim=0).unsqueeze(0)
            else:
                next_x_words = min(i+self.context_window + 1, num_tokens)
                after_embedding = all_embeddings[0][i+1:next_x_words].mean(dim=0).unsqueeze(0)

            current_embedding = all_embeddings[0][i].unsqueeze(0).detach().numpy()
            before_embedding = before_embedding.detach().numpy()
            after_embedding = after_embedding.detach().numpy()
            # print("Current embedding shape:", current_embedding.shape, np.isnan(current_embedding).any())
            # print("Before embedding shape:", before_embedding.shape, np.isnan(before_embedding).any())
            # print("After embedding shape:", after_embedding.shape, np.isnan(after_embedding).any())

            before_similarity = self.get_similarity(current_embedding, before_embedding)
            after_similarity = self.get_similarity(current_embedding, after_embedding)
            # print("Word:", word)
            # print("Similarity with context before:", before_similarity)
            # print("Similarity with context after:", after_similarity)

            if before_similarity > self.similarity_threshold and after_similarity > self.similarity_threshold:
                keep_words.append(word)
        return " ".join(keep_words)
    
    def __call__(self, text:str) -> str:
        """
        Keeps words that are relevant to the context of the text.
        - Compares the similarity of the word's embedding with the embeddings of the words before and after it.
        - If the similarity is above a certain threshold, the word is kept.

        Args:
            text (str): The text to filter.
        """
        chunks = self.chunker.get_chunks(
                                        text, 
                                        return_as_text=False, 
                                        stride=self.chunker.chunk_size # Non-overlapping chunks
                                        )
        all_embeddings, all_tokens = self.get_embeddings_and_tokens_from_chunks(chunks)
        filtered_text = self.filter_based_on_context(all_tokens=all_tokens, all_embeddings=all_embeddings)
        return filtered_text