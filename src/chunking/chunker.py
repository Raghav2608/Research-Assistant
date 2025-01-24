import torch
from transformers import BertTokenizer
from typing import List, Union

class Chunker:
    def __init__(self, chunk_size:int=512):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.chunk_size = chunk_size

    def convert_text_to_tokens(self, text:str) -> List[str]:
        """
        Converts a text string into a list of tokens.
        
        Args:
            text (str): The text to convert into tokens.
        """
        tokens = self.tokenizer(text, return_tensors="pt")
        print(tokens.keys())
        print(len(tokens["input_ids"]), len(tokens["token_type_ids"]), len(tokens["attention_mask"]))
        print(len(tokens), len(tokens["input_ids"][0]), len(tokens["token_type_ids"][0]), len(tokens["attention_mask"][0]))
        return tokens
    
    def get_chunks(self, text:str, return_as_text:bool=False) -> Union[List[str], List[torch.Tensor]]:
        
        tokens = self.convert_text_to_tokens(text)
        num_tokens = len(tokens)
        print(type(num_tokens), type(self.chunk_size))
        num_chunks = (num_tokens + self.chunk_size - 1) // self.chunk_size
        
        chunks = []
        for i in range(num_chunks):
            chunk_input_ids = tokens["input_ids"][0][i*self.chunk_size:min((i+1)*self.chunk_size, num_tokens)]
            chunk_token_type_ids = tokens["token_type_ids"][0][i*self.chunk_size:min((i+1)*self.chunk_size, num_tokens)]
            chunk_attention_mask = tokens["attention_mask"][0][i*self.chunk_size:min((i+1)*self.chunk_size, num_tokens)]
            chunk = {
                "input_ids": chunk_input_ids.unsqueeze(0),
                "token_type_ids": chunk_token_type_ids.unsqueeze(0),
                "attention_mask": chunk_attention_mask.unsqueeze(0)
            }

            if return_as_text:
                chunk = self.tokenizer.decode(chunk)
            chunks.append(chunk)
        return chunks