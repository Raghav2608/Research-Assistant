import re
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')

class TextPreprocessor:
    def remove_newlines(self, text:str) -> str:
        return re.sub(r'\s+', ' ', text)
    
    def remove_links(self, text:str) -> str:
        pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        return re.sub(pattern, text)
    
    def remove_special_characters(self, text:str) -> str:
        return re.sub(r"[^a-zA-Z0-9\s`~!@#$%^&*()_+\-=\[\]{};:'\"\\|,.<>/?]", "", text)
    
    def remove_non_english_words(self, text:str) -> str:
        split_text = text.split(" ")
        english_words = [word for word in split_text if wordnet.synsets(word)]
        return " ".join(english_words)
    
    def __call__(self, text:str) -> str:
        text = self.remove_newlines(text)
        
        text = self.remove_links(text)
        text = self.remove_newlines(text)

        text = self.remove_special_characters(text)
        text = self.remove_newlines(text)

        text = self.remove_non_english_words(text)
        text = self.remove_newlines(text)
        return text