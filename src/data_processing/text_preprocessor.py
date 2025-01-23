import re
import nltk
import wordninja
from nltk.corpus import wordnet
nltk.download('wordnet')

class TextPreprocessor:
    def remove_newlines(self, text:str) -> str:
        return re.sub(r'\s+', ' ', text)
    
    def remove_links(self, text:str) -> str:
        pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        return re.sub(pattern, '',text)
    
    def remove_special_characters(self, text:str) -> str:
        return re.sub(r"[^a-zA-Z0-9\s`~!@#$%^&*()_+\-=\[\]{};:'\"\\|,.<>/?]", "", text)
    
    def remove_non_english_words(self, text:str) -> str:
        split_text = text.split(" ")
        english_words = [word for word in split_text if wordnet.synsets(word)]
        return " ".join(english_words)
    
    def extract_words(self, text:str) -> str:
        all_words = wordninja.split(text)
        print(all_words)
        print(len(all_words))
        return " ".join(all_words)
    
    def __call__(self, text:str) -> str:
        text = self.remove_newlines(text)

        operations = [
                    self.remove_links, 
                    self.remove_special_characters, 
                    self.extract_words,
                    self.remove_non_english_words,
                    ]
        for operation in operations:
            text = operation(text)
            text = self.remove_newlines(text)
        return text