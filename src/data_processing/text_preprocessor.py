import re

class TextPreprocessor:
    def remove_newlines(self, text:str) -> str:
        return text.replace("\n", " ")
    
    def remove_special_characters(self, text:str) -> str:
        return re.sub(r"[^a-zA-Z0-9\s`~!@#$%^&*()_+\-=\[\]{};:'\"\\|,.<>/?]", "", text)

    def __call__(self, text:str) -> str:
        text = self.remove_special_characters(text)
        text = self.remove_newlines(text)
        return text