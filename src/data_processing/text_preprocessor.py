import re
import nltk
import wordninja
from nltk.corpus import wordnet
nltk.download('wordnet')

class TextPreprocessor:
    def remove_newlines(self, text:str) -> str:
        """
        Removes whitespace characters (spaces, tabs, newlines, etc.) 
        
        Args:
            text (str): The text to remove whitespace characters from.
        """
        return re.sub(r'\s+', ' ', text)
    
    def remove_links(self, text:str) -> str:
        """
        Removes any URLs from the text.

        Args:
            text (str): The text to remove URLs from.
        """
        pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        return re.sub(pattern, '',text)
    
    def keep_only_alphanumeric(self, text:str) -> str:
        """
        Removes any characters that are not a letter, digit or underscore.
        
        Args:
            text (str): The text to remove non-alphanumeric characters from.
        """
        return re.sub(r'\W+', ' ', text)
    
    def remove_non_english_words(self, text:str) -> str:
        """
        Removes any non-English words from the text.
        - Checks if the word is in the WordNet dictionary.
        - Keeps single letter words if they are "a" or "i" or a number.
        
        Args:
            text (str): The text to remove non-English words from.
        """
        split_text = text.split(" ")
        english_words = []
        for word in split_text:
            if wordnet.synsets(word):
                word_length = len(word)
                if word_length > 1:
                    english_words.append(word)
                else:
                    is_numeric = word.isnumeric()
                    is_a_or_i = (word.lower() == "a" or word.lower() == "i")
                    if is_numeric or (word_length == 1 and is_a_or_i):
                        english_words.append(word)
        return " ".join(english_words)
    
    def extract_words(self, text:str) -> str:
        """
        Splits the text into words and re-joins them with spaces.
        - Used to split words that are concatenated together.

        Args:
            text (str): The text to apply the operation to.
        """
        all_words = wordninja.split(text)
        print(all_words)
        print(len(all_words))
        return " ".join(all_words)
    
    def remove_repeated_words_and_adjacent_numbers(self, text:str) -> str:
        """
        Removes repeated words and numbers that are adjacent to each other.

        Args:
            text (str): The text to apply the operation to.
        """
        split_text = text.split(" ")
        final_words = []
        prev = None
        for word in split_text:
            
            if prev == word: # Same word repeated
                continue

            if prev is not None:
                if prev.isnumeric() and word.isnumeric(): # Numbers adjacent to each other
                    continue

            prev = word
            final_words.append(word)

        return " ".join(final_words)
    
    def __call__(self, text:str) -> str:
        """
        The main method that applies all the text processing operations to the input text.

        Args:
            text (str): The text to process.
        """
        text = self.remove_newlines(text)

        operations = [
                    self.remove_links, 
                    self.keep_only_alphanumeric, 
                    self.extract_words,
                    self.remove_non_english_words,
                    ]
        for operation in operations:
            text = operation(text)
            text = self.remove_newlines(text)

        text = self.remove_repeated_words_and_adjacent_numbers(text)
        return text