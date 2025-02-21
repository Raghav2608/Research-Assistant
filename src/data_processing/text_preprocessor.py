import re
import nltk
import wordninja
from nltk.corpus import wordnet, stopwords
from src.data_processing.contextual_filtering import ContextualFilter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4') # WordNet 1.4
nltk.download('stopwords')
nltk.download("punkt_tab")
nltk.download('averaged_perceptron_tagger') # For lemmatization

class TextPreprocessor:
    def __init__(self):
        self.contextual_filter = ContextualFilter()
        self.operations = [
                        self.remove_links, 
                        self.keep_only_alphanumeric,
                        self.extract_words, # Splits concatenated words
                        self.remove_non_english_words,
                        self.remove_stopwords,
                        self.lemmatize,
                        self.remove_repeated_words_and_adjacent_numbers,
                        ]
        self.lemmatizer = WordNetLemmatizer()
        self.english_stopwords = set(stopwords.words('english')) # English stopwords

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
    
    def lemmatize(self, text:str) -> str:
        """
        Lemmatizes the words in the text (i.e., converts words to their base form).

        Args:
            text (str): The text to lemmatize.
        """
        return " ".join([self.lemmatizer.lemmatize(word) for word in word_tokenize(text)])
    
    def remove_stopwords(self, text:str) -> str:
        """
        Removes English stopwords from the text.

        Args:
            text (str): The text to remove stopwords from.
        """
        all_words = text.split(" ")
        filtered_words = []
        for word in all_words:
            if word.lower() not in self.english_stopwords:
                filtered_words.append(word)
        return " ".join(filtered_words)
    
    def __call__(self, text:str) -> str:
        """
        The main method that applies all the text processing operations to the input text.
        - Applies the operations in a loop until the text stops changing.

        Args:
            text (str): The text to process.
        """
        prev = None
        num_repeats = 0
        while True:
            prev = text
            text = self.remove_newlines(text)
            for operation in self.operations:
                text = operation(text)
                text = self.remove_newlines(text)
            num_repeats += 1
            if text == prev:
                break
        print("NR", num_repeats)
        # print("Text before contextual filtering:\n", text)
        
        # Apply a single run of contextual filtering.
        text = self.contextual_filter(text)
        for operation in self.operations:
            text = operation(text)
            # print("Text:\n", text)
            text = self.remove_newlines(text)
        return text