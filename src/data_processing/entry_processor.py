from src.data_processing.text_preprocessor import TextPreprocessor
from typing import Dict, Any

class EntryProcessor:
    def __init__(self):
        self.text_preprocessor = TextPreprocessor()
    
    def validate_entry(self, entry:Dict[str, Any]) -> bool: 
        if "content" not in entry:
            raise ValueError("Entry does not contain 'content' key.")
        if not isinstance(entry["content"], str):
            raise ValueError("Entry 'content' is not a string.")
        if "summary" not in entry:
            raise ValueError("Entry does not contain 'summary' key.")
        if not isinstance(entry["summary"], str):
            raise ValueError("Entry 'summary' is not a string.")
        return True
    
    def __call__(self, entry:Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the text content of an entry.
        """
        self.validate_entry(entry)
        entry["content"] = self.text_preprocessor(entry["content"])
        entry["summary"] = self.text_preprocessor(entry["summary"])
        return entry