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
    
    def summarise_entry(entry:Dict[str, Any]) -> str:
        """
        Summarises a paper entries into a string containing 
        the key information about each paper.
        
        Args:
            entry (Dict[str, Any]): The paper entry to summarise.
        """
        paper_string = ""
        paper_string += f"ID: {entry['id']}\n"
        paper_string += f"Title: {entry['title']}\n"
        paper_string += f"Summary: {entry['summary']}\n"
        paper_string += f"Authors: {', '.join(entry['authors'])}\n"
        paper_string += f"Published: {entry['published']}\n"
        paper_string += f"PDF Link: {entry['pdf_link']}\n"
        paper_string += f"Content: {entry['content']}\n"
        return paper_string
    
    def __call__(self, entry:Dict[str, Any]) -> str:
        """
        Processes the text content of an entry.
        """
        self.validate_entry(entry)
        entry["content"] = self.text_preprocessor(entry["content"])
        entry["summary"] = self.text_preprocessor(entry["summary"])
        return self.summarise_entry(entry)