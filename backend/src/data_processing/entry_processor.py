from typing import Dict, Any
from datetime import datetime

from backend.src.data_processing.text_preprocessor import TextPreprocessor

class EntryProcessor:
    """
    Class for processing paper entries and standardising their structure.
    - Ensures that the entries contain all the required keys and that the values have the correct types.
    - Processes the text content of the entries.
    - Summarises the information about each paper (entry) as a string.

    Entries are expected to be in the following format:
    {
        "id": str,
        "title": str,
        "summary": str,
        "authors": List[str],
        "published": str,
        "paper_link": str,
        "content": str (optional),
        "citationCount": int,
        "influentialCitationCount": int
    }

    For example:
    {
        "id": "1234.56789",
        "title": "A Sample Paper Title",
        "summary": "This paper presents a new method for...",
        "authors": ["Alice", "Bob"],
        "published": "2021-01-01",
        "pdf_link": "https://www.semanticscholar.org/paper/1234.56789",
        "content": ""
        "citationCount": 100,
        "influentialCitationCount": 20
    }
    """
    def __init__(self):
        self.text_preprocessor = TextPreprocessor()
        self.entries_requirements = {
                                    "id": str,
                                    "title": str,
                                    "summary": str,
                                    "authors": list,
                                    "published": str,
                                    "paper_link": str,
                                    "content": str
                                    }
        self.acceptable_missing_keys = ["authors", "published", "paper_link", "content"]
    
    def validate_entry(self, entry:Dict[str, Any]) -> None:
        """
        Validates the structure of a paper entry.
        - Ensures that the entry contains all the required keys
          and that the values have the correct types.

        Args:
            entry (Dict[str, Any]): The paper entry to validate.
        """
        for key, value_type in self.entries_requirements.items():
            if key not in entry and key in self.acceptable_missing_keys: # Skip missing keys that are acceptable
                continue
            if not isinstance(entry[key], value_type):
                raise ValueError(f"Entry key: {key} has the wrong type: {type(entry[key])}, expected: {value_type}")
        
        if len(entry["authors"]) > 0:
            if not all(isinstance(author, str) for author in entry["authors"]):
                raise ValueError("Entry authors list contains non-string values")
    
    def summarise_entry(self, entry:Dict[str, Any]) -> str:
        """
        Summarises a paper entries into a string containing 
        the key information about each paper.
        
        Args:
            entry (Dict[str, Any]): The paper entry to summarise.
        """
        current_time = datetime.now().strftime("%Y-%m-%d/%H:%M:%S") # YYYY-MM-DD HH:MM:SS
        paper_string = ""
        paper_string += f"Time: {current_time}\n"
        paper_string += f"ID: {entry['id']}\n"
        paper_string += f"Title: {entry['title']}\n"
        paper_string += f"Summary: {entry['summary']}\n"
        paper_string += f"Authors: {', '.join(entry['authors'])}\n"
        paper_string += f"Published: {entry['published']}\n"
        paper_string += f"Semantic Scholar Link: {entry['paper_link']}\n"
        paper_string += f"Content: {entry['content']}\n"
        return paper_string
    
    def __call__(self, entry:Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the text content of an entry.

        Args:
            entry (Dict[str, Any]): The paper entry to process.
        """
        self.validate_entry(entry)
        if "content" in entry:
            entry["content"] = self.text_preprocessor(entry["content"])
        entry["summary"] = self.text_preprocessor(entry["summary"])
        entry["title"] = self.text_preprocessor.remove_newlines(entry["title"]).strip()
        return entry