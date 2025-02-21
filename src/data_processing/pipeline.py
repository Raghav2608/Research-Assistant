import time
from typing import Dict, Any

from src.data_processing.entry_processor import EntryProcessor

class DataProcessingPipeline:
    """
    A class that represents the data processing pipeline for structured paper entries.
    This pipeline is used for processing the text content of the entries.
    """
    def __init__(self):
        self.entry_processor = EntryProcessor()
        
    def process(self, entries:Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the text content of the entries.

        Args:
            entries (Dict[str, Any]): The structured paper entries to process.
        """
        start_time = time.perf_counter()
        for i, entry in enumerate(entries):
            print(f"Paper: {i+1}")
            paper_content = entry["content"]
            print(paper_content)
            print("Number of characters (before processing):", len(paper_content))

            # Process the entry (in-place)
            entries[i] = self.entry_processor(entry)

            processed_content = entry["content"]
            print(processed_content)
            print("Number of characters (after processing):", len(processed_content))
            print("\n")

        end_time = time.perf_counter()
        print(f"Time taken: {end_time-start_time:.2f} seconds")
        return entries