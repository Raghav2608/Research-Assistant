import time
from typing import Dict, Any

from backend.src.data_processing.entry_processor import EntryProcessor

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
        time_takens = []
        for i, entry in enumerate(entries):
            try:
                start_time = time.perf_counter()
                print(f"Paper: {i+1}")

                # Process the entry (in-place)
                entries[i] = self.entry_processor(entry)
                
                print("\n")
                end_time = time.perf_counter()
                time_takens.append(end_time-start_time)
            except Exception as e:
                print(f"Error processing paper {i+1}: {e}")
                time_takens.append(0)
        
        for i in range(len(entries)):
            print(f"Time taken for paper {i+1}: {time_takens[i]:.2f} seconds")
    
        if len(entries) > 1:
            print(f"Average time taken per paper: {sum(time_takens)/len(entries):.2f} seconds")
        print(f"Total time taken: {sum(time_takens):.2f} seconds")
        return entries