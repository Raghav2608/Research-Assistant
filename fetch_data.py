"""
Small script for how to use the data pipeline to fetch and process papers from arXiv and other source.
"""

from backend.src.data_pipeline import DataPipeline
if __name__ == "__main__":
    
    test_sentence = "Are there any recent advancements in transformer models?"
    data_pipeline = DataPipeline()
    entries = data_pipeline.run(test_sentence)

    print(type(entries))
    
    for i, entry in enumerate(entries):
        print(type(entry))
        print(f"Entry {i+1}: {entry.keys()}")