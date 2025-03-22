import os

from backend.src.data_ingestion.data_pipeline import DataPipeline
from backend.src.RAG.query_generator import ResearchQueryGenerator

def main():
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    query_generator = ResearchQueryGenerator(openai_api_key=OPENAI_API_KEY,session_id="foo")

    # Define a sample query. You can change this query as needed.
    sample_queries = [
        # "GANS in machine learning",
        # "Are there any recent advancements in transformer models?",
        # "What are the applications of transformer models in recent research?",
        "I am researching how deep learning can be applied to the early detection of neurodegenerative diseases such as Alzheimer's and Parkinson's. I am particularly interested in papers that discuss convolutional neural networks (CNNs) or transformer-based models for medical image analysis, especially MRI or CT scans. The papers should focus on explainability and feature extraction techniques rather than just model performance."
    ]

    additional_queries = query_generator.generate(sample_queries[0])
    print(additional_queries)
    
    # Create an instance of the DataPipeline
    pipeline = DataPipeline()
    
    # Run the pipeline with the sample query
    results = pipeline.run(additional_queries)
    
    # Print the processed results
    print("Processed entries:")
    for entry in results:
        print(entry)

if __name__ == "__main__":
    main()
