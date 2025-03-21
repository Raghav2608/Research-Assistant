from backend.src.data_ingestion.data_pipeline import DataPipeline

def main():
    # Define a sample query. You can change this query as needed.
    sample_query = "GANs in machine learning"
    print("Testing data pipeline with query:", sample_query)
    
    # Create an instance of the DataPipeline
    pipeline = DataPipeline()
    
    # Run the pipeline with the sample query
    results = pipeline.run(sample_query)
    
    # Print the processed results
    print("Processed entries:")
    for entry in results:
        print(entry)

if __name__ == "__main__":
    main()
