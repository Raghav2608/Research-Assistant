import os
from clearml import Task
from clearml.automation import (
    DiscreteParameterRange, HyperParameterOptimizer, RandomSearch,
    UniformParameterRange
)
from datasets import load_dataset
from backend.src.data_ingestion.arxiv.utils import fetch_arxiv_papers, parse_papers

def fetch_paper(paper_id):
    """
    Fetches the paper content and metadata using the arXiv API.
    """
    search_query = f"{paper_id}"
    papers_xml = fetch_arxiv_papers(search_query, start=0, max_results=1)
    if "<opensearch:totalResults>0</opensearch:totalResults>" in papers_xml:
        print(f"No paper found for paper id {paper_id}.")
        return None
    papers = parse_papers(papers_xml)
    if not papers:
        print(f"No paper fetched for paper id {paper_id}.")
        return None
    return papers[0]

def objective_function(**kwargs):
    """
    This function will be called by the HyperParameterOptimizer with different hyperparameter combinations.
    It runs the trial script and returns the evaluation metrics.
    """
    print(f"Running task with hyperparameters: {kwargs}")  # Debug log

    # Initialize the QueryResponder with the current hyperparameters
    from backend.src.RAG.query_responder import QueryResponder
    query_responder = QueryResponder(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=kwargs.get("temperature", 0.7),
        max_tokens=kwargs.get("max_tokens", 100),
        top_p=kwargs.get("top_p", 1.0),
        frequency_penalty=kwargs.get("frequency_penalty", 0.0),
        presence_penalty=kwargs.get("presence_penalty", 0.0),
    )

    # Load the dataset
    dataset = load_dataset("taesiri/arxiv_qa", split="train[:16]")  # Use the first 16 rows

    # Fetch the paper
    paper_id = dataset[0]["paper_id"].replace("arXiv:", "")  # Extract paper ID from the first row
    paper = fetch_paper(paper_id)
    if not paper:
        raise ValueError(f"Failed to fetch paper with ID: {paper_id}")

    # Run the trial script
    from trial import evaluate_arxiv_qa
    evaluate_arxiv_qa(query_responder, dataset, paper)

    # Retrieve the logged metrics from ClearML
    task = Task.current_task()
    metrics = task.get_last_scalar_metrics()

    # Return the BERTScore (or another metric) with a fallback value
    bert_score = metrics.get("BERTScore", {}).get("Average", 0.0)
    print(f"BERTScore: {bert_score}")
    return bert_score

# Initialize ClearML task
task = Task.init(
    project_name="Large Group Project",
    task_name="RAG Pipeline Hyperparameter Optimization 2",
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False,
    output_uri=True
)

# Set up the hyperparameter optimizer
an_optimizer = HyperParameterOptimizer(
    base_task_id="2ff724e7fe0e40588d0c310c21ccf77b",  
    hyper_parameters=[
        UniformParameterRange(name="temperature", min_value=0.1, max_value=1.0, step_size=0.1),
        DiscreteParameterRange(name="max_tokens", values=[50, 100, 200, 300]),
        UniformParameterRange(name="top_p", min_value=0.5, max_value=1.0, step_size=0.1),
        UniformParameterRange(name="frequency_penalty", min_value=0.0, max_value=1.0, step_size=0.1),
        UniformParameterRange(name="presence_penalty", min_value=0.0, max_value=1.0, step_size=0.1),
    ],
    objective_metric_title="BERTScore",
    objective_metric_series="Average",
    objective_metric_sign="max",
    max_number_of_concurrent_tasks=4,  # Number of concurrent tasks
    optimizer_class=RandomSearch,  # Optimization strategy
    execution_queue="default",  # Queue to enqueue tasks
    time_limit_per_job=120,  # Time limit per task (in minutes)
    total_max_jobs=20,  # Total number of tasks to run
)

# Enqueue the hyperparameter sweep task
task.execute_remotely(queue_name="default")

# Start optimization
an_optimizer.start()
an_optimizer.wait()
an_optimizer.stop()

print('Hyperparameter sweep completed successfully')