import os
import logging
from clearml import Task
from clearml.automation import (
    DiscreteParameterRange, HyperParameterOptimizer, RandomSearch,
    UniformParameterRange
)

# Try to load Optuna optimizer
try:
    from clearml.automation.optuna import OptimizerOptuna
    aSearchStrategy = OptimizerOptuna
except ImportError as ex:
    try:
        from clearml.automation.hpbandster import OptimizerBOHB
        aSearchStrategy = OptimizerBOHB
    except ImportError as ex:
        logging.getLogger().warning(
            'Apologies, it seems you do not have \'optuna\' or \'hpbandster\' installed, '
            'we will be using RandomSearch strategy instead')
        aSearchStrategy = RandomSearch

def job_complete_callback(
    job_id,                 # type: str
    objective_value,        # type: float
    objective_iteration,    # type: int
    job_parameters,         # type: dict
    top_performance_job_id  # type: str
):
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print('WOOT WOOT we broke the record! BERTScore reached {}'.format(objective_value))

def objective_function(**kwargs):
    """
    This function will be called by the HyperParameterOptimizer with different hyperparameter combinations.
    It runs the RAG pipeline and returns the evaluation metrics.
    """
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
    # Adjust memory window size if needed
    query_responder.combined_memory.memories[1].k = kwargs.get("memory_window_size", 3)
    # Run the evaluation script
    from track_gpt_experiments import evaluate_arxiv_qa
    evaluate_arxiv_qa(query_responder)
    # Retrieve the logged metrics from ClearML
    task = Task.current_task()
    metrics = task.get_last_scalar_metrics()
    # Debugging: Print the metrics
    print("Logged Metrics:", metrics)
    # Return the BERTScore (or another metric) with a fallback value
    bert_score = metrics.get("BERTScore", {}).get("Average", 0.0)
    print("BERTScore:", bert_score)
    return bert_score

# Initialize ClearML task
task = Task.init(
    project_name="Large Group Project",
    task_name="RAG Pipeline Hyperparameter Optimization",
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False,
    output_uri="https://<PAT>@github.kcl.ac.uk/k22039642/5CCSAGAPLargeGroupProject.git"
)


# Set up arguments
args = {
    'template_task_id': "4858e87990794e6aba9aae8ab6f92cf3",  # Your base task ID
    'run_as_service': False,
}
args = task.connect(args)

# Set default queue name
execution_queue = 'default'

# Set up the hyperparameter optimizer
an_optimizer = HyperParameterOptimizer(
    # The base task to optimize
    base_task_id=args['template_task_id'],
    # Define the hyperparameters to optimize
    hyper_parameters=[
        UniformParameterRange(name="temperature", min_value=0.1, max_value=1.0, step_size=0.1),
        DiscreteParameterRange(name="max_tokens", values=[50, 100, 200, 300]),
        UniformParameterRange(name="top_p", min_value=0.5, max_value=1.0, step_size=0.1),
        UniformParameterRange(name="frequency_penalty", min_value=0.0, max_value=1.0, step_size=0.1),
        UniformParameterRange(name="presence_penalty", min_value=0.0, max_value=1.0, step_size=0.1),
        DiscreteParameterRange(name="memory_window_size", values=[1, 3, 5]),
    ],
    # Objective metric to optimize
    objective_metric_title="BERTScore",
    objective_metric_series="Average",
    objective_metric_sign="max",
    # Limit concurrent experiments
    max_number_of_concurrent_tasks=4,
    # Use the optimizer class we determined above
    optimizer_class=aSearchStrategy,
    # Execution queue
    execution_queue=execution_queue,
    # Time limit per job in minutes
    time_limit_per_job=120,
    # How often to check experiments (in minutes)
    pool_period_min=0.2,
    # Maximum number of total jobs to run
    total_max_jobs=20,
    # Minimum iterations per job
    min_iteration_per_job=1,
    # Maximum iterations per job
    max_iteration_per_job=1,
)

# If running as a service
if args['run_as_service']:
    task.execute_remotely(queue_name='services', exit_process=True)

# Set reporting period
an_optimizer.set_report_period(0.2)

# Start optimization with callback
an_optimizer.start(job_complete_callback=job_complete_callback)

# Set time limit for optimization (24 hours)
an_optimizer.set_time_limit(in_minutes=24.0 * 60)

# Wait for completion
an_optimizer.wait()

# Get top experiments
top_exp = an_optimizer.get_top_experiments(top_k=3)
print("Top performing experiments:")
for i, exp in enumerate(top_exp):
    print(f"{i+1}. Experiment ID: {exp.id}, Parameters: {exp.get_parameters()}")

# Stop the optimizer
an_optimizer.stop()

print('Optimization completed successfully')
