import os
from clearml import Task, HyperParameterOptimizer
from clearml.automation import UniformParameterRange, DiscreteParameterRange

hyperparameters = {
    "temperature": UniformParameterRange(min_value=0.1, max_value=1.0, step_size=0.1),
    "max_tokens": DiscreteParameterRange(values=[50, 100, 200, 300]),
    "top_p": UniformParameterRange(min_value=0.5, max_value=1.0, step_size=0.1),
    "frequency_penalty": UniformParameterRange(min_value=0.0, max_value=1.0, step_size=0.1),
    "presence_penalty": UniformParameterRange(min_value=0.0, max_value=1.0, step_size=0.1),
    "memory_window_size": DiscreteParameterRange(values=[1, 3, 5]),  # Number of recent exchanges to keep in memory
}

def objective_function(**kwargs):
    """
    This function will be called by the HyperParameterOptimiser with different hyperparameter combinations.
    It runs the RAG pipeline and returns the evaluation metrics.
    """
    # Initialise with the current hyperparameters
    from query_responder import QueryResponder
    query_responder = QueryResponder(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=kwargs.get("temperature", 0.7),
        max_tokens=kwargs.get("max_tokens", 100),
        top_p=kwargs.get("top_p", 1.0),
        frequency_penalty=kwargs.get("frequency_penalty", 0.0),
        presence_penalty=kwargs.get("presence_penalty", 0.0),
    )

    # Adjust window size if needed?
    query_responder.combined_memory.memories[1].k = kwargs.get("memory_window_size", 3)

    # evaluation script
    from src.clearml.track_gpt_experiments import evaluate_arxiv_qa
    evaluate_arxiv_qa(query_responder)

    task = Task.current_task()
    metrics = task.get_last_scalar_metrics()

    
    return metrics.get("BERTScore", {}).get("Average", 0.0)

# Initialise ClearML task
task = Task.init(
    project_name="Large Group Project",
    task_name="RAG Pipeline Hyperparameter Optimization",
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False
)

# hyperparameter optimiser
optimizer = HyperParameterOptimizer(
    base_task_id="<BASE_TASK_ID>", 
    hyper_parameters=hyperparameters,
    objective_metric_title="BERTScore",
    objective_metric_series="Average",
    objective_metric_sign="max", 
    optimizer_class="Optuna",  
    execution_queue="default",  
    max_number_of_concurrent_tasks=4,  
    optimization_time_limit=24.0,  
    compute_time_limit=120,  # Time limit
    total_max_jobs=20,  # number of hyperparameter combinations to try
    min_iteration_per_job=1,
    max_iteration_per_job=1,
)

# optimisation process
optimizer.start()
optimizer.wait()
optimizer.stop()

# best ones
best_params = optimizer.get_top_experiments(top_k=1)[0].get_parameters()
print("Best Hyperparameters:", best_params)
