from clearml import Task
from clearml.automation import HyperParameterOptimizer, UniformParameterRange, DiscreteParameterRange, RandomSearch

def main():
    # Initialize ClearML task
    task = Task.init(
        project_name="Large Group Project",
        task_name="RAG Pipeline Hyperparameter Optimization 2",
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False,
        output_uri=True
    )

    # Debugging: Print base task hyperparameters
    base_task = Task.get_task(task_id="1fabccaa4f284c83b75f012e1e8bfc4d")
    print("Base Task Hyperparameters:", base_task.get_parameters())

    # Set up the hyperparameter optimizer
    an_optimizer = HyperParameterOptimizer(
        base_task_id="1fabccaa4f284c83b75f012e1e8bfc4d",  # Ensure this matches the base task ID
        hyper_parameters=[
            UniformParameterRange(name="General/temperature", min_value=0.1, max_value=1.0, step_size=0.1),
            DiscreteParameterRange(name="General/max_tokens", values=[50, 100, 200, 300]),
            UniformParameterRange(name="General/top_p", min_value=0.5, max_value=1.0, step_size=0.1),
            UniformParameterRange(name="General/frequency_penalty", min_value=0.0, max_value=1.0, step_size=0.1),
            UniformParameterRange(name="General/presence_penalty", min_value=0.0, max_value=1.0, step_size=0.1),
        ],
        objective_metric_title="BERTScore",
        objective_metric_series="Average",
        objective_metric_sign="max",
        max_number_of_concurrent_tasks=4,
        optimizer_class=RandomSearch,
        execution_queue="default",
        time_limit_per_job=120,
        total_max_jobs=20,
    )

    # Enqueue the hyperparameter sweep task for remote execution
    task.execute_remotely(queue_name="default")

    # Start optimization
    an_optimizer.start()
    an_optimizer.wait()
    an_optimizer.stop()

    print('Hyperparameter sweep completed successfully')

if __name__ == "__main__":
    main()