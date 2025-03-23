import os
from clearml import Task
from clearml.automation import (
    HyperParameterOptimizer,
    UniformParameterRange,
    DiscreteParameterRange,
    RandomSearch
)

def main():
    task = Task.init(
        project_name="Large Group Project",
        task_name="RAG Pipeline Hyperparameter Sweep",
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False,
    )

    #not real task
    TEMPLATE_TASK_ID = "1fabccaa4f284c83b75f012e1e8bfc4d"

    optimizer = HyperParameterOptimizer(
        base_task_id=TEMPLATE_TASK_ID,
        hyper_parameters=[
            UniformParameterRange("temperature", min_value=0.1, max_value=1.0, step_size=0.1),
            DiscreteParameterRange("max_tokens", values=[50, 100, 200, 300]),
            UniformParameterRange("top_p", min_value=0.5, max_value=1.0, step_size=0.1),
            UniformParameterRange("frequency_penalty", min_value=0.0, max_value=1.0, step_size=0.1),
            UniformParameterRange("presence_penalty", min_value=0.0, max_value=1.0, step_size=0.1),
        ],
        objective_metric_title="BERTScore",
        objective_metric_series="Average",
        objective_metric_sign="max",
        optimizer_class=RandomSearch,
        max_number_of_concurrent_tasks=4,
        total_max_jobs=20,
        execution_queue="default",
        time_limit_per_job=120,
    )
    task.execute_remotely(queue_name="default")
    optimizer.start()
    optimizer.wait()
    optimizer.stop()

    top3 = optimizer.get_top_experiments(top_k=3)
    print("Top 3 experiments:", [t.id for t in top3])

if __name__ == "__main__":
    main()
