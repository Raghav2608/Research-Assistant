from clearml import Task
import json
import matplotlib.pyplot as plt

# Initialise a ClearML task for comparison
task = Task.init(
    project_name="GAP Generative Model",
    task_name="GPT-4o-mini Experiment Comparison"
)

# Retrieve stored experiment tracking task
experiment_task = Task.get_task(project_name="GAP Generative Model", task_name="GPT-4o-mini Experiment Tracking")

if experiment_task is None:
    raise ValueError("Could not find the GPT-4o Experiment Tracking task in ClearML!")

# Retrieve stored results from the experiment's artifacts
artifact_name = "GPT-4o Responses"
if artifact_name not in experiment_task.artifacts:
    raise ValueError(f"Artifact '{artifact_name}' not found in the experiment tracking task.")

dataset_path = experiment_task.artifacts[artifact_name].get_local_copy()

# Load experiment results from the JSON file
with open(dataset_path, "r") as f:
    results = json.load(f)

# Extract data for analysis
latencies = [r["latency"] for r in results["experiments"]]
tokens = [r["tokens"] for r in results["experiments"]]
prompts = [r["prompt"] for r in results["experiments"]]

# Plot latency per prompt
plt.figure(figsize=(8, 4))
plt.bar(prompts, latencies, color="blue")
plt.xlabel("Prompts")
plt.ylabel("Response Time (sec)")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.title("GPT-4o Latency per Prompt")
plt.tight_layout()
plt.show()

# Plot token usage per prompt
plt.figure(figsize=(8, 4))
plt.bar(prompts, tokens, color="green")
plt.xlabel("Prompts")
plt.ylabel("Tokens Used")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.title("GPT-4o Token Usage per Prompt")
plt.tight_layout()
plt.show()

# Close ClearML task
task.close()
