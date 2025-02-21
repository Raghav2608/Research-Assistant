from clearml import Task, Logger
from openai import OpenAI
import time
import os
import json

# Initialise ClearML Task
task = Task.init(
    project_name="GAP Generative Model",
    task_name="GPT-4omini Experiment Tracking",
    output_uri=True
)

# Fetch API key from environment
api_key = os.getenv("OPENAI_API_KEY")

# Ensure key is available
if not api_key:
    raise ValueError("OpenAI API key is missing! Please set it in your environment variables.")

# Initialise OpenAI client
client = OpenAI()

# Track hyperparameters
params = {
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 500,
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.6,
}
task.connect(params)

# Log GPT-3.5 responses
def log_gpt_experiment(prompt, iteration):
    logger = Logger.current_logger()

    # Track response time
    start_time = time.time()
    
    response = client.chat.completions.create(
        model=params["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=params["temperature"],
        max_tokens=params["max_tokens"],
        top_p=params["top_p"],
        frequency_penalty=params["frequency_penalty"],
        presence_penalty=params["presence_penalty"]
    )
    
    latency = time.time() - start_time
    gpt_output = response.choices[0].message.content
    total_tokens = response.usage.total_tokens

    # Log experiment data
    logger.report_text(f"Prompt: {prompt}")
    logger.report_text(f"Response: {gpt_output}")
    logger.report_scalar("Latency", "response_time", latency, iteration=iteration)
    logger.report_scalar("Token Usage", "total_tokens", total_tokens, iteration=iteration)

    print(f"Iteration {iteration}\nPrompt: {prompt}\nResponse: {gpt_output}\nLatency: {latency:.2f} sec\nTokens: {total_tokens}\n")

    return {"iteration": iteration, "prompt": prompt, "response": gpt_output, "latency": latency, "tokens": total_tokens}

# Run Experiment
if __name__ == "__main__":
    prompts = [
        "Summarize the latest research in computer vision.",
        "Explain the significance of transformer models in AI.",
        "What are the key challenges in NLP today?"
    ]

    results = {"responses": []}

    for i, prompt in enumerate(prompts):
        results["responses"].append(log_gpt_experiment(prompt, i + 1))

    # Save results to JSON 
    json_path = os.path.join(task.get_logger().get_log_directory(), "gpt3_responses.json")
    with open(json_path, "w") as f:
        json.dump(results, f)

    # Upload artifact to ClearML
    task.upload_artifact("GPT-4o Responses", artifact_object=json_path)

    task.close()
