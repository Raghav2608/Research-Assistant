import os
import time
import json
from clearml import Task, Logger
from openai import OpenAI

task = Task.init(
    project_name="GAP Generative Model",
    task_name="GPT-4o-mini Automatic Parameter Sweep",
    output_uri=True
)


api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key is missing! Please set it in your environment variables.")


client = OpenAI()


base_params = {
    "model": "gpt-4o-mini",
    "temperature": 0.7,         # default; will be overwritten in the loop
    "max_tokens": 500,          # default; will be overwritten in the loop
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.6,
}


task.connect(base_params)


temperature_values = [0.5, 0.7, 0.9]
max_tokens_values = [300, 500]

prompts = [
    "Summarize the latest research in computer vision.",
    "Explain the significance of transformer models in AI.",
    "What are the key challenges in natural language processing today?"
]

# Container
results = {"experiments": []}

# loops over each combination of temperature and max_tokens
experiment_iteration = 1
for temp in temperature_values:
    for max_tok in max_tokens_values:
        # Update the parameters for this combination.
        base_params["temperature"] = temp
        base_params["max_tokens"] = max_tok

        for prompt in prompts:
            # Record the start time.
            start_time = time.time()

        
            response = client.chat.completions.create(
                model=base_params["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=base_params["temperature"],
                max_tokens=base_params["max_tokens"],
                top_p=base_params["top_p"],
                frequency_penalty=base_params["frequency_penalty"],
                presence_penalty=base_params["presence_penalty"]
            )

           
            latency = time.time() - start_time

          
            gpt_output = response.choices[0].message.content
            total_tokens = response.usage.total_tokens

            # Log experiment data
            logger = Logger.current_logger()
            logger.report_text(f"Prompt: {prompt}")
            logger.report_text(f"Response: {gpt_output}")
            logger.report_scalar("Latency", "response_time", latency, iteration=experiment_iteration)
            logger.report_scalar("Token Usage", "total_tokens", total_tokens, iteration=experiment_iteration)

            # Append the results 
            results["experiments"].append({
                "iteration": experiment_iteration,
                "prompt": prompt,
                "temperature": temp,
                "max_tokens": max_tok,
                "latency": latency,
                "tokens": total_tokens,
                "response": gpt_output
            })

            print(f"Iteration {experiment_iteration}:")
            print(f"  Prompt: {prompt}")
            print(f"  Temperature: {temp}, Max Tokens: {max_tok}")
            print(f"  Latency: {latency:.2f} sec, Tokens: {total_tokens}")
            print(f"  Response: {gpt_output}\n")

            experiment_iteration += 1


results_dir = task.get_logger().get_log_directory()
json_path = os.path.join(results_dir, "experiment_results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=4)

# Upload JSON file as an artifact to ClearML.
task.upload_artifact("GPT-4o Responses", artifact_object=json_path)


task.close()
