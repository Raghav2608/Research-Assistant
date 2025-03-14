import os
import time
import json
from clearml import Task, Logger
from openai import OpenAI
from bert_score import score  # For semantic similarity evaluation


task = Task.init(
    project_name="GAP Generative Model",
    task_name="GPT-4o-mini Automatic Parameter Sweep",
    output_uri=True  # Enable storing artifacts and logs in ClearML
)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key is missing! Please set it in your environment variables.")

client = OpenAI()

base_params = {
    "model": "gpt-4",  # Replace with your actual model name
    "temperature": 0.7,  # Default; will be overwritten in the loop
    "max_tokens": 500,  # Default; will be overwritten in the loop
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.6,
}

# connects base parameters to ClearML for tracking
task.connect(base_params)

# hyperparameters
temperature_values = [0.5, 0.7, 0.9]
max_tokens_values = [300, 500]

# prompts for evaluation
prompts = [
    "Summarize the latest research in computer vision.",
    "Explain the significance of transformer models in AI.",
    "What are the key challenges in natural language processing today?"
]

# Ground truth responses for evaluation, to be replaced
ground_truths = [
    "Recent research in computer vision focuses on...",
    "Transformer models have revolutionized AI by...",
    "Key challenges in NLP include...",
]

# Container
results = {"experiments": []}

# Loop over each combination of temperature and max_tokens
experiment_iteration = 1
for temp in temperature_values:
    for max_tok in max_tokens_values:
        # Update the parameters for this combination
        base_params["temperature"] = temp
        base_params["max_tokens"] = max_tok

        for i, prompt in enumerate(prompts):
            # Record the start time
            start_time = time.time()

            # Generate response from the model
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

            # Evaluate using BERTScore
            if i < len(ground_truths): 
                P, R, F1 = score([gpt_output], [ground_truths[i]], lang="en", model_type="bert-base-uncased")
                bertscore_f1 = F1.mean().item()  # F1 score for evaluation
            else:
                bertscore_f1 = None  # No ground truth 

            # Log to ClearML
            logger = Logger.current_logger()
            logger.report_text(f"Prompt: {prompt}")
            logger.report_text(f"Response: {gpt_output}")
            logger.report_scalar("Latency", "response_time", latency, iteration=experiment_iteration)
            logger.report_scalar("Token Usage", "total_tokens", total_tokens, iteration=experiment_iteration)
            if bertscore_f1 is not None:
                logger.report_scalar("BERTScore F1", "score", bertscore_f1, iteration=experiment_iteration)

            # Append results to container
            results["experiments"].append({
                "iteration": experiment_iteration,
                "prompt": prompt,
                "temperature": temp,
                "max_tokens": max_tok,
                "latency": latency,
                "tokens": total_tokens,
                "response": gpt_output,
                "bertscore_f1": bertscore_f1
            })

        
            print(f"Iteration {experiment_iteration}:")
            print(f"  Prompt: {prompt}")
            print(f"  Temperature: {temp}, Max Tokens: {max_tok}")
            print(f"  Latency: {latency:.2f} sec, Tokens: {total_tokens}")
            print(f"  Response: {gpt_output}")
            if bertscore_f1 is not None:
                print(f"  BERTScore F1: {bertscore_f1:.4f}")
            print("\n")

            experiment_iteration += 1

# Save results to JSON file
results_dir = task.get_logger().get_log_directory()
json_path = os.path.join(results_dir, "experiment_results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=4)

# Upload artifact
task.upload_artifact("GPT-4 Responses", artifact_object=json_path)


task.close()
