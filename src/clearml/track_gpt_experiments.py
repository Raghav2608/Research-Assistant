import os
import time
import json
from clearml import Task, Logger
from openai import OpenAI
from bert_score import score  # For semantic similarity evaluation

task = Task.init(
    project_name="GAP Generative Model",
    task_name="GPT-4o-mini Hyperparameter Sweep",
    output_uri=True  # Enable storing artifacts and logs in ClearML
)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key is missing! Please set it in your environment variables.")

client = OpenAI()

# Base parameters
base_params = {
    "model": "gpt-4o",  
    "temperature": 0.7,  
    "max_tokens": 500,  
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.6,
}
task.connect(base_params)

# hyperparameter values to sweep
temperature_values = [0.5, 0.7, 0.9]
max_tokens_values = [300, 500]

prompts = [
    "Summarize the latest research in computer vision.",
    "Explain the significance of transformer models in AI.",
    "What are the key challenges in natural language processing today?"
]

ground_truths = [
    """Recent computer vision research focuses on transformer-based models like Vision Transformers (ViTs), efficiency-focused architectures like MobileNetV4 for edge computing, physics-inspired vision methods (PhyCV) for image enhancement, and AI-based solutions improving low-light or night vision imaging.""",
    """Transformer models significantly impacted AI due to their parallel processing capabilities, scalability to large datasets, and versatility across domains like language processing and computer vision, enabling powerful models such as GPT and Vision Transformers.""",
    """Major NLP challenges include achieving genuine understanding and context-awareness, addressing data-driven biases, ensuring fairness in model outputs, and acquiring high-quality, diverse datasets, particularly for low-resource languages and specialized fields."""
]

results = {"experiments": []}

# Loop over each combination of hyperparameters
experiment_iteration = 1
for temp in temperature_values:
    for max_tok in max_tokens_values:
        base_params["temperature"] = temp
        base_params["max_tokens"] = max_tok

        for i, prompt in enumerate(prompts):
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

            # model output and token usage
            gpt_output = response.choices[0].message.content
            total_tokens = response.usage.total_tokens

            # Evaluate with BERTScore
            if i < len(ground_truths):  
                P, R, F1 = score([gpt_output], [ground_truths[i]], lang="en", model_type="bert-base-uncased")
                bertscore_f1 = F1.mean().item()  # F1 score for evaluation
            else:
                bertscore_f1 = None  # No ground truth available for this prompt

            logger = Logger.current_logger()
            logger.report_text(f"Prompt: {prompt}")
            logger.report_text(f"Response: {gpt_output}")
            logger.report_scalar("Latency", "response_time", latency, iteration=experiment_iteration)
            logger.report_scalar("Token Usage", "total_tokens", total_tokens, iteration=experiment_iteration)
            if bertscore_f1 is not None:
                logger.report_scalar("BERTScore F1", "score", bertscore_f1, iteration=experiment_iteration)

            # Append to container
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

            # Print results to console
            print(f"Iteration {experiment_iteration}:")
            print(f"  Prompt: {prompt}")
            print(f"  Temperature: {temp}, Max Tokens: {max_tok}")
            print(f"  Latency: {latency:.2f} sec, Tokens: {total_tokens}")
            print(f"  Response: {gpt_output}")
            if bertscore_f1 is not None:
                print(f"  BERTScore F1: {bertscore_f1:.4f}")
            print("\n")

            experiment_iteration += 1

# Save to JSON file
results_dir = task.get_logger().get_log_directory()
json_path = os.path.join(results_dir, "experiment_results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=4)

#artifact for ClearML
task.upload_artifact("GPT-4 Responses", artifact_object=json_path)


task.close()
