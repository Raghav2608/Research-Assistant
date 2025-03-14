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
    "Recent advancements in computer vision encompass several key areas:

Vision Transformers (ViTs): Building upon the transformer architecture, ViTs have been applied to image recognition tasks, offering an alternative to convolutional neural networks (CNNs). They process images by dividing them into patches and have shown promising results in various applications, including image classification and segmentation. ​
en.wikipedia.org
+1
en.wikipedia.org
+1

MobileNetV4: This iteration of the MobileNet series focuses on efficient neural network architectures for mobile and edge devices. MobileNetV4 introduces the "universal inverted bottleneck" and attention modules with multi-query attention, enhancing performance while maintaining computational efficiency. ​
en.wikipedia.org

Physics-Inspired Computer Vision (PhyCV): PhyCV leverages algorithms derived from physical phenomena to perform tasks such as edge detection and image enhancement. By emulating light propagation through diffractive media, PhyCV offers efficient and interpretable solutions for various vision tasks. ​
en.wikipedia.org

AI-Enhanced Night Vision: Companies like Deepnight are integrating AI with low-light cameras to develop affordable night vision goggles. This technology enhances low-light imagery using AI image processing, significantly reducing costs and expanding applications beyond the military. ",
    "Transformer models have revolutionized artificial intelligence by introducing a mechanism that allows models to process data non-sequentially, capturing contextual relationships more effectively. Their significance includes:​

Parallel Processing: Unlike traditional recurrent neural networks (RNNs), transformers can process input data in parallel, leading to faster training times and the ability to handle larger datasets. ​
en.wikipedia.org

Scalability: Transformers have been scaled to create large language models like GPT-4, demonstrating capabilities in generating human-like text, translation, and summarization. Their architecture supports the development of models with billions of parameters, enhancing performance across various tasks. ​
businessinsider.com

Cross-Domain Applications: Beyond natural language processing, transformers have been adapted for computer vision (e.g., Vision Transformers) and other fields, showcasing their versatility and effectiveness in modeling complex data structures.",
    "Despite significant advancements, NLP faces several ongoing challenges:

Understanding and Context: Current models, while proficient at generating text, often lack true comprehension and may produce plausible-sounding but incorrect or nonsensical answers. This limitation highlights the gap between pattern recognition and genuine understanding. ​

Bias and Fairness: NLP models trained on large datasets can inadvertently learn and propagate biases present in the data, leading to unfair or discriminatory outcomes. Addressing this requires developing methods to detect and mitigate biases in AI systems. ​
lemonde.fr

Data Quality and Scarcity: High-quality, diverse datasets are essential for training robust NLP models. However, obtaining such data can be challenging, especially for low-resource languages or specialized domains, limiting the models' applicability and performance in these areas. ",
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
