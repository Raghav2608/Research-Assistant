import os
import json
from typing import List, Dict
from datasets import load_dataset
from rouge_score import rouge_scorer
from clearml import Task, Logger

# Import from RAG pipeline components
from query_responder.py import QueryResponder
from retrieval_engine.py import RetrievalEngine

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

task = Task.init(project_name="Large Group Project", task_name="RAG Pipeline Evaluation")

def extract_qa_pairs(dataset_split):
    """
    Extracts QA pairs from the QASPER dataset.
    """
    qa_pairs = []
    for paper in dataset_split:
        title = paper.get("title", "No title")
        for qa in paper.get("qas", []):
            question = qa.get("question", "No question")
            answers = qa.get("answers", [])
            reference_answer = answers[0].get("answer", "No answer provided.") if answers else "No answer provided."
            qa_pairs.append({
                "prompt": f"Title: {title}\nQuestion: {question}",
                "question": question,
                "reference_answer": reference_answer,
                "paper": paper
            })
    return qa_pairs

def main():
    logger.info("Loading QASPER dataset...")
    qasper = load_dataset("allenai/qasper", split="test")
    
    # Extract QA pairs from the dataset
    logger.info("Extracting QA pairs...")
    qa_pairs = extract_qa_pairs(qasper)
    
    # Initialise existing RetrievalEngine and QueryResponder
    logger.info("Initializing RetrievalEngine and QueryResponder...")
    retrieval_engine = RetrievalEngine(openai_api_key="your-api-key")  # Replace with your actual initialization
    query_responder = QueryResponder(openai_api_key="your-api-key")  # Replace with your actual initialization
    
    # Process each QA pair
    results = []
    for qa in qa_pairs:
        retrieved_docs = retrieval_engine.retrieve([qa["question"]])
        
        generated_answer = query_responder.generate_answer(retrieved_docs, qa["question"])
        
        results.append({
            "question": qa["question"],
            "generated_answer": generated_answer,
            "reference_answer": qa["reference_answer"]
        })
    
    # Evaluate using ROUGE metrics
    logger.info("Evaluating answers...")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = []
    for result in results:
        score = scorer.score(result["reference_answer"], result["generated_answer"])
        scores.append(score)
    
    avg_rouge1 = sum(s["rouge1"].fmeasure for s in scores) / len(scores) if scores else 0
    avg_rougeL = sum(s["rougeL"].fmeasure for s in scores) / len(scores) if scores else 0
    
    logger.info("Logging metrics to ClearML...")
    Logger.current_logger().report_scalar("ROUGE-1", "Average", avg_rouge1)
    Logger.current_logger().report_scalar("ROUGE-L", "Average", avg_rougeL)

    logger.info("Evaluation Results:")
    print("\nSample Evaluation Result:")
    print("Question:", results[0]["question"])
    print("Reference Answer:", results[0]["reference_answer"])
    print("Generated Answer:", results[0]["generated_answer"])
    print("\nAverage ROUGE-1 Score:", avg_rouge1)
    print("Average ROUGE-L Score:", avg_rougeL)
    
    logger.info("Saving results to file...")
    results_file = 'qasper_evaluation_results.jsonl'
    with open(results_file, 'w') as f:
        for result in results:
            json.dump(result, f)
            f.write('\n')
    
    # Upload results file to ClearML
    task.upload_artifact(name="Evaluation Results", artifact_object=results_file)
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()
