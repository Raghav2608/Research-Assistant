import os
import sys
import json
import logging
import time
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk

from datasets import load_dataset
from clearml import Task, Logger
from dotenv import load_dotenv

# Adjust Python path to find your local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# Import functions to fetch and parse arXiv papers from your utils.py
from backend.src.data_ingestion.arxiv.utils import fetch_arxiv_papers, parse_papers

# We now only use the QueryResponder; the retrieval engine is removed.
from backend.src.RAG.query_responder import QueryResponder

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_paper(paper_id):
    """
    Fetches the paper content and metadata using the arXiv API.
    """
    search_query = f"{paper_id}"
    papers_xml = fetch_arxiv_papers(search_query, start=0, max_results=1)
    if "<opensearch:totalResults>0</opensearch:totalResults>" in papers_xml:
        logger.warning(f"No paper found for paper id {paper_id}.")
        return None
    papers = parse_papers(papers_xml)
    if not papers:
        logger.warning(f"No paper fetched for paper id {paper_id}.")
        return None
    return papers[0]

def calculate_meteor_score(reference, hypothesis):
    """
    Calculate METEOR score between reference and hypothesis.
    """
    if not reference or not hypothesis:
        logger.warning("Empty reference or hypothesis provided to METEOR calculation. Returning score 0.0")
        return 0.0

    from nltk.tokenize import word_tokenize
    ref_tokens = word_tokenize(reference.lower())
    hyp_tokens = word_tokenize(hypothesis.lower())
    logger.info(f"Reference Tokens: {ref_tokens}")
    logger.info(f"Hypothesis Tokens: {hyp_tokens}")

    score = meteor_score([ref_tokens], hyp_tokens)
    return score

def calculate_bleu_score(reference, hypothesis):
    """
    Calculate BLEU score between reference and hypothesis.
    """
    if not reference or not hypothesis:
        logger.warning("Empty reference or hypothesis provided to BLEU calculation. Returning score 0.0")
        return 0.0

    from nltk.tokenize import word_tokenize
    ref_tokens = [word_tokenize(reference.lower())]
    hyp_tokens = word_tokenize(hypothesis.lower())
    score = sentence_bleu(ref_tokens, hyp_tokens)
    return score

def calculate_rouge_score(reference, hypothesis):
    """
    Calculate ROUGE score between reference and hypothesis.
    """
    if not reference or not hypothesis:
        logger.warning("Empty reference or hypothesis provided to ROUGE calculation. Returning score 0.0")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores

def calculate_bertscore(reference, hypothesis):
    """
    Calculate BERTScore between reference and hypothesis.
    """
    if not reference or not hypothesis:
        logger.warning("Empty reference or hypothesis provided to BERTScore calculation. Returning score 0.0")
        return 0.0

    P, R, F1 = bert_score([hypothesis], [reference], lang="en")
    return F1.mean().item()

def evaluate_arxiv_qa(query_responder, dataset, paper, iteration):
    """
    Evaluates the QueryResponder on the dataset and logs metrics.
    """
    results = []
    meteor_scores = []
    bleu_scores = []
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    bert_scores = []
    latencies = []

    logger.info("Generating answers for each QA pair...")

    for i, sample in enumerate(dataset):
        # Ensure sample is a dictionary
        if not isinstance(sample, dict):
            logger.error(f"Sample {i} is not a dictionary: {sample}")
            continue

        question = sample.get("question")
        reference_answer = sample.get("answer")
        if not question or not reference_answer:
            logger.warning(f"Missing question or answer in sample {i}. Skipping.")
            continue

        # Use the full PDF content if available; otherwise, fall back to the summary.
        context_text = paper.get("content") or paper.get("summary")
        if not context_text:
            logger.warning(f"No content available for paper id {paper.get('id')}. Skipping sample.")
            continue

        # Build a document in the expected format for QueryResponder:
        # A dictionary with 'metadata' (which contains the 'link') and 'page_content'.
        document = {
            "metadata": {"link": paper.get("pdf_link", "No link")},
            "page_content": context_text
        }

        # Generate the answer using this single-document context.
        start_time = time.time()  # Start latency measurement
        generated_answer = query_responder.generate_answer([document], question)
        latency = time.time() - start_time  # End latency measurement
        latencies.append(latency)

        # Handle dict return types from generate_answer
        if isinstance(generated_answer, dict):
            generated_answer = generated_answer.get("text") or generated_answer.get("content", "")
        generated_answer = str(generated_answer)

        logger.info(f"--- QA Pair {i+1} ---")
        logger.info(f"Question: {repr(question)}")
        logger.info(f"Reference Answer (len={len(reference_answer)}): {repr(reference_answer)}")
        logger.info(f"Generated Answer (len={len(generated_answer)}): {repr(generated_answer)}")

        try:
            # Calculate METEOR score
            meteor_score_value = calculate_meteor_score(reference_answer, generated_answer)
            meteor_scores.append(meteor_score_value)
            logger.info(f"METEOR Score: {meteor_score_value}")

            # Calculate BLEU score
            bleu_score_value = calculate_bleu_score(reference_answer, generated_answer)
            bleu_scores.append(bleu_score_value)
            logger.info(f"BLEU Score: {bleu_score_value}")

            # Calculate ROUGE score
            rouge_score_value = calculate_rouge_score(reference_answer, generated_answer)
            for key in rouge_scores:
                rouge_scores[key].append(rouge_score_value[key].fmeasure)
            logger.info(f"ROUGE-1: {rouge_score_value['rouge1'].fmeasure}")
            logger.info(f"ROUGE-2: {rouge_score_value['rouge2'].fmeasure}")
            logger.info(f"ROUGE-L: {rouge_score_value['rougeL'].fmeasure}")

            # Calculate BERTScore
            bert_score_value = calculate_bertscore(reference_answer, generated_answer)
            bert_scores.append(bert_score_value)
            logger.info(f"BERTScore: {bert_score_value}")
        except Exception as e:
            logger.error(f"Error calculating scores: {e}")
            meteor_score_value = 0.0
            bleu_score_value = 0.0
            rouge_score_value = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
            bert_score_value = 0.0

        results.append({
            "question": question,
            "reference_answer": reference_answer,
            "generated_answer": generated_answer,
            "meteor_score": meteor_score_value,
            "bleu_score": bleu_score_value,
            "rouge1_score": rouge_score_value["rouge1"].fmeasure,
            "rouge2_score": rouge_score_value["rouge2"].fmeasure,
            "rougeL_score": rouge_score_value["rougeL"].fmeasure,
            "bert_score": bert_score_value,
            "latency": latency
        })

    # Calculate average scores
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_rouge1 = sum(rouge_scores["rouge1"]) / len(rouge_scores["rouge1"]) if rouge_scores["rouge1"] else 0.0
    avg_rouge2 = sum(rouge_scores["rouge2"]) / len(rouge_scores["rouge2"]) if rouge_scores["rouge2"] else 0.0
    avg_rougeL = sum(rouge_scores["rougeL"]) / len(rouge_scores["rougeL"]) if rouge_scores["rougeL"] else 0.0
    avg_bert = sum(bert_scores) / len(bert_scores) if bert_scores else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    # Log metrics to ClearML
    logger.info("Logging metrics to ClearML...")
   # Log metrics to ClearML with specific iteration
    Logger.current_logger().report_scalar("METEOR", "Average", avg_meteor, iteration=iteration)
    Logger.current_logger().report_scalar("BLEU", "Average", avg_bleu, iteration=iteration)
    Logger.current_logger().report_scalar("ROUGE-1", "Average", avg_rouge1, iteration=iteration)
    Logger.current_logger().report_scalar("ROUGE-2", "Average", avg_rouge2, iteration=iteration)
    Logger.current_logger().report_scalar("ROUGE-L", "Average", avg_rougeL, iteration=iteration)
    Logger.current_logger().report_scalar("BERTScore", "Average", avg_bert, iteration=iteration)
    Logger.current_logger().report_scalar("Latency", "Average", avg_latency, iteration=iteration)


    # Print average scores
    print("\nAverage Scores:")
    print(f"METEOR: {avg_meteor}")
    print(f"BLEU: {avg_bleu}")
    print(f"ROUGE-1: {avg_rouge1}")
    print(f"ROUGE-2: {avg_rouge2}")
    print(f"ROUGE-L: {avg_rougeL}")
    print(f"BERTScore: {avg_bert}")
    print(f"Latency: {avg_latency}")

    # Save results to a file
    results_file = 'arxiv_qa_evaluation_results.jsonl'
    with open(results_file, 'w', encoding='utf-8') as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False)
            f.write('\n')

    # Upload the artifact using the task instance
    task = Task.current_task()
    task.upload_artifact(name="Evaluation Results", artifact_object=results_file)
    logger.info(f"Evaluation complete! Results saved to {results_file}")

def main():
    # Initialize ClearML task
  
    task = Task.init(
        project_name="Large Group Project",
        task_name="RAG Pipeline Arxiv QA Evaluation different metrics 2",
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=False,
        output_uri=True
    )

    # Load the .env file from the parent directory
    env_file_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(env_file_path)

    # Retrieve the OpenAI API key from the environment
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY not found in .env file.")

    # Set the OPENAI_API_KEY environment variable
    os.environ["OPENAI_API_KEY"] = openai_key

    # Upload the .env file as an artifact (optional, for remote execution)
    task.upload_artifact(name="env_file", artifact_object=env_file_path)

    # Execute remotely
    #task.execute_remotely(queue_name="default")

    # Define hyperparameters to loop through
    hyperparameter_combinations = [
        {"temperature": 0.7, "max_tokens": 100, "top_p": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0},
        {"temperature": 0.8, "max_tokens": 150, "top_p": 0.9, "frequency_penalty": 0.1, "presence_penalty": 0.1},
        {"temperature": 0.9, "max_tokens": 200, "top_p": 0.8, "frequency_penalty": 0.2, "presence_penalty": 0.2},
        {"temperature": 0.7, "max_tokens": 150, "top_p": 0.9, "frequency_penalty": 0.1, "presence_penalty": 0.1},
        {"temperature": 0.9, "max_tokens": 150, "top_p": 0.9, "frequency_penalty": 0.1, "presence_penalty": 0.1},
        {"temperature": 0.6, "max_tokens": 150, "top_p": 0.9, "frequency_penalty": 0.1, "presence_penalty": 0.1},
        # {"temperature": 0.9, "max_tokens": 200, "top_p": 0.8, "frequency_penalty": 0.2, "presence_penalty": 0.2},
        {"temperature": 0.7, "max_tokens": 200, "top_p": 0.8, "frequency_penalty": 0.2, "presence_penalty": 0.2},
        {"temperature": 0.9, "max_tokens": 150, "top_p": 0.9, "frequency_penalty": 0.2, "presence_penalty": 0.2},
        {"temperature": 0.7, "max_tokens": 150, "top_p": 0.9, "frequency_penalty": 0.2, "presence_penalty": 0.2},
        {"temperature": 0.9, "max_tokens": 200, "top_p": 0.9, "frequency_penalty": 0.2, "presence_penalty": 0.2},
        {"temperature": 0.7, "max_tokens": 200, "top_p": 0.9, "frequency_penalty": 0.2, "presence_penalty": 0.2},
        {"temperature": 0.9, "max_tokens": 150, "top_p": 0.8, "frequency_penalty": 0.2, "presence_penalty": 0.2},
        {"temperature": 0.7, "max_tokens": 150, "top_p": 0.8, "frequency_penalty": 0.1, "presence_penalty": 0.1},
        
         ]

    # Load the dataset
    from datasets import load_dataset
    dataset = load_dataset("taesiri/arxiv_qa", split="train[:16]")  # Use the first 16 rows

    # Verify the dataset structure
    print("First sample in dataset:", dataset[0])

    # Fetch the paper (assuming the first row contains the paper ID)
    paper_id = dataset[0]["paper_id"].replace("arXiv:", "")  # Extract paper ID from the first row
    paper = fetch_paper(paper_id)  # Fetch the paper content
    if not paper:
        raise ValueError(f"Failed to fetch paper with ID: {paper_id}")


        # Loop through hyperparameter combinations with index
    for idx, hyperparameters in enumerate(hyperparameter_combinations):
        logger.info(f"Evaluating with hyperparameters: {hyperparameters}")

        # Connect hyperparameters to the task
        task.connect(hyperparameters)

        # Initialize the QueryResponder
        query_responder = QueryResponder(
      
            openai_api_key=openai_key,
            session_id="foo",  # Use a unique session ID for each run
            temperature=hyperparameters["temperature"],
            max_tokens=hyperparameters["max_tokens"],
            top_p=hyperparameters["top_p"],
            frequency_penalty=hyperparameters["frequency_penalty"],
            presence_penalty=hyperparameters["presence_penalty"],
          
        )

        # Evaluate with current iteration number
        evaluate_arxiv_qa(query_responder, dataset, paper, iteration=idx+1)

        # Add a small delay to ensure ClearML logging completes
        time.sleep(1)

if __name__ == "__main__":
    main()
