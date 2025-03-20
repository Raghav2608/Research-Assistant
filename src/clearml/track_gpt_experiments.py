import os
import sys
import json
import logging
from nltk.translate.meteor_score import meteor_score
import nltk

from datasets import load_dataset
from clearml import Task, Logger

# Adjust Python path to find your local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import existing ingestion functions from your pipeline.
from src.data_ingestion.arxiv.utils import fetch_arxiv_papers, parse_papers

from src.RAG.retrieval_engine import RetrievalEngine
from src.RAG.query_responder import QueryResponder
nltk.download('punkt')

nltk.download('punkt_tab')
nltk.download('wordnet')
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize ClearML task
task = Task.init(
    project_name="Large Group Project",
    task_name="RAG Pipeline Arxiv QA Evaluation (Using METEOR)",
    output_uri=True
)

def ingest_arxiv_abstracts(retrieval_engine, search_query: str, start: int, max_results: int):
    """
    Ingests arXiv paper abstracts into the retrieval engine.
    """
    logger.info(f"Fetching arXiv papers with query: {search_query}")
    papers_xml = fetch_arxiv_papers(search_query, start, max_results)
    papers = parse_papers(papers_xml)
    logger.info(f"Fetched {len(papers)} papers.")

    docs_to_index = []
    for paper in papers:
        abstract = paper.get("summary", "").strip()
        if abstract:
            entry = {
                "summary": abstract,
                "title": paper.get("title", ""),
                "published": paper.get("published", ""),
                "pdf_link": paper.get("pdf_link", "")
            }
            docs_to_index.append(entry)
   
    docs = retrieval_engine.convert_entries_to_docs(docs_to_index)
    retrieval_engine.split_and_add_documents(docs)
    logger.info(f"Ingested {len(docs)} abstracts into the vector store.")

def calculate_meteor_score(reference, hypothesis):
    """
    Calculate METEOR score between reference and hypothesis.
    
    Args:
        reference: Reference text string.
        hypothesis: Hypothesis text string.
        
    Returns:
        METEOR score as a float.
    """
    if not reference or not hypothesis:
        logger.warning("Empty reference or hypothesis provided to METEOR calculation. Returning score 0.0")
        return 0.0

    # Tokenize the strings as meteor_score expects tokenized inputs
    from nltk.tokenize import word_tokenize
    ref_tokens = word_tokenize(reference.lower())
    hyp_tokens = word_tokenize(hypothesis.lower())
    # ref_tokens = word_tokenize(reference_answer.lower())
    # hyp_tokens = word_tokenize(generated_answer.lower())
    logger.info(f"Reference Tokens: {ref_tokens}")
    logger.info(f"Hypothesis Tokens: {hyp_tokens}")

    score = meteor_score([ref_tokens], hyp_tokens)
    return score

def evaluate_arxiv_qa(retrieval_engine, query_responder):
    """
    Loads the taesiri/arxiv_qa dataset, generates answers using the RAG pipeline,
    and evaluates them with METEOR metrics.
    """
    # Ensure required NLTK resources are downloaded
    for resource in ['tokenizers/punkt', 'wordnet']:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split('/')[-1])
   
    logger.info("Loading taesiri/arxiv_qa dataset...")
    dataset = load_dataset("taesiri/arxiv_qa", split="train[:2]")  # Adjust slice as needed

    results = []
    meteor_scores = []
    logger.info("Generating answers for each QA pair...")
   
    for i, sample in enumerate(dataset):
        question = sample["question"]
        reference_answer = sample["answer"]

        # Retrieve documents and generate answer
        retrieved_docs = retrieval_engine.retrieve([question])
        generated_answer = query_responder.generate_answer(retrieved_docs, question)
       
        # Handle dict return types from generate_answer
        if isinstance(generated_answer, dict):
            generated_answer = generated_answer.get("text") or generated_answer.get("content", "")
        # Force generated_answer to a string
        generated_answer = str(generated_answer)
       
        logger.info(f"--- QA Pair {i+1} ---")
        logger.info(f"Question: {repr(question)}")
        logger.info(f"Reference Answer (len={len(reference_answer)}): {repr(reference_answer)}")
        logger.info(f"Generated Answer (len={len(generated_answer)}): {repr(generated_answer)}")

        try:
            score = calculate_meteor_score(reference_answer, generated_answer)
            meteor_scores.append(score)
            logger.info(f"METEOR Score: {score}")
        except Exception as e:
            logger.error(f"Error calculating METEOR score: {e}")
            score = 0.0
           
        results.append({
            "paper_id": sample["paper_id"],
            "model": sample["model"],
            "question": question,
            "reference_answer": reference_answer,
            "generated_answer": generated_answer,
            "meteor_score": score
        })

    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

    logger.info("Logging METEOR metrics to ClearML...")
    Logger.current_logger().report_scalar("METEOR", "Average", avg_meteor, iteration=1)

    print("\nAverage METEOR Score:", avg_meteor)

    results_file = 'arxiv_qa_evaluation_results.jsonl'
    with open(results_file, 'w', encoding='utf-8') as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False)
            f.write('\n')
    task.upload_artifact(name="Evaluation Results", artifact_object=results_file)
    logger.info(f"Evaluation complete! Average METEOR score: {avg_meteor}")

def main():
    from dotenv import load_dotenv
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY", None)
    if not openai_key:
        raise ValueError("OPENAI_API_KEY not found in environment.")

    # Initialize RAG pipeline components.
    retrieval_engine = RetrievalEngine(openai_api_key=openai_key)
    query_responder = QueryResponder(openai_api_key=openai_key)

    # Ingest arXiv abstracts.
    search_query = "all:machine learning"  # Adjust query as needed.
    ingest_arxiv_abstracts(retrieval_engine, search_query, start=0, max_results=5)

    # Evaluate the dataset.
    evaluate_arxiv_qa(retrieval_engine, query_responder)

if __name__ == "__main__":
    main()
