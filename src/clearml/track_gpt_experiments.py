# import os
# import json
# from typing import List, Dict
# from datasets import load_dataset
# from rouge_score import rouge_scorer
# from clearml import Task, Logger
# # Import from RAG pipeline components
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# from src.RAG.query_responder import QueryResponder 
# from src.RAG.retrieval_engine import RetrievalEngine
# from src.RAG.query_generator import ResearchQueryGenerator
# from dotenv import load_dotenv
# import logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# import os

# # e.g., read from environment
# load_dotenv()
# actual_key = os.getenv("OPENAI_API_KEY", "fallback-or-raise-error")


# task = Task.init(project_name="Large Group Project", task_name="RAG Pipeline Evaluation 2")

# def extract_qa_pairs(dataset_split):
#     """
#     Extracts QA pairs from the QASPER dataset.
#     """
#     qa_pairs = []
#     for paper in dataset_split:
#         title = paper.get("title", "No title")
      
#         for qa_list in paper.get("qas", []):
#             question = qa_list[0] if len(qa_list) > 0 else "No question"
#             reference_answer = qa_list[1] if len(qa_list) > 1 else "No answer"
#             #reference_answer = answers[0].get("answer", "No answer provided.") if answers else "No answer provided."
#             qa_pairs.append({
#                 "prompt": f"Title: {title}\nQuestion: {question}",
#                 "question": question,
#                 "reference_answer": reference_answer,
#                 "paper": paper
#             })
#     return qa_pairs

# def ingest_qasper_abstracts(retrieval_engine, qasper_dataset):
#     """Ingest all QASPER abstracts into the retrieval engine's vector store."""
#     docs_to_index = []
#     for paper in qasper_dataset:
#         abstract_text = paper.get("abstract", "")
#         if abstract_text.strip():
#             # Build a single 'entry' dict that matches convert_entries_to_docs' structure
#             entry = {
#                 "summary": abstract_text,
#                 "title": paper.get("title", ""),
#                 "published": paper.get("published", ""),
#                 "pdf_link": "N/A"
#             }
#             docs_to_index.append(entry)

#     # Now convert them to Documents
#     docs = retrieval_engine.convert_entries_to_docs(docs_to_index)
#     retrieval_engine.split_and_add_documents(docs)
#     print(f"Ingested {len(docs)} documents from QASPER abstracts.")


# def main():
#     logger.info("Loading QASPER dataset...")
#     qasper = load_dataset("allenai/qasper", split="test[20:23]" )
    
#     logger.info("Initializing RetrievalEngine...")
#     retrieval_engine = RetrievalEngine(openai_api_key=actual_key)
    
#     # 2. Ingest all abstracts once
#     logger.info("Ingesting QASPER abstracts...")
#     ingest_qasper_abstracts(retrieval_engine, qasper)
    

#     # Extract QA pairs from the dataset
#     logger.info("Extracting QA pairs...")
#     qa_pairs = extract_qa_pairs(qasper)
    
#     # Initialise existing RetrievalEngine and QueryResponder
#     #logger.info("Initializing RetrievalEngine and QueryResponder...")

#     logger.info("Initializing QueryResponder...")
#     query_responder = QueryResponder(openai_api_key=actual_key)
    

#     # retrieval_engine = RetrievalEngine(openai_api_key=actual_key)
#     #query_responder = QueryResponder(openai_api_key=actual_key)  # Replace with your actual initialization
    
#     # Process each QA pair
#     logger.info("Running retrieval + generation on QA pairs...")
#     results = []
#     for qa in qa_pairs:
#         #docs = retrieval_engine.convert_entries_to_docs(entries=qa["abstract"])#qa["context"])
#         #retrieval_engine.split_and_add_documents(docs=docs) # Add documents to ChromaDB (save)

#         retrieved_docs = retrieval_engine.retrieve([qa["question"]])
#         generated_answer = query_responder.generate_answer(retrieved_docs, qa["question"])
#         if isinstance(generated_answer, dict):
#             generated_answer = generated_answer.get("content") or generated_answer.get("text", "")

#         # metric(qa["answer"],generated_answer)   
#         # log("rouge",metric)    
#         # hallucination(context,generated) 
#         results.append({
#             "question": qa["question"],
#             "generated_answer": generated_answer,
#             "reference_answer": qa["reference_answer"]
#         })
    
#     # Evaluate using ROUGE metrics
#     logger.info("Evaluating answers...")
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
#     scores = []
#     for result in results:
#         score = scorer.score(result["reference_answer"], result["generated_answer"])
#         scores.append(score)
    
#     avg_rouge1 = sum(s["rouge1"].fmeasure for s in scores) / len(scores) if scores else 0
#     avg_rougeL = sum(s["rougeL"].fmeasure for s in scores) / len(scores) if scores else 0
    
#     logger.info("Logging metrics to ClearML...")
#     Logger.current_logger().report_scalar("ROUGE-1", "Average", avg_rouge1, iteration=1)
#     Logger.current_logger().report_scalar("ROUGE-L", "Average", avg_rougeL, iteration=1)


#     logger.info("Evaluation Results:")
#     print("\nSample Evaluation Result:")
#     print("Question:", results[0]["question"])
#     print("Reference Answer:", results[0]["reference_answer"])
#     print("Generated Answer:", results[0]["generated_answer"])
#     print("\nAverage ROUGE-1 Score:", avg_rouge1)
#     print("Average ROUGE-L Score:", avg_rougeL)
    
#     logger.info("Saving results to file...")
#     results_file = 'qasper_evaluation_results.jsonl'
#     with open(results_file, 'w') as f:
#         for result in results:
#             json.dump(result, f)
#             f.write('\n')
    
#     # Upload results file to ClearML
#     task.upload_artifact(name="Evaluation Results", artifact_object=results_file)
    
#     logger.info("Evaluation complete!")

# if __name__ == "__main__":
#     main()
import os
import sys
import json
import logging

from datasets import load_dataset
from rouge_score import rouge_scorer
from clearml import Task, Logger

# Adjust Python path to find your local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.RAG.retrieval_engine import RetrievalEngine
from src.RAG.query_responder import QueryResponder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

task = Task.init(
    project_name="Large Group Project",
    task_name="RAG Pipeline QASPER Evaluation (with correct QA parsing)",
    output_uri=True
)

def get_first_nonempty_freeform(sub_answer_list):
    """
    In QASPER, each sub-answer can have "free_form_answer".
    Return the first non-empty free_form_answer, or 'No answer' if none found.
    """
    for sub_ans in sub_answer_list:
        # sub_ans might look like:
        # {
        #   "unanswerable": false,
        #   "free_form_answer": "...",
        #   ...
        # }
        text = sub_ans.get("free_form_answer", "").strip()
        if text:
            return text
    return "No answer"

def extract_qa_pairs(qasper_dataset):
    """
    Extracts (question, reference_answer) from QASPER,
    given that 'qas' is a dictionary with:
       qas["question"] -> list of question strings
       qas["answers"]  -> list of dict, each with "answer": [ sub-answers... ]
    We'll match them by index.
    """
    qa_pairs = []

    for paper in qasper_dataset:
        qas_dict = paper.get("qas", {})
        if not isinstance(qas_dict, dict):
            continue  # skip if unexpected structure

        # The 'question' list
        questions_list = qas_dict.get("question", [])
        # The 'answers' list (same length as questions_list ideally)
        answers_list = qas_dict.get("answers", [])

        # match them by index
        for i in range(len(questions_list)):
            question_text = questions_list[i].strip()
            if i < len(answers_list):
                ans_obj = answers_list[i]  # e.g. {"answer": [ { ... }, { ...} ] }
                sub_answer_list = ans_obj.get("answer", [])
                reference_answer = get_first_nonempty_freeform(sub_answer_list)
            else:
                reference_answer = "No answer"

            if question_text and reference_answer:
                qa_pairs.append({
                    "question": question_text,
                    "reference_answer": reference_answer
                })

    return qa_pairs


def ingest_qasper_abstracts(retrieval_engine, qasper_dataset):
    """
    Ingest the 'abstract' of each paper into the retrieval engine.
    """
    docs_to_index = []
    for paper in qasper_dataset:
        abstract_text = paper.get("abstract", "")
        if abstract_text.strip():
            entry = {
                "summary": abstract_text,
                "title": paper.get("title", ""),
                "published": paper.get("published", ""),
                "pdf_link": "N/A"
            }
            docs_to_index.append(entry)

    docs = retrieval_engine.convert_entries_to_docs(docs_to_index)
    retrieval_engine.split_and_add_documents(docs)
    logger.info(f"Ingested {len(docs)} abstracts into the vector store.")


def main():
    from dotenv import load_dotenv
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY", None)
    if not openai_key:
        raise ValueError("OPENAI_API_KEY not found in environment.")

    logger.info("Loading QASPER dataset (test split) with a small slice for debugging...")

    # If the first 10 have no QAs, try another slice, e.g. [20:30]
    qasper_dataset = load_dataset("allenai/qasper", split="test[7:10]")

    retrieval_engine = RetrievalEngine(openai_api_key=openai_key)
    # 1) Ingest the abstracts once
    ingest_qasper_abstracts(retrieval_engine, qasper_dataset)

    # 2) Extract Q/A pairs
    logger.info("Extracting QA pairs...")
    qa_pairs = extract_qa_pairs(qasper_dataset)
    logger.info(f"Found {len(qa_pairs)} Q/A pairs in QASPER subset.")

    # 3) Initialize QueryResponder
    query_responder = QueryResponder(openai_api_key=openai_key)

    # 4) For each QA, retrieve + generate
    results = []
    logger.info("Generating answers for each QA pair...")
    for i, qa in enumerate(qa_pairs):
        user_question = qa["question"]
        reference_answer = qa["reference_answer"]

        retrieved_docs = retrieval_engine.retrieve([user_question])
        generated_answer = query_responder.generate_answer(retrieved_docs, user_question)

        # If the chain returns a dict, extract text
        if isinstance(generated_answer, dict):
            generated_answer = generated_answer.get("text") or generated_answer.get("content", "")

        results.append({
            "question": user_question,
            "reference_answer": reference_answer,
            "generated_answer": generated_answer
        })

    # 5) Evaluate with ROUGE
    logger.info("Evaluating answers...")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = []
    for r in results:
        score = scorer.score(r["reference_answer"], r["generated_answer"])
        scores.append(score)

    if scores:
        avg_rouge1 = sum(s["rouge1"].fmeasure for s in scores) / len(scores)
        avg_rougeL = sum(s["rougeL"].fmeasure for s in scores) / len(scores)
    else:
        avg_rouge1, avg_rougeL = 0.0, 0.0

    logger.info("Logging metrics to ClearML...")
    Logger.current_logger().report_scalar("ROUGE-1", "Average", avg_rouge1, iteration=1)
    Logger.current_logger().report_scalar("ROUGE-L", "Average", avg_rougeL, iteration=1)

    logger.info("Evaluation Results:")
    if results:
        print("\nSample Evaluation Result:")
        print("Question:", results[0]["question"])
        print("Reference Answer:", results[0]["reference_answer"])
        print("Generated Answer:", results[0]["generated_answer"])

    print("\nAverage ROUGE-1 Score:", avg_rouge1)
    print("Average ROUGE-L Score:", avg_rougeL)

    logger.info("Saving results to file...")
    results_file = 'qasper_evaluation_results.jsonl'
    with open(results_file, 'w', encoding='utf-8') as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False)
            f.write('\n')

    task.upload_artifact(name="Evaluation Results", artifact_object=results_file)
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()


-------- JASMINE VERSION

import os
import sys
import json
import logging
from nltk.translate import meteor_score
import nltk

from datasets import load_dataset
from clearml import Task, Logger

# Adjust Python path to find your local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import existing ingestion functions from your pipeline.
# These functions are defined in src/data/fetch_data.py as per your ingestion pipeline.
from src.data.fetch_data import fetch_arxiv_papers, parse_papers

from src.RAG.retrieval_engine import RetrievalEngine
from src.RAG.query_responder import QueryResponder

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
    
    This function uses the existing ingestion pipeline functions:
    - fetch_arxiv_papers: fetches the XML result from arXiv.
    - parse_papers: converts the XML into a list of paper dictionaries.
    
    It then extracts only the abstract (from the 'summary' field) along with
    title, published date, and pdf_link, and ingests these as documents.
    """
    logger.info(f"Fetching arXiv papers with query: {search_query}")
    papers_xml = fetch_arxiv_papers(search_query, start, max_results)
    papers = parse_papers(papers_xml)
    logger.info(f"Fetched {len(papers)} papers.")

    docs_to_index = []
    for paper in papers:
        # Use only the abstract (summary) from the paper.
        abstract = paper.get("summary", "").strip()
        if abstract:
            entry = {
                "summary": abstract,
                "title": paper.get("title", ""),
                "published": paper.get("published", ""),
                "pdf_link": paper.get("pdf_link", "")
            }
            docs_to_index.append(entry)
    
    # Convert entries to documents and ingest them.
    docs = retrieval_engine.convert_entries_to_docs(docs_to_index)
    retrieval_engine.split_and_add_documents(docs)
    logger.info(f"Ingested {len(docs)} abstracts into the vector store.")

def calculate_meteor_score(reference, hypothesis):
    """
    Calculate METEOR score between reference and hypothesis.
    
    Args:
        reference: Reference text string
        hypothesis: Hypothesis text string
        
    Returns:
        METEOR score as a float
    """
    # Tokenize the texts into words
    reference_tokens = nltk.word_tokenize(reference.lower())
    hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
    
    # Calculate METEOR score
    score = meteor_score.meteor_score([reference_tokens], hypothesis_tokens)
    return score

def evaluate_arxiv_qa(retrieval_engine, query_responder):
    """
    Loads the taesiri/arxiv_qa dataset, generates answers using the RAG pipeline,
    and evaluates them with METEOR metrics.
    """
    # Make sure NLTK resources are downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('wordnet')
    except LookupError:
        nltk.download('wordnet')
    
    logger.info("Loading taesiri/arxiv_qa dataset...")
    dataset = load_dataset("taesiri/arxiv_qa", split="train[:10]")  # Adjust slice as needed

    results = []
    meteor_scores = []
    logger.info("Generating answers for each QA pair...")
    
    for i, sample in enumerate(dataset):
        # Expected columns: paper_id, model, question, answer
        question = sample["question"]
        reference_answer = sample["answer"]

        # Retrieve documents from the ingested abstracts
        retrieved_docs = retrieval_engine.retrieve([question])
        generated_answer = query_responder.generate_answer(retrieved_docs, question)
        
        # Handle different return types from generate_answer
        if isinstance(generated_answer, dict):
            generated_answer = generated_answer.get("text") or generated_answer.get("content", "")

        # Debug logs to help diagnose any mismatch issues.
        logger.info(f"--- QA Pair {i+1} ---")
        logger.info(f"Question: {repr(question)}")
        logger.info(f"Reference Answer (len={len(reference_answer)}): {repr(reference_answer)}")
        logger.info(f"Generated Answer (len={len(generated_answer)}): {repr(generated_answer)}")

        # Calculate METEOR score
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

    # Calculate average METEOR score
    if meteor_scores:
        avg_meteor = sum(meteor_scores) / len(meteor_scores)
    else:
        avg_meteor = 0.0

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

    # Initialize your RAG pipeline components.
    retrieval_engine = RetrievalEngine(openai_api_key=openai_key)
    query_responder = QueryResponder(openai_api_key=openai_key)

    # 1) Ingest arXiv abstracts using your existing ingestion pipeline.
    search_query = "all:machine learning"  # Adjust query as needed.
    ingest_arxiv_abstracts(retrieval_engine, search_query, start=0, max_results=5)

    # 2) Evaluate the taesiri/arxiv_qa dataset.
    evaluate_arxiv_qa(retrieval_engine, query_responder)

if __name__ == "__main__":
    main()
