import os
import json
import logging
from typing import List, Dict
from multiprocessing import Pool
from datasets import load_dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory, CombinedMemory
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from clearml import Task, Logger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialise
task = Task.init(project_name="QASPER Evaluation", task_name="RAG Pipeline Evaluation")

class QueryResponder:
    """
    Class that responds to a user query by combining context and the query, then using an LLM model
    to generate a context-aware answer.
    """
    def __init__(self, openai_api_key: str):
        os.environ["USER_AGENT"] = "myagent"  
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)  
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)

        # Memory components
        summary_memory = ConversationSummaryMemory(
            llm=self.llm,
            memory_key="summary_history",
            input_key="question",
            output_key="text"
        )
        window_memory = ConversationBufferWindowMemory(
            k=3,
            memory_key="window_history",
            input_key="question",
            output_key="text"
        )
        self.combined_memory = CombinedMemory(memories=[summary_memory, window_memory])

        answer_prompt_template = """
        You are a helpful research assistant.
        Below are relevant excerpts from academic papers:

        {context}

        The user has asked the following question:
        {question}

        Please provide a concise, well-structured answer **and include direct quotes or references** from the provided context.
        Use the format [Source: link] (link will be given to you with every paper right after the word "Source").
        """
        answer_prompt = PromptTemplate(
            template=answer_prompt_template,
            input_variables=["context", "question"]
        )
        self.qa_chain = LLMChain(llm=self.llm, prompt=answer_prompt, memory=self.combined_memory)

    def format_documents(self, retrieved_docs: List[Dict]) -> str:
        """
        Formats the retrieved documents into a single string for the LLM model.
        """
        formatted_content = "\n\n".join(
            f"Source: {doc['metadata']['link']}\nTitle: {doc['metadata'].get('title', 'No title')}\nContent: {doc['page_content']}"
            for doc in retrieved_docs
        )
        return formatted_content

    def combine_context_and_question(self, context_text: str, user_query: str) -> Dict[str, str]:
        """
        Combines the context and user query into a dictionary.
        """
        return {"context": context_text, "question": user_query}

    def generate_answer(self, retrieved_docs: List[Dict], user_query: str) -> str:
        """
        Generates an answer based on retrieved documents and the user query.
        """
        try:
            formatted_content = self.format_documents(retrieved_docs)
            prompt = self.combine_context_and_question(formatted_content, user_query)
            answer = self.qa_chain.invoke(prompt)
            return answer["text"] 
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Error generating answer."

def retrieve_relevant_docs(paper):
    """
    Placeholder retrieval function.
    Returns a list of documents. Here, it uses the paper's abstract as context.
    """
    return [{
        "metadata": {"link": "https://example.com/paper", "title": paper.get("title", "No title")},
        "page_content": paper.get("abstract", "No abstract available.")
    }]

def extract_qa_pairs(dataset_split):
    """
    Extracts QA pairs from the QASPER dataset.
    For each paper, it extracts the title, question, and the first reference answer.
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

def process_qa_pair(qa, query_responder):
    """
    Processes a single QA pair: retrieves documents, generates an answer, and evaluates it.
    """
    retrieved_docs = retrieve_relevant_docs(qa["paper"])
    generated_answer = query_responder.generate_answer(retrieved_docs, qa["question"])
    return {
        "question": qa["question"],
        "generated_answer": generated_answer,
        "reference_answer": qa["reference_answer"]
    }

def main():
    logger.info("Loading QASPER dataset...")
    qasper = load_dataset("allenai/qasper", split="test")
    
    logger.info("Extracting QA pairs...")
    qa_pairs = extract_qa_pairs(qasper)
    
    logger.info("Initializing QueryResponder...")
    query_responder = QueryResponder(openai_api_key="your-api-key")
    
    logger.info("Processing QA pairs...")
    with Pool(processes=4) as pool:  
        results = pool.starmap(process_qa_pair, [(qa, query_responder) for qa in qa_pairs])
    

    logger.info("Evaluating answers...")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_scores, rougeL_scores, bleu_scores, meteor_scores = [], [], [], []
    
    for result in results:
        rouge_scores = scorer.score(result["reference_answer"], result["generated_answer"])
        rouge1_scores.append(rouge_scores["rouge1"].fmeasure)
        rougeL_scores.append(rouge_scores["rougeL"].fmeasure)
        
        reference = result["reference_answer"].split()
        generated = result["generated_answer"].split()
        bleu_scores.append(sentence_bleu([reference], generated))
        meteor_scores.append(meteor_score([reference], generated))
    
   
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
    
    # Log metrics
    logger.info("Logging metrics to ClearML...")
    Logger.current_logger().report_scalar("ROUGE-1", "Average", avg_rouge1)
    Logger.current_logger().report_scalar("ROUGE-L", "Average", avg_rougeL)
    Logger.current_logger().report_scalar("BLEU", "Average", avg_bleu)
    Logger.current_logger().report_scalar("METEOR", "Average", avg_meteor)
    
    
    logger.info("Evaluation Results:")
    print("\nSample Evaluation Result:")
    print("Question:", results[0]["question"])
    print("Reference Answer:", results[0]["reference_answer"])
    print("Generated Answer:", results[0]["generated_answer"])
    print("\nAverage Scores:")
    print("ROUGE-1:", avg_rouge1)
    print("ROUGE-L:", avg_rougeL)
    print("BLEU:", avg_bleu)
    print("METEOR:", avg_meteor)
    
    # Save to JSONL file
    logger.info("Saving results to file...")
    results_file = 'qasper_evaluation_results.jsonl'
    with open(results_file, 'w') as f:
        for result in results:
            json.dump(result, f)
            f.write('\n')
    
    # Upload results 
    task.upload_artifact(name="Evaluation Results", artifact_object=results_file)
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()
