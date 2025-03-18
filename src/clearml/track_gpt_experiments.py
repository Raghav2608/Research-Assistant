import os
import json
from typing import Dict, List, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory, CombinedMemory
from rouge_score import rouge_scorer
from datasets import load_dataset
from clearml import Task, Logger

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


task = Task.init(project_name="QASPER Evaluation", task_name="RAG Pipeline Evaluation")

class RetrievalEngine:
    """
    A class that handles the retrieval of documents based on user queries.
    """
    def __init__(self, openai_api_key: str):
        """
        Initialize the Retrieval Engine with the OpenAI API key and the ChromaDB.
        """
        os.environ["USER_AGENT"] = "myagent"  
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)

        PERSIST_DIR = "chroma_db"
        if not os.path.exists(PERSIST_DIR):
            os.makedirs(PERSIST_DIR)
        
        self.vector_store = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        
        self.SEARCH_K = 5 
        self.FETCH_K = self.SEARCH_K * 3  
        self.initiate_vector_retriever()

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    def initiate_vector_retriever(self) -> None:
        """
        Initializes the vector retriever for the ChromaDB.
        """
        self.vector_retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.SEARCH_K, "fetch_k": self.FETCH_K}
        )
    
    def convert_entries_to_docs(self, entries: List[Dict[str, str]]) -> List[Document]:
        """
        Converts the retrieved entries into document objects.
        """
        docs = []
        for entry in entries:
            doc = Document(
                page_content=entry["summary"],  
                metadata={
                    "title": entry["title"],
                    "published": entry["published"],
                    "link": entry["pdf_link"],
                },
            )
            docs.append(doc)
        return docs
    
    def split_and_add_documents(self, docs: List[Document]) -> None:
        """
        Splits the documents into chunks and adds the chunks to the ChromaDB.
        """
        all_splits = self.text_splitter.split_documents(docs)
        self.vector_store.add_documents(all_splits)
        self.initiate_vector_retriever()  

    def convert_docs_to_dicts(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """
        Converts the documents into dictionaries containing the page content and metadata.
        """
        doc_dicts = []
        for doc in docs:
            doc_dict = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            doc_dicts.append(doc_dict)
        return doc_dicts
    
    def retrieve(self, user_queries: List[str]) -> List[Dict[str, Any]]:
        """
        The main function for retrieving documents based on the user query.
        """
        all_results = []
        for user_query in user_queries:
            results = self.vector_store.similarity_search_with_score(query=user_query, k=self.SEARCH_K)
            all_results.extend(results)
        
        if len(all_results) == 0:
            return []
        
        # Sort documents by score
        all_results = sorted(all_results, key=lambda x: x[1], reverse=True)

        retrieved_docs = [doc for doc, score in all_results]
        retrieved_docs = self.convert_docs_to_dicts(retrieved_docs)
        return retrieved_docs

class QueryResponder:
    """
    Class that responds to a user query by combining context and the query, then using an LLM model
    to generate a context-aware answer.
    """
    def __init__(self, openai_api_key: str):
        os.environ["USER_AGENT"] = "myagent"  
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)

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

    def generate_answer(self, retrieved_docs: List[Dict], user_query: str) -> str:
        """
        Generates an answer based on retrieved documents and the user query.
        """
        try:
            formatted_content = "\n\n".join(
                f"Source: {doc['metadata']['link']}\nContent: {doc['page_content']}"
                for doc in retrieved_docs
            )
            prompt = {"context": formatted_content, "question": user_query}
            answer = self.qa_chain.invoke(prompt)
            return answer["text"]
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Error generating answer."

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
    # Load the QASPER dataset 
    logger.info("Loading QASPER dataset...")
    qasper = load_dataset("allenai/qasper", split="test")
    
    logger.info("Extracting QA pairs...")
    qa_pairs = extract_qa_pairs(qasper)
    
    logger.info("Initializing RetrievalEngine and QueryResponder...")
    retrieval_engine = RetrievalEngine(openai_api_key="OPEN_AI_KEY")
    query_responder = QueryResponder(openai_api_key="OPEN_AI_KEY")
    
    # Process each QA pair
    results = []
    for qa in qa_pairs:
        # Retrieve relevant documents for the current paper
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
    
    # Save to a JSONL file
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
