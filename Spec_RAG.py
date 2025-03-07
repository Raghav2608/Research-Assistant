import os
import getpass
import logging
import urllib
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.tools import tool

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Ingestion utils ( existing code for fetching from arXiv)
from src.data_ingestion.arxiv.utils import fetch_arxiv_papers, parse_papers
from keybert import KeyBERT

load_dotenv()
os.environ['USER_AGENT'] = 'myagent'
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# Initialize models
small_llm = ChatOpenAI(model="gpt-4o-mini")   # cheaper or smaller model
big_llm   = ChatOpenAI(model="gpt-4o-mini")     # larger model
query_gen_llm = ChatOpenAI(model="gpt-4o-mini") # query_generator

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Local Chroma DB
PERSIST_DIR = "chroma_db"
os.makedirs(PERSIST_DIR, exist_ok=True)
vector_store = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

def search_and_document(search_query: str, limit=5):
    """Fetches from arXiv, parse into Documents, limit to 'limit' docs."""
    start = 0
    max_results = 10  # Reduce to speed up
    xml_papers = fetch_arxiv_papers(search_query, start, max_results)
    entries = parse_papers(xml_papers)

    docs = []
    for i, entry in enumerate(entries):
        if i >= limit:
            break
        doc = Document(
            page_content=entry["summary"],   # use entry["content"] or skip PDF parsing, by using entry["summary"]
            metadata={
                "title": entry["title"],
                "published": entry["published"],
                "link": entry["pdf_link"],
            },
        )
        docs.append(doc)
    return docs

def generate_query(user_query:str):
    query_prompt_template = PromptTemplate(
    template="""
    "You are an intelligent query generation assistant specialized in formulating precise search queries for the Semantic Scholar API. Your task is to convert natural language search prompts into well-structured queries that maximize relevance and accuracy. 
    Ensure that:
    1) ONLY Boolean operators (AND, OR, NOT) are used effectively to refine search results.
    2) Unnecessary words are removed, ensuring the query remains concise and precise.
    3) Assumptions are made logically when the user's query is vague, but clarification is requested when needed.
    4) MOST IMPORTANT: dont cover every aspect of the query, if the query is very specific keep your query general

    Example 1: Basic Topic Search
    User Input:
    Find papers on reinforcement learning in robotics.
    
    Output :
    reinforcement learning AND robotics

    Example 2: Multi-constraint Search
    User Input:
    I’m looking for research papers on the use of Graph Neural Networks (GNNs) in drug discovery."

    ChatGPT Output (ArXiv Query):
    Graph Neural Networks AND drug discovery

    So here is the actual input:
    {user_query}

    """,
        input_variables=["user_query"]
    )
    
    query_chain = LLMChain(llm=query_gen_llm,prompt=query_prompt_template)
    return query_chain.invoke({"user_query": user_query})["text"]


def split_and_add_documents(docs: list[Document]):
    """Chunk docs, embed, and add to Chroma DB."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    all_splits = text_splitter.split_documents(docs)
    vector_store.add_documents(all_splits)

def clean_search_query(search_query: str) -> str:
    """Encode the query so it doesn't contain invalid URL chars."""
    return urllib.parse.quote_plus(search_query)

@tool()
def retrieve(user_query: str):
    """
    Single-step retrieval (no multi-query).
    1) fetch from arXiv
    2) chunk & add to vector DB
    3) retrieve top k=4 docs from vector store
    """
    safe_query = clean_search_query(user_query)
    docs = search_and_document(safe_query)

    if not docs:
        return {
            "content": "No relevant documents found.",
            "artifact": []
        }

    # Index docs in Chroma
    split_and_add_documents(docs)

    # Create a retriever
    vector_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 10})

    retrieved_docs = vector_retriever.invoke(user_query)

    if not retrieved_docs:
        return {
            "content": "No relevant documents found.",
            "artifact": []
        }

    # Serialize for debugging
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )

    return {
        "content": serialized,
        "artifact": retrieved_docs
    }

draft_prompt_template = PromptTemplate(
    template="""
You are an academic research assistant that helps users with their need. you have the following context from academic papers:

{context}

The user asked: {question}

Please produce a short DRAFT answer. 
It doesn't have to be fully polished or perfectly accurate—just a best-effort draft.

Additionally, **please include direct quotes from the provided context in square brackets [like this],
and specify which line or reference from the context you are quoting.**
""",
    input_variables=["context", "question"]
)

refine_prompt_template = PromptTemplate(
    template="""
We have the following context from academic papers:

{context}

We also have a DRAFT answer:

{draft_answer}

Please refine and correct the DRAFT answer to ensure it is accurate, coherent, 
and directly addresses the user's question. Provide a final, high-quality answer.
""",
    input_variables=["context", "draft_answer"]
)



draft_chain = LLMChain(llm=small_llm, prompt=draft_prompt_template)
refine_chain = LLMChain(llm=big_llm,   prompt=refine_prompt_template)


@tool()
def answer_with_speculative_rag(user_query: str):
    """
    1) Use 'retrieve' to get relevant docs
    2) Use a small LLM to produce a 'draft' answer
    3) Use a bigger LLM to refine/correct the draft
    4) Return the final refined answer
    """
    iterations = 5
    llm_query = generate_query(user_query)
    print(llm_query)
    response_dict = retrieve(llm_query)
    docs = response_dict["artifact"]
    
    if not docs:
        return response_dict["content"]  # "No relevant docs" or error msg

    # Combine top docs into a single context
    top_docs = docs[:10]  # limit how many docs to pass
    context_text = "\n\n".join(doc.page_content for doc in top_docs)

    # Produce a draft
    draft_answer = draft_chain.invoke({"context": context_text, "question": user_query})
    
    # Refine the draft
    final_answer = refine_chain.invoke({
        "context": context_text,     
        "draft_answer": draft_answer
    })

    return final_answer   # Contains bith the context and the final answer(used to ensure llm is using context gotten). 
                          # But will be made to only return the final answer

# Example
if __name__ == "__main__":
    test_query = '''
    I am researching how deep learning can be applied to the early detection of neurodegenerative diseases such as Alzheimer's and Parkinson's. I am particularly interested in papers that discuss convolutional neural networks (CNNs) or transformer-based models for medical image analysis, especially MRI or CT scans. The papers should focus on explainability and feature extraction techniques rather than just model performance. I would also prefer research published after 2021 and categorized under either Machine Learning (cs.LG) or Medical Imaging (eess.IV)

    '''
    final_answer = answer_with_speculative_rag(test_query)
    print("=== Final Answer ===")
    print(final_answer)