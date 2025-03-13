
from langchain import hub
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langgraph.graph import START, StateGraph, MessagesState
from src.data_ingestion.arxiv.utils import fetch_arxiv_papers,parse_papers
from langchain_community.vectorstores import Chroma
#from langchain_chroma import Chroma
import getpass
import os
import shutil
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage
#from langgraph.prebuilt import ToolNode
from langchain.retrievers import EnsembleRetriever, BM25Retriever, MultiQueryRetriever
import logging
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory, CombinedMemory
import json

# To see the query generated
logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)



load_dotenv()
os.environ['USER_AGENT'] = 'myagent'
if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

PERSIST_DIR = "chroma_db"
#if os.path.exists(PERSIST_DIR):
  #  shutil.rmtree(PERSIST_DIR)
os.makedirs(PERSIST_DIR, exist_ok=True)

vector_store = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

 # Initialize Vector Store Retriever
vector_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 4}
    )

summary_memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="summary_history",
    input_key="question",    # The name of the user input in chain
    output_key="text"      # The name of the chain's output
)

# Keep the last N exchanges in full
window_memory = ConversationBufferWindowMemory(
    k=3,                     # number of most recent exchanges to keep 
    memory_key="window_history",
    input_key="question",
    output_key="text"
)

# Combined memory 
combined_memory = CombinedMemory(
    memories=[summary_memory, window_memory]
)


# Define Query Generator
@tool(response_format="content")
def generate_search_query(user_prompt: str):
    """Generate multiple variations of a research query while handling edge cases.
    Returns a JSON list of possible queries or an error message."""

    system_prompt = f"""
    You are a research assistant specializing in refining user queries for recent research retrieval or retrieval based on given papers.
    
    Your ONLY output should be valid JSON with no extra text, in the format:
    ["query_variation_1", "query_variation_2", "query_variation_3"]

    If you cannot produce a valid list, return:
    ["ERROR: Failed to generate valid queries. Please try again 2."]


    **Your Responsibilities:**
    1. If the query is **too broad** (e.g., "AI"), make it more specific.
    2. If the query is **ambiguous** (e.g., "bias"), provide different possible meanings.
    3. If the query is **too narrow**, generalize it slightly while keeping it relevant.
    4. If the query is **invalid** (too short, gibberish), return: `"I don't understand. ERROR: Invalid query. Please provide more details."`

    **User Query:** "{user_prompt}"
    """

    generated_query = llm.invoke(system_prompt).content

    try:
        query_variations = json.loads(generated_query)
    except json.JSONDecodeError:
        return ["ERROR: Failed to generate valid queries. Please try again."]

    return query_variations


# Function to fetch and process documents
def search_and_document(search_query: str, limit=5):
    start = 0
    max_results = 100 # or bigger if you want
    xml_papers = fetch_arxiv_papers(search_query, start, max_results)
    entries = parse_papers(xml_papers)

    # Limit the final doc count
    docs = []
    for i, entry in enumerate(entries):
        if i >= limit:
            break
        # convert to Document object or do your existing logic
        doc = Document(
            page_content=entry["summary"],   # Use entry["summary"] to skip parsing
            metadata={
                "title": entry["title"],
                "published": entry["published"],
                "link": entry["pdf_link"],
            },
        )
        docs.append(doc)

    return docs


# Splits and add document to ChromaDB
def split_and_add_documents(docs:list[Document]):
   # existing_docs = vector_store.similarity_search("test query", k=1)

    #if not existing_docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        all_splits = text_splitter.split_documents(docs)
        
        # Index chunks into Chroma
        vector_store.add_documents(all_splits)

import urllib

def clean_search_query(search_query: str) -> str:
    """Encode the query so it doesn't contain invalid URL chars."""
    # Instead of just replacing spaces with '+', let's do a robust encoding
    return urllib.parse.quote_plus(search_query)

# Retrieval Function
@tool()
def retrieve(user_query: str):
    """Retrieve information related to a query."""
    
    generated_queries = generate_search_query(user_query)

    if isinstance(generated_queries, str) and "ERROR" in generated_queries:
        # Return a dict with 'content' and 'artifact' keys
        return {
            "content": generated_queries,
            "artifact": []
        }
    elif isinstance(generated_queries, list):
        # If the entire list is just a single error
        if len(generated_queries) == 1 and "ERROR" in generated_queries[0]:
            return {
                "content": generated_queries[0],
                "artifact": []
            }
    else:
        # If we got something weird (not a list, not a string)
        return {
            "content": "ERROR: Invalid query generation output.",
            "artifact": []
        }

    docs = []
    for query in generated_queries:
        # clean_search_query sanitizes or modifies the docs 
        # so they can be safely used in the next steps
        safe_query = clean_search_query(query)
        docs.extend(search_and_document(safe_query))

    if not docs:
        return {
            "content": "No relevant documents found.",
            "artifact": []
        }

    split_and_add_documents(docs)

    keyword_retriever = BM25Retriever.from_documents(docs) if docs else None

    # Combine into Hybrid Retriever
    if keyword_retriever:
        hybrid_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, keyword_retriever],
            weights=[0.5, 0.5]
        )
    else:
        hybrid_retriever = vector_retriever
    
    retrieved_docs = []
    for query in generated_queries:
        retrieved_docs.extend(
            hybrid_retriever.get_relevant_documents(query)
        )

    if not retrieved_docs:
        return {
            "content": "No relevant documents found.",
            "artifact": []
        }

    serialized = "\n\n".join(
        f"Source: {doc.metadata['link']}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )

    return {
        "content": serialized,
        "artifact": retrieved_docs
    }


answer_prompt_template = """
You are a helpful research assistant. 
Below are relevant excerpts from academic papers:

{context}

The user has asked the following question:
{question}

Please provide a concise, well-structured answer **and include direct quotes or references** from the provided context. 
Use the format [Source: link] (link will be given to you with every paper right after word source).
"""


answer_prompt = PromptTemplate(
    template=answer_prompt_template,
    input_variables=["context", "question"]
)

# LLMChain for the final QA step
qa_chain = LLMChain(llm=llm, prompt=answer_prompt, memory=combined_memory)

def ask_with_context(context, question):
     return {
       "context": context,
       "question": question
    }

# retrieval + final answer generation
@tool()
def answer_with_rag(user_query: str):
    """
    1. Use the 'retrieve' function to get relevant documents.
    2. Stuff those docs into the QA chain as context.
    3. Return a final answer generated by the LLM.
    """
    response_dict = retrieve(user_query)
    # Using the dictionary of content and artifact
    serialized = response_dict["content"]
    docs = response_dict["artifact"]


    # If no docs found, just return the message
    if not docs:
        return serialized  # Likely "No relevant documents found."

    # Limit the number of docs if you have many
    top_docs = docs[:10]  # take top 5 if needed

    # Combine text into a single context string
    context_text = "\n\n".join(doc.page_content for doc in top_docs)

    # Run the final LLM chain
    print("=== CONTEXT TEXT ===")
    print(context_text)
    print("=== END CONTEXT ===")

    answer = qa_chain.run(ask_with_context(context_text, user_query))
    return answer


if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit", "end"]:
            break
        final_answer = answer_with_rag(user_input)
        print("=== Final Answer ===")
        print("Researcher: ",final_answer)



