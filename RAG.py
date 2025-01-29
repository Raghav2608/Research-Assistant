
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.vectorstores import InMemoryVectorStore
from src.data_ingestion.arxiv.utils import fetch_arxiv_papers, extract_link
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import getpass
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['USER_AGENT'] = 'myagent'

search_query = "all:transformer"
start = 0
max_results = 3

xml_papers = fetch_arxiv_papers(search_query, start, max_results)
links = extract_link(xml_papers)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#vector_store = InMemoryVectorStore(embeddings)

""" when its time for deployment lets host chroma as a remote server ore deploy using docker """

Persist_dir = "/shared_storage/chroma_db"
vector_store = Chroma(persist_directory=Persist_dir, embedding_function=embeddings)

# Load and chunk contents of the blog

for i in range(len(links)):
    loader = PyPDFLoader(links[i])
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    # Index chunks into Chroma
    _ = vector_store.add_documents(all_splits)

# Save the Chroma database persistently
vector_store.persist()

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retriever = vector_store.as_retriever()
    retrieved_docs = retriever.get_relevant_documents(state["question"]) #note this still uses similarity search
    #retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    #response = llm.invoke(messages)
    #{"answer": response.content}
    print(messages)
    return {"answer":messages}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

result = graph.invoke({"question": "What are transfromers?"})

print(f'Context: {result["context"]}\n\n')
print(f'Answer: {result["answer"]}')