
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.vectorstores import InMemoryVectorStore
from src.data_ingestion.arxiv.utils import fetch_arxiv_papers, extract_link, extract_abstract, parse_papers
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
entries = parse_papers(xml_papers)
abstracts = extract_abstract(entries)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#vector_store = InMemoryVectorStore(embeddings)

""" when its time for deployment lets host chroma as a remote server ore deploy using docker """

PERSIST_DIR = "chroma_db"
os.makedirs(PERSIST_DIR, exist_ok=True)

vector_store = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

# Load and chunk contents of the blog
'''
"id": entry["id"],
            "title": entry["title"],
            "summary": entry["summary"].strip(),
            "authors": [author["name"] for author in entry["author"]],
            "published": entry["published"],
            "pdf_link": pdf_link
'''
docs = []
for entry in entries:
    # convert the returned results to Document object and encoding all the metadata 
    doc = Document(page_content=entry["summary"],metadata = {"title":entry["title"],
                                                              "authors":entry["authors"],
                                                              "published":entry["published"],
                                                              "link":entry["pdf_link"]})
    
    # creating a list of documents 
    docs.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    all_splits = text_splitter.split_documents(docs)

# Index chunks into Chroma
_ = vector_store.add_documents(all_splits)

# Save the Chroma database persistently
vector_store.persist()

# Save the Chroma database persistently
vector_store.persist()
# Index chunks


# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})
    retrieved_docs = retriever.get_relevant_documents(state["question"]) #note this still uses similarity search
    #retrieved_docs = vector_store.similarity_search(state["question"])
    if not retrieved_docs:
       return {"context": [], "answer": "No relevant documents found."}
    
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

result = graph.invoke({"question": "What are transformers?"})

print(f'Context: {result["context"]}\n\n')
print(f'Answer: {result["answer"]}')