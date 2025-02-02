
from langchain import hub
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph, MessagesState
from src.data_ingestion.arxiv.utils import fetch_arxiv_papers,parse_papers
from langchain_chroma import Chroma
import getpass
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode




load_dotenv()
os.environ['USER_AGENT'] = 'myagent'
if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

'''
search_query = "all:transformers+machine+learning+review"
start = 0
max_results = 25

xml_papers = fetch_arxiv_papers(search_query, start, max_results)
entries = parse_papers(xml_papers)
'''

#vector_store = InMemoryVectorStore(embeddings)

""" when its time for deployment lets host chroma as a remote server ore deploy using docker """

PERSIST_DIR = "chroma_db"
os.makedirs(PERSIST_DIR, exist_ok=True)

vector_store = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

'''
# Load and chunk contents of the blog
docs = []
for entry in entries:
    # convert the returned results to Document object and encoding all the metadata 
    doc = Document(page_content=entry["summary"],metadata = {"title":entry["title"],
                                                              "published":entry["published"],
                                                              "link":entry["pdf_link"]})
    
    # creating a list of documents 
    docs.append(doc)
'''
@tool(response_format="content")
def generate_search_query(user_prompt:str):
    """Generate answer."""
    system_message_content = (
        "You are research searching agent your job is to generate effective research queries from given user prompts"
        "that can be put into a research database for searching relevant papers, only give a single query and no explanation"
        "If you don't know the answer,say I don't know "
        "don't know."
        "Here is an example for the query and response"
        "Question: I want to find research on Deep Time series analysis and what deep learning techniques are being used currently for the same"
        'Answer: (deep learning) AND (time series analysis) AND (survey OR review OR recent advancements)'
        "\n\n"
        f"{user_prompt}"
    )
    generated_query = llm.invoke(system_message_content)
    return generated_query 

def search_and_document(search_query:str):
    start = 0
    max_results = 25
    xml_papers = fetch_arxiv_papers(search_query, start, max_results)
    entries = parse_papers(xml_papers)
    docs = []
    for entry in entries:
        # convert the returned results to Document object and encoding all the metadata 
        doc = Document(page_content=entry["summary"],metadata = {"title":entry["title"],
                                                                "published":entry["published"],
                                                                "link":entry["pdf_link"]})
        
        # creating a list of documents 
        docs.append(doc)
    return docs

def split_and_add_documents(docs:list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    all_splits = text_splitter.split_documents(docs)
    
    # Index chunks into Chroma
    _ = vector_store.add_documents(all_splits)



'''
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
all_splits = text_splitter.split_documents(docs)
  
# Index chunks into Chroma
_ = vector_store.add_documents(all_splits)
'''

def clean_search_query(search_query:str):
    search_query = search_query.replace(" ","+")
    
    return search_query


# Define application steps
@tool(response_format="content_and_artifact")
def retrieve(user_query:str):
    """Retrieve information related to a query."""
    generated_query = generate_search_query(user_query)
    cleaned_search_query = clean_search_query(generated_query.content)
    docs = search_and_document(cleaned_search_query)
    split_and_add_documents(docs)
    
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 4})
    retrieved_docs = retriever.get_relevant_documents(user_query) #note this still uses similarity search
    
    #retrieved_docs = vector_store.similarity_search(state["question"])
    if not retrieved_docs:
       return "", "No relevant documents found."
    
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    
    return serialized, retrieved_docs


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])

# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for summarizing research based on retrieved context and user prompt"
        "Use the following pieces of retrieved context to summarize and present it in a list form  "
        "If you don't know the answer, say that you "
        "don't know."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}







'''
def generate(state: State):
    print(len(state["context"]))
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    {"answer": response.content}
    return  {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

result = graph.invoke({"question": "What are transformers?"})

print(f'Context: {result["context"]}\n\n')
print(f'Answer: {result["answer"]}')
'''