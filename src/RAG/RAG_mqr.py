import os

from typing import Dict, List, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.data_ingestion.arxiv.utils import fetch_arxiv_papers,parse_papers
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory, CombinedMemory
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma

class RAG:
    def __init__(self, openai_api_key:str):
        os.environ["USER_AGENT"] = "myagent" # Always set a user agent
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)

        self.PERSIST_DIR = "chroma_db"
        if not os.path.exists(self.PERSIST_DIR):
            os.makedirs(self.PERSIST_DIR)
        
        self.vector_store = Chroma(persist_directory=self.PERSIST_DIR, embedding_function=self.embeddings)
        
        # Initialize Vector Store Retriever
        self.vector_retriever = self.vector_store.as_retriever(
                                                            search_type="mmr",
                                                            search_kwargs={"k": 4, "fetch_k": 4}
                                                            )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    
    def convert_entries_to_docs(self, entries:List[Dict[str, str]]) -> List[Document]:
        docs = []
        for entry in entries:
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
    def split_and_add_documents(self, docs:List[Document]):
        all_splits = self.text_splitter.split_documents(docs)
        
        # Index chunks into Chroma
        self.vector_store.add_documents(all_splits)

    # Retrieval Function
    def retrieve(self, user_query) -> List[Dict[str, Any]]:
        # Check if we have any documents first:
        print(user_query)
        results = self.vector_store.similarity_search_with_score(query=user_query, k=5) # Get top 5 results
        print("Number of results: ", len(results))

        # No results found
        if len(results) == 0:
            return []
        
        # >=1 results found

        # Sort documents by score:
        results = sorted(results, key=lambda x: x[1], reverse=True) # Descending order

        retrieved_docs = []
        for doc, score in results:
            print(type(doc))
            print(doc)
            print(score)
            print("\n")
            retrieved_docs.append(doc)

        return retrieved_docs
    