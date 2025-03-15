import os

from typing import Dict, List, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

class RetrievalEngine:
    """
    A class that handles the retrieval of documents based on user queries.

    - First it attempts to retrieve relevant documents from the existing database.
    - If no relevant documents are found:
        - It searches for more documents via data ingestion.
        - It then converts the retrieved entries to documents and adds them to the ChromaDB.
        - It then attempts to retrieve the documents again.
    """
    def __init__(self, openai_api_key:str):
        """
        Initialize the Retrieval Engine with the OpenAI API key and the ChromaDB.

        Args:
            openai_api_key (str): The OpenAI API key for the OpenAI services.
        """
        os.environ["USER_AGENT"] = "myagent" # Always set a user agent
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)

        PERSIST_DIR = "chroma_db"
        if not os.path.exists(PERSIST_DIR):
            os.makedirs(PERSIST_DIR)
        
        self.vector_store = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        
        self.SEARCH_K = 5 # Number of documents to return
        self.FETCH_K = self.SEARCH_K * 3 # Number of documents to fetch
        self.initiate_vector_retriever()

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    def initiate_vector_retriever(self) -> None:
        """
        Initializes the vector retriever for the ChromaDB.
        - Used to initialise / update the vector retriever after adding documents to the ChromaDB.
        """
        self.vector_retriever = self.vector_store.as_retriever(
                                                            search_type="mmr",
                                                            search_kwargs={"k": self.SEARCH_K, "fetch_k": self.FETCH_K}
                                                            )
    
    def convert_entries_to_docs(self, entries:List[Dict[str, str]]) -> List[Document]:
        """
        Converts the retrieved entries into document objects.

        Args:
            entries (List[Dict[str, str]]): A list of retrieved entries containing information
                                            about the research papers.
        """
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
    
    def split_and_add_documents(self, docs:List[Document]) -> None:
        """
        Splits the documents into chunks and adds the chunks to the ChromaDB.

        Args:
            docs (List[Document]): A list of documents to split and add to the ChromaDB.
        """
        all_splits = self.text_splitter.split_documents(docs)

        # Index chunks into Chroma
        self.vector_store.add_documents(all_splits)

        # Update the vector retriever
        self.initiate_vector_retriever() 

    def convert_docs_to_dicts(self, docs:List[Document]) -> List[Dict[str, Any]]:
        """
        Converts the documents into dictionaries containing the page content and metadata.

        Args:
            docs (List[Document]): A list of documents to convert to dictionaries.
        """
        doc_dicts = []
        for doc in docs:
            doc_dict = {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                        }
            doc_dicts.append(doc_dict)
        return doc_dicts
    
    def retrieve(self, user_query:str) -> List[Dict[str, Any]]:
        """
        The main function for retrieving documents based on the user query.
        
        Args:
            user_query (str): The user query to search for relevant documents.
        """
        # Check if we have any documents first:
        print(user_query)

        results = self.vector_store.similarity_search_with_score(query=user_query, k=self.SEARCH_K) # Get top K results
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
        
        retrieved_docs = self.convert_docs_to_dicts(retrieved_docs)
        return retrieved_docs