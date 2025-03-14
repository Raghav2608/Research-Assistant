import os
import urllib

from typing import Dict, List
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

from src.RAG.query_generator import ResearchQueryGenerator

class RAG:
    def __init__(self, openai_api_key:str):
        os.environ["USER_AGENT"] = "myagent" # Always set a user agent
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
        self.query_generator = ResearchQueryGenerator(openai_api_key=openai_api_key)
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

        summary_memory = ConversationSummaryMemory(
                                                llm=self.llm,
                                                memory_key="summary_history",
                                                input_key="question", # The name of the user input in chain
                                                output_key="text" # The name of the chain's output
                                                )

        # Keep the last N exchanges in full
        window_memory = ConversationBufferWindowMemory(
                                                        k=3, # Number of most recent exchanges to keep 
                                                        memory_key="window_history",
                                                        input_key="question",
                                                        output_key="text"
                                                        )

        # Combined memory 
        self.combined_memory = CombinedMemory(
                                            memories=[
                                                    summary_memory,
                                                    window_memory
                                                    ]
                                            )


        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        
        answer_prompt_template = """
        You are a helpful research assistant. 
        Below are relevant excepts from academic papers:

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
        self.qa_chain = LLMChain(llm=self.llm, prompt=answer_prompt, memory=self.combined_memory)
    
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
    def split_and_add_documents(self,docs:list[Document]):
        all_splits = self.text_splitter.split_documents(docs)
        
        # Index chunks into Chroma
        self.vector_store.add_documents(all_splits)

    def clean_search_query(self,search_query: str) -> str:
        """Encode the query so it doesn't contain invalid URL chars."""
        # Instead of just replacing spaces with '+', let's do a robust encoding
        return urllib.parse.quote_plus(search_query)

    # Retrieval Function
    def retrieve(self,user_query: str):
        """Retrieve information related to a query."""
        generated_queries = self.query_generator.generate(user_query)

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
        print(generated_queries)
        for query in generated_queries:
            # clean_search_query sanitizes or modifies the docs 
            # so they can be safely used in the next steps
            safe_query = self.clean_search_query(query)
            print(safe_query)
            docs.extend(self.search_and_document(safe_query))

        if not docs:
            return {
                    "content": "No relevant documents found.",
                    "artifact": []
                    }

        self.split_and_add_documents(docs)
        keyword_retriever = BM25Retriever.from_documents(docs) if docs else None

        # Combine into Hybrid Retriever
        if keyword_retriever:
            hybrid_retriever = EnsembleRetriever(
                                                retrievers=[
                                                            self.vector_retriever, 
                                                            keyword_retriever
                                                            ],
                                                weights=[0.5, 0.5]
                                                )
        else:
            hybrid_retriever = self.vector_retriever
        
        retrieved_docs = []
        for query in generated_queries:
            relevant_documents = hybrid_retriever.get_relevant_documents(query)
            retrieved_docs.extend(relevant_documents)

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
    
    def combine_context_and_question(self, context_text:str, user_query:str) -> Dict[str, str]:
        return {"context": context_text, "question": user_query}

    # retrieval + final answer generation
    def answer_with_rag(self,user_query:str):
        """
        1. Use the 'retrieve' function to get relevant documents.
        2. Stuff those docs into the QA chain as context.
        3. Return a final answer generated by the LLM.
        """
        response_dict = self.retrieve(user_query)
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

        combined_query = self.combine_context_and_question(context_text, user_query)
        answer = self.qa_chain.run(combined_query)
        return answer