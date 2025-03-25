from backend.src.RAG.retrieval_engine import RetrievalEngine
from dotenv import load_dotenv
import os
from langchain_core.documents import Document

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise EnvironmentError("openai key not set in environment.")

retrieval_engine = RetrievalEngine(openai_api_key=OPENAI_API_KEY)

doc1 = Document(page_content="nofjnafjqofnqofn",metadata={
                                "title": "title",
                                "published": "published",
                                "link": "l1"
                                })

doc2 = Document(page_content='Sarita',metadata={
                                "title": "title",
                                "published": "published",
                                "link": "l2"
                                })


retrieval_engine.split_and_add_documents([doc1,doc2])

doc3 = Document(page_content='Raghav',metadata={
                                "title": "title",
                                "published": "published",
                                "link": "l2"
                                })

retrieval_engine.split_and_add_documents([doc3,doc1])

print(retrieval_engine.vector_store.get())

