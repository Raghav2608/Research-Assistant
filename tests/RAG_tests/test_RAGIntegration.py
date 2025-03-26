from backend.src.RAG.query_generator import ResearchQueryGenerator
from backend.src.RAG.query_responder import QueryResponder 
from backend.src.RAG.retrieval_engine import RetrievalEngine
import pytest
import os
import logging
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from backend.src.data_ingestion.data_pipeline import DataPipeline
from langchain_core.documents import Document

@pytest.fixture
def setup_rag_system():
    """Set up the RAG system components with test configurations."""
     # To see the query generated
    logging.basicConfig()
    logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

    load_dotenv()
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        raise EnvironmentError("openai key not set in environment.")
    os.environ["USER_AGENT"] = "myagent" # Always set a user agent
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)

    query_gen = ResearchQueryGenerator(openai_api_key=OPENAI_API_KEY,session_id="test")
    retriever = RetrievalEngine(openai_api_key=OPENAI_API_KEY)
    retriever.vector_store = Chroma(collection_name="test_collection", embedding_function=embeddings)
    responder = QueryResponder(openai_api_key=OPENAI_API_KEY,session_id="test")
    
    return query_gen, retriever, responder

@pytest.mark.integration
def test_rag_good_prompt_pipeline(setup_rag_system):
    """Integration test for the full RAG pipeline."""
    query_gen, retriever, responder = setup_rag_system

    # Step 1: Query Generation
    user_input = "What are the latest advances in reinforcement learning?"
    generated_queries = query_gen.generate(user_input)
    assert isinstance(generated_queries, list)  # Ensure query is generated
    
    doc1 = Document(
    page_content="Deep Q-Networks (DQN) have revolutionized reinforcement learning by integrating deep learning with Q-learning, enabling RL agents to perform well in high-dimensional state spaces.",
    metadata={
        "title": "Deep Q-Networks and Their Impact",
        "published": "2024",
        "link": "https://www.aijournal.com/dqn-advancements"
    }
    )

    doc2 = Document(
        page_content="Proximal Policy Optimization (PPO) has become the gold standard for training RL agents due to its stability and efficiency in optimizing policy gradients.",
        metadata={
            "title": "Advances in Policy Gradient Methods",
            "published": "2023",
            "link": "https://www.neurips2023.com/ppo-research"
        }
    )

    doc3 = Document(
        page_content="Multi-Agent Reinforcement Learning (MARL) has enabled complex decision-making in cooperative and competitive settings, advancing applications in robotics and autonomous systems.",
        metadata={
            "title": "The Rise of Multi-Agent RL",
            "published": "2024",
            "link": "https://www.icml2024.com/marl-developments"
        }
    )

    doc4 = Document(
        page_content="Offline RL techniques, such as conservative Q-learning (CQL), allow agents to learn from fixed datasets without direct environment interaction, making RL applicable to real-world scenarios.",
        metadata={
            "title": "Offline Reinforcement Learning: A New Frontier",
            "published": "2024",
            "link": "https://www.rlreview.com/offline-rl"
        }
    )

    # List of documents
    docs = [doc1, doc2, doc3, doc4]
    retriever.split_and_add_documents(docs=docs) # Add documents to ChromaDB (save)
    
    # Step 2: Retrieval
    retrieved_docs = retriever.retrieve(generated_queries)
    assert isinstance(retrieved_docs, list)
    assert len(retrieved_docs) > 0  # Ensure some documents are retrieved

    # Step 3: Response Generation
    final_response = responder.generate_answer(retrieved_docs=retrieved_docs,user_query=user_input)
    assert isinstance(final_response, str)
    assert len(final_response) > 0  # Ensure response is generated

    print("Integration Test Passed ✅")

@pytest.mark.integration
def test_rag_bad_prompt_pipeline(setup_rag_system):
    query_gen, retriever, responder = setup_rag_system

    # Step 1: Query Generation
    user_input = "hi"
    generated_queries = query_gen.generate(user_input)
    assert  generated_queries == "ERROR" # Ensure query is generated

    # Step 2: Response Generation
    final_response = responder.generate_answer(retrieved_docs=[],user_query=user_input)
    assert isinstance(final_response, str)
    assert len(final_response) > 0  # Ensure response is generated


