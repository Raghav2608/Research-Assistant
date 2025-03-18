import os

from typing import List, Dict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory, CombinedMemory

class QueryResponder:
    """
    Class that responds to a user query by combining the context and user query and
    passing it to an LLM model for a context-aware response.    
    """
    def __init__(self, openai_api_key:str):
        os.environ["USER_AGENT"] = "myagent" # Always set a user agent
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
        
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

    def format_documents(self, retrieved_docs:List[str]) -> str:
        """
        Formats the retrieved documents into a single string for the LLM model.

        Args:
            retrieved_docs (List[str]): A list of retrieved documents.
        """
        formatted_content = "\n\n".join(
                                f"Source: {doc['metadata']['link']}\nContent: {doc['page_content']}"
                                for doc in retrieved_docs
                                )
        return formatted_content
    
    def combine_context_and_question(self, context_text:str, user_query:str) -> Dict[str, str]:
        """
        Combines the context and user query into a single dictionary for the LLM model.

        Args:
            context_text (str): The context text to be passed to the LLM model.
            user_query (str): The user query to be passed to the LLM model.
        """
        return {"context": context_text, "question": user_query}
    
    def generate_answer(self, retrieved_docs:List[str], user_query:str) -> str:
        """
        Generates an answer based on the retrieved documents and user query by
        prompting the LLM model.

        Args:
            retrieved_docs (List[str]): A list of retrieved documents.
            user_query (str): The user query.
        """
        formatted_content = self.format_documents(retrieved_docs)
        prompt = self.combine_context_and_question(formatted_content, user_query)
        answer = self.qa_chain.invoke(prompt)
        return answer