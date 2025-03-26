
from operator import itemgetter
from typing import List

from langchain_openai.chat_models import ChatOpenAI
from langchain_mongodb import MongoDBChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import os
import logging
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from langchain_community.adapters.openai import convert_dict_to_message

load_dotenv()
# uri = os.getenv("MONGODB_URI")
# if not uri:
#     raise Exception("MongoDB URI not found. Please set 'MONGODB_URI' in the environment variables.")

# # Initialize the MongoDB client
# client = MongoClient(uri, server_api=ServerApi('1'))
# db = client["chat_history"]
# collection = db["test_chat_history"]

# # Function to add a conversation and maintain a fixed window size
# def add_conversation(user, conversation, window_size=3):
#     collection.update_one(
#         {"user": user},
#         {"$push": {"conversations": {"$each": [conversation], "$slice": -window_size}}},
#         upsert=True
#     )


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

store = {} # <--------- replace this with mongoDB or some other database and then use it to get session by id 

def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]


class Memory():
    def __init__(self):
        load_dotenv()
        self.uri = os.getenv("MONGODB_URI")
        if not self.uri:
            raise Exception("MongoDB URI not found. Please set 'MONGODB_URI' in the environment variables.")
    
    def get_session_query_generator(self,session_id:str) -> BaseChatMessageHistory:
        current_message= MongoDBChatMessageHistory(
                session_id=session_id,
                connection_string=self.uri,
                database_name="chat-history",
                collection_name="user_history_generator",
                history_size=10
            )
        return current_message
    
    def get_session_query_responder(self,session_id:str) -> BaseChatMessageHistory:
        current_message= MongoDBChatMessageHistory(
                session_id=session_id,
                connection_string=self.uri,
                database_name="chat-history",
                collection_name="user_history_responder",
                history_size=10
            )
        return current_message
        