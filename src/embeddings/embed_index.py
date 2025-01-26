from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embedding import HuggingFaceEmbeddings

file_path = ".\data\example\research_example.pdf"

# load the pdf of the research paper
pdf_loader = PyPDFLoader(file_path)
documents = pdf_loader.load()

#spits the text into chunks

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(documents)

#embeddings
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2" )

# generating the embeddings for each chunk, WE CAN REMOVE THIS ONCE WE KNOW IT WORKS
# CAUSE THE VECTOR_DB DOES THIS AUTOMATICALLY
embeddings = [embeddings_model.embed_documents(chunk.page_content) for chunk in all_splits]

#indexing the chunk and its embeddings
"""
use Chroma because of its persistence storage , stores metadata alongside
embeddings. ALTHOUGH not optimized for very large-scale datasets(billions of vectors)
. it doesnt use multi-GPU or ann searches (approximate nearest neighbour)"""
# pip install langchain-chroma
from langchain_chroma import Chroma

vector_db = Chroma.from_documents(all_splits, embeddings_model)

""" OR USE FAISS, for large-scale datasets(millions or billions of vectors), very fast 
nearest-neighbor searches. accuracy. although we will need to handle metadata storage and 
database management ourselves. no persistent storage. doesnt store metadata alongisde embeddings unlike chroma.

from langchain_commuinity.vectorstores import FAISS

vector_db= FAISS.from_documents(all_splits, embeddings)

 """