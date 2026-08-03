#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

def download_embedding():
    
    model_name = "BAAI/bge-large-en-v1.5"
    embedding = HuggingFaceEmbeddings(
        model_name = model_name
    )
    return embedding
