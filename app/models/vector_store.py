import chromadb
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from models.embedding import download_embedding



class VectorStore:
    def __init__(self, path):
        self.embedding = download_embedding()
        self.vector_store = Chroma(
            persist_directory = path,
            embedding_function = self.embedding
        )

    def add_documents(self, documents):
        self.vector_store.add_documents(documents)

    def similarity_search(self, query, k=4):
        return self.vector_store.similarity_search(query, k=k)
