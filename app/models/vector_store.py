import chromadb
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from .embedding import download_embedding


class VectorStore:
    def __init__(self, path):
        self.embedding = download_embedding()
        self.vector_store = Chroma(
            persist_directory = path,
            embedding_function = self.embedding
        )

    def add_documents(self, documents):
        self.vector_store.add_documents(documents)

    def similarity_search(self, query, k=6):
        return self.vector_store.similarity_search(query, k=k)
    
    def as_retriever(self, search_type="similarity", search_kwargs=None):
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs or {"k": 6}
        )
