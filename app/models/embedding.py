from langchain.embeddings import HuggingFaceEmbeddings


def download_embedding():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding = HuggingFaceEmbeddings(
        model_name = model_name
    )
    return embedding
