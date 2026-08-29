# RAG-Based-Knowledge-Management-System

# A Retrieval-Augmented LLM Frame work for Multi-Document Q&A

A **Retrieval-Augmented Generation (RAG)** based Knowledge Management System that enables users to upload documents and ask natural-language questions. The system retrieves relevant document content and uses an LLM to generate contextual and evidence-based answers.

## 🚀 Features

* Supports **PDF, DOCX, and TXT** documents
* Automatic document loading and text chunking
* Semantic search using vector embeddings
* Conversational multi-turn Q&A
* Secure document storage using **Amazon S3**
* Context-aware responses using **Google Gemini 2.0 Flash**

## 🛠️ Tech Stack

* **Python**
* **LangChain**
* **Google Gemini 2.0 Flash**
* **Hugging Face all-MiniLM-L6-v2**
* **ChromaDB**
* **Amazon S3 / boto3**
* **ConversationBufferMemory**

## 🔄 Workflow

```text
Documents
   ↓
LangChain Loaders
   ↓
Text Chunking
   ↓
MiniLM Embeddings
   ↓
ChromaDB
   ↓
Semantic Retrieval
   ↓
Gemini 2.0 Flash
   ↓
Contextual Answer
```

The system uses 1000-character chunks with 200-character overlap for document processing.

## 📊 Results

| Metric    | LLM Only |       RAG |
| --------- | -------: | --------: |
| Accuracy  |    84.6% | **90.7%** |
| Precision |    83.8% | **88.9%** |
| Recall    |    82.4% | **89.5%** |

The proposed RAG system achieved better accuracy, precision, and recall than the standalone LLM.

## 🎯 Objective

To build a **secure, scalable, and context-aware document question-answering system** that improves factual accuracy and reduces LLM hallucinations through semantic retrieval.

## 👩‍💻 Authors

**Yeluri Harsha Vardhini** and team
Department of Information Technology
Lakireddy Bali Reddy College of Engineering
