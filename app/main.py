from flask import Flask, request, render_template, jsonify
#from models.vector_store import VectorStore
from models.vector_store import VectorStore
#from services.llm_service import LLMService
from services.llm_service import LLMService
from services.storage_service import S3Storage
#from config import config
from config import config

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import logging
import os
import time
import shutil

# -------------------------
# Flask & service setup
# -------------------------
app = Flask(__name__)

vector_store = VectorStore(config.VECTOR_DB_PATH)
storage_service = S3Storage()
llm_service = LLMService(vector_store)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# -------------------------
# Utility: process document
# -------------------------
def process_document(file):
    """Loads and splits the uploaded document into text chunks."""
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    
    try:
        file.save(temp_path)
        logger.debug(f"File temporarily saved at {temp_path}")

        
        start_time = time.time()
        if file.filename.endswith('.pdf'):
            loader = PyPDFLoader(temp_path)
        elif file.filename.endswith('.txt'):
            loader = TextLoader(temp_path)
        elif file.filename.endswith('.docx'):
            loader = Docx2txtLoader(temp_path)
        else:
            raise ValueError("Unsupported file type")

        documents = loader.load()
        logger.debug(f"Document loaded in {time.time() - start_time:.2f}s, total pages: {len(documents)}")

        # Text splitter configuration
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        start_time = time.time()
        text_chunks = text_splitter.split_documents(documents)
        logger.debug(f"Document split into {len(text_chunks)} chunks in {time.time() - start_time:.2f}s")

        return text_chunks

    finally:
        # Clean up temp directory safely
        shutil.rmtree(temp_dir, ignore_errors=True)


# -------------------------
# Routes
# -------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_documents():
    try:
        logger.debug("Upload endpoint called")

        if 'file' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('file')
        if not files or all(file.filename.strip() == '' for file in files):
            return jsonify({'error': 'No files selected'}), 400

        results = []

        for file in files:
            filename = file.filename
            if not filename:
                continue

            logger.info(f"Processing file: {filename}")

            # Validate extension
            if not filename.lower().endswith(('.txt', '.pdf', '.docx')):
                logger.warning(f"Unsupported file type: {filename}")
                results.append({'filename': filename, 'error': 'Unsupported file type'})
                continue

            try:
                # Step 1: Load and split text
                text_chunks = process_document(file)
            except Exception as e:
                logger.exception(f"Error processing document '{filename}'")
                results.append({'filename': filename, 'error': f'Processing error: {str(e)}'})
                continue

            # Step 2: Upload file to S3
            try:
                start_time = time.time()
                file.seek(0)
                storage_service.upload_file(file, filename)
                logger.debug(f"S3 upload completed in {time.time() - start_time:.2f}s")
            except Exception as e:
                logger.exception(f"Error uploading '{filename}' to S3")
                results.append({'filename': filename, 'error': f'S3 upload error: {str(e)}'})
                continue

            # Step 3: Add to vector store
            try:
                start_time = time.time()
                vector_store.add_documents(text_chunks)
                logger.debug(f"Vector store indexing done in {time.time() - start_time:.2f}s")
            except Exception as e:
                logger.exception(f"Error adding chunks of '{filename}' to vector store")
                results.append({'filename': filename, 'error': f'Vector store error: {str(e)}'})
                continue

            results.append({
                'filename': filename,
                'message': 'Uploaded and processed successfully',
                'chunks_processed': len(text_chunks)
            })

        return jsonify({'results': results}), 200

    except Exception as e:
        logger.exception("Unexpected error during upload")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


@app.route('/query', methods=['POST'])
def query():
    data = request.json
    if not data or 'question' not in data or 'session_id' not in data:
        return jsonify({'error': 'Missing question or session_id'}), 400

    try:
        response = llm_service.get_response(data['question'], data['session_id'])
        return jsonify({'response': response})
    except Exception as e:
        logger.exception("Error in query route")
        return jsonify({'error': str(e)}), 500


# -------------------------
# Run the app
# -------------------------
if __name__ == '__main__':
    # Run with debug=False for faster uploads
    app.run(host='0.0.0.0', port=8080, debug=False)
