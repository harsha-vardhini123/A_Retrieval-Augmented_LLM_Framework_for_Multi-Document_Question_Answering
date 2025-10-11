import os
from dotenv import load_dotenv

load_dotenv()

class config:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
    AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
    AWS_BUCKET_KEY = os.getenv('AWS_BUCKET_KEY')
    VECTOR_DB_PATH = 'vector_db'