import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

HF_TOKEN = os.environ.get("HF_TOKEN")
API_KEY = os.environ.get("COHERE_API_KEY")  

HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = os.path.join(BASE_DIR, "vectorstore", "db_faiss")
DATA_PATH = os.path.join(BASE_DIR, "data")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
