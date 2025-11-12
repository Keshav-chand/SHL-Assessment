import os
import pandas as pd
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

logger = get_logger(__name__)

def load_excel_files():  # Load Excel instead of PDF
    try:
        if not os.path.exists(DATA_PATH):
            raise CustomException("Data path does not exist")
        
        logger.info(f"Loading Excel files from {DATA_PATH}")
        
        # Assume Excel file has columns: 'Query', 'Assessment', 'URL'
        excel_files = [f for f in os.listdir(DATA_PATH) if f.endswith((".xls", ".xlsx"))]
        if not excel_files:
            logger.warning("No Excel files found")
            return []

        documents = []
        for file in excel_files:
            df = pd.read_excel(os.path.join(DATA_PATH, file))
            for _, row in df.iterrows():
                # Combine all columns into a single text string
                text = " | ".join([str(row[col]) for col in df.columns])
                documents.append(Document(page_content=text))
        
        logger.info(f"Successfully loaded {len(documents)} documents from Excel")
        return documents

    except Exception as e:
        error_message = CustomException("Failed to load Excel", e)
        logger.error(str(error_message))
        return []


def create_text_chunks(documents):  # Reuse same text splitter
    try:
        if not documents:
            raise CustomException("No documents were found")
        logger.info(f"Splitting {len(documents)} documents into chunks")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

        text_chunks = text_splitter.split_documents(documents)

        logger.info(f"Generated {len(text_chunks)} text chunks")
        return text_chunks
    except Exception as e:
        error_message = CustomException("Failed to generate chunks", e)
        logger.error(str(error_message))
        return []
