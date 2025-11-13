from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from app.components.llm import load_llm
from app.components.vector_store import load_vector_store, save_vector_store
from app.components.pdf_loader import load_excel_files, create_text_chunks  # fixed imports

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

CUSTOM_PROMPT_TEMPLATE = """Given the job description or query below, recommend the 6-10 most relevant SHL assessments from the catalog. Include the assessment name and the URL.

Context:
{context}

Query:
{question}

Answer:
"""

def set_custom_prompt():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

def create_qa_chain():
    try:
        logger.info("Loading vector store for context")
        db = load_vector_store()

        # If no vector store exists, generate it from documents
        if db is None:
            logger.warning("Vector store not found. Creating a new one...")

            documents = load_excel_files()  # load raw documents (Excel rows)
            if not documents:
                raise CustomException("No documents found to build vector store.")

            text_chunks = create_text_chunks(documents)  # split into chunks
            if not text_chunks:
                raise CustomException("Failed to generate text chunks from documents.")

            db = save_vector_store(text_chunks)  # save and return vectorstore
            if db is None:
                raise CustomException("Failed to save and load vector store.")

        llm = load_llm()
        if llm is None:
            raise CustomException("LLM not loaded")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={'k': 5}),
            return_source_documents=False,
            chain_type_kwargs={'prompt': set_custom_prompt()}
        )

        logger.info("Successfully created the QA chain")
        return qa_chain

    except Exception as e:
        error_message = CustomException("Failed to create QA chain", e)
        logger.error(str(error_message))
        return None
