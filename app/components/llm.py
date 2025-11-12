from langchain_cohere import ChatCohere
from app.config.config import API_KEY
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def load_llm(model_name: str = "command-a-03-2025", cohere_api_key: str = API_KEY):
    """
    Load and return a Cohere LLM via LangChain (latest supported version).
    """
    try:
        logger.info(f"Loading LLM from Cohere using model: {model_name}...")

        llm = ChatCohere(
            model=model_name,
            temperature=0.3,
            max_tokens=256,
            cohere_api_key=cohere_api_key,
        )

        logger.info("LLM loaded successfully from Cohere.")
        return llm

    except Exception as e:
        error_message = CustomException("Failed to load an LLM from Cohere", e)
        logger.error(str(error_message))
        return None
