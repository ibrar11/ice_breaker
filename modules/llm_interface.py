from langchain_community.embeddings import HuggingFaceEmbeddings
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from llama_index.llms.ibm import WatsonxLLM
import logging

logger = logging.getLogger(__name__)

import config

def create_huggingface_embedding():
    huggingFace_embedding = HuggingFaceEmbeddings()
    return huggingFace_embedding

def create_watsonx_llm(
    temperature: float = config.TEMPERATURE,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    decoding_method: str = "sample"
):
    parameters = {
        GenParams.DECODING_METHOD: decoding_method,
        GenParams.MAX_NEW_TOKENS: max_new_tokens,
        GenParams.MIN_NEW_TOKENS: config.MIN_NEW_TOKENS,
        GenParams.TEMPERATURE: temperature,
        GenParams.TOP_K: config.TOP_K,
        GenParams.TOP_P: config.TOP_P,
    }

    watsonx_llm = WatsonxLLM(
        model_id=config.LLM_MODEL_ID,
        url=config.MOCK_DATA_URL,
        apikey=config.PROXYCURL_API_KEY,
        project_id=config.WATSONX_PROJECT_ID,
        params=parameters, 
    )
    logger.info(f"Created Watsonx LLM model: {config.LLM_MODEL_ID}")
    return watsonx_llm

def change_llm_model(new_model_id: str) -> None:
    """Change the LLM model to use.
    
    Args:
        new_model_id: New LLM model ID to use.
    """
    global config
    config.LLM_MODEL_ID = new_model_id
    logger.info(f"Changed LLM model to: {new_model_id}")