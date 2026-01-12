from pydantic import PrivateAttr
import asyncio
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from llama_index.core.embeddings.multi_modal_base import BaseEmbedding
from sentence_transformers import SentenceTransformer
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import logging

logger = logging.getLogger(__name__)

import config

def HuggingFaceWrapper():
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embed_model

def create_hf_llm(
    temperature: float = config.TEMPERATURE,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    decoding_method: str = "sample",
):
    tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        config.LLM_MODEL_ID,
        device_map="auto",
        dtype="auto"
        )
    try:

        llm = HuggingFaceLLM(
            model_name=config.LLM_MODEL_ID,
            tokenizer_name=config.LLM_MODEL_ID,
            max_new_tokens=max_new_tokens,
            model=model,
            messages_to_prompt=None,
            completion_to_prompt=None,
            generate_kwargs={
                "temperature": temperature,
                "top_k": config.TOP_K,
                "top_p": config.TOP_P,
                "do_sample": decoding_method == "sample",
            }
        )

        logger.info(f"Created HuggingFace LLM model: {config.LLM_MODEL_ID}")
        return llm

    except Exception as e:
        logger.error(f"Failed to create HuggingFace LLM: {e}")
        return None

def change_llm_model(new_model_id: str) -> None:
    """Change the LLM model to use.
    
    Args:
        new_model_id: New LLM model ID to use.
    """
    global config
    config.LLM_MODEL_ID = new_model_id
    logger.info(f"Changed LLM model to: {new_model_id}")