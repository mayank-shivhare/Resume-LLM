import os
import time
from huggingface_hub import InferenceClient
from openai import OpenAI

_cached_clients = {}


def get_inference_client() -> InferenceClient:
    """Get or create a cached Hugging Face InferenceClient."""
    global _cached_clients
    if "default" in _cached_clients:
        return _cached_clients["default"]
    
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise ValueError(
            "HUGGINGFACE_HUB_TOKEN not found in environment. "
            "Please add your Hugging Face token to .env file. "
            "Get a token at: https://huggingface.co/settings/tokens"
        )
    client = InferenceClient(token=token)
    _cached_clients["default"] = client
    return client


def get_openai_client() -> OpenAI:
    """Get OpenAI-compatible client for Hugging Face router."""
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise ValueError(
            "HUGGINGFACE_HUB_TOKEN not found in environment. "
            "Please add your Hugging Face token to .env file. "
            "Get a token at: https://huggingface.co/settings/tokens"
        )
    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=token,
    )


def generate_response(
    llm_pipeline,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> tuple[str, str]:
    """
    Generate a response using Hugging Face Inference Providers via OpenAI-compatible API.

    Returns:
        (answer_text, inference_source) where inference_source is:
            "☁️ HuggingFace Inference Providers"
    """
    if isinstance(llm_pipeline, str):
        model_id = llm_pipeline
        client = get_openai_client()
        
        t0 = time.time()
        try:
            # Use OpenAI-compatible chat completions endpoint
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            raise RuntimeError(
                f"Unable to reach Hugging Face inference endpoint. "
                f"Check your internet connection and that your HF token has inference permissions. "
                f"Underlying error: {e}\n\nFull traceback:\n{error_details}"
            ) from e
        elapsed = time.time() - t0

        text = response.choices[0].message.content.strip()
        source = f"☁️ HuggingFace Inference Providers ({elapsed:.1f}s)"
        return text, source
    raise ValueError("Expected model_id as string.")
