import os
import time
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

_cached_pipelines = {}


def load_llm(model_name: str) -> pipeline:
    """Load and cache a HuggingFace seq2seq model for text generation."""
    global _cached_pipelines
    if model_name in _cached_pipelines:
        return _cached_pipelines[model_name]

    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=token)

    # Use MPS (Apple Silicon GPU) if available
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        token=token,
        low_cpu_mem_usage=True,
    ).to(device)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=1,
        device=-1,  # model already placed; avoid duplicate placement
    )
    _cached_pipelines[model_name] = pipe
    return pipe


def generate_response(
    llm_pipeline,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> tuple[str, str]:
    """
    Generate a response using either the HuggingFace Cloud Inference API or a
    local pipeline (Apple Silicon MPS / CPU fallback).

    Returns:
        (answer_text, inference_source)  where inference_source is one of:
            "☁️ HuggingFace Cloud API"
            "💻 Local M1 (MPS)"
            "💻 Local CPU"
    """
    if isinstance(llm_pipeline, str):
        model_id = llm_pipeline
        token = os.getenv("HUGGINGFACE_HUB_TOKEN")

        try:
            import requests

            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"

            api_url = f"https://api-inference.huggingface.co/models/{model_id}"
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                },
            }
            t0 = time.time()
            response = requests.post(api_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            elapsed = time.time() - t0
            res_json = response.json()

            if isinstance(res_json, list) and len(res_json) > 0:
                text = res_json[0].get("generated_text", "").strip()
            elif isinstance(res_json, dict) and "generated_text" in res_json:
                text = res_json["generated_text"].strip()
            else:
                raise ValueError(f"Unexpected response format: {res_json}")

            source = f"☁️ HuggingFace Cloud API ({elapsed:.1f}s)"
            return text, source

        except Exception as e:
            print(f"Cloud inference failed ({e}). Falling back to local inference...")
            local_pipeline = load_llm(model_id)
            return _generate_local(local_pipeline, prompt, max_new_tokens, temperature)
    else:
        return _generate_local(llm_pipeline, prompt, max_new_tokens, temperature)


def _generate_local(
    llm_pipeline, prompt: str, max_new_tokens: int, temperature: float
) -> tuple[str, str]:
    """Run inference locally and return (answer_text, inference_source)."""
    device_label = "M1 (MPS)" if torch.backends.mps.is_available() else "CPU"
    t0 = time.time()
    output = llm_pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=False,
        truncation=True,
    )
    elapsed = time.time() - t0
    text = output[0]["generated_text"].strip()
    source = f"💻 Local {device_label} ({elapsed:.1f}s)"
    return text, source
