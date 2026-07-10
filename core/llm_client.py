# core/llm_client.py
import os
import time
import json
import hashlib
from dotenv import load_dotenv
from openai import OpenAI
from openai import APIStatusError, APIConnectionError, RateLimitError

load_dotenv()

# OpenCode Zen exposes an OpenAI-compatible /chat/completions endpoint.
# Get a key at https://opencode.ai (Zen) and put it in .env as OPENCODE_API_KEY.
_client = None

def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENCODE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENCODE_API_KEY is not set. Get a key at https://opencode.ai and add it to .env"
            )
        _client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("OPENCODE_BASE_URL", "https://opencode.ai/zen/v1"),
        )
    return _client

# Default model can be overridden via .env (OPENCODE_MODEL).
DEFAULT_MODEL = os.getenv("OPENCODE_MODEL", "deepseek-v4-flash-free")

CACHE_PATH = "corpus/processed/llm_cache.json"

def _load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

def _cache_key(prompt, model):
    return hashlib.sha256(f"{model}::{prompt}".encode()).hexdigest()



def ask_llm(prompt, model=DEFAULT_MODEL, max_retries=5, use_cache=True):
    cache = _load_cache() if use_cache else {}
    key = _cache_key(prompt, model)

    if use_cache and key in cache:
        print("    (using cached response — no API call made)")
        return cache[key]

    for attempt in range(max_retries):
        try:
            response = _get_client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.choices[0].message.content

            if use_cache:
                cache[key] = answer
                _save_cache(cache)

            return answer
        except (RateLimitError, APIConnectionError) as e:
            wait_time = 5 * (attempt + 1)   # was 15 — lighter backoff for a more generous provider
            print(f"    (temporary error, waiting {wait_time}s before retry {attempt+1}/{max_retries}...)")
            time.sleep(wait_time)
        except APIStatusError as e:
            if e.status_code in (429, 500, 502, 503, 504):
                wait_time = 5 * (attempt + 1)   # was 15 — lighter backoff for a more generous provider
                print(f"    (temporary error, waiting {wait_time}s before retry {attempt+1}/{max_retries}...)")
                time.sleep(wait_time)
            else:
                raise

    raise RuntimeError(f"Failed after {max_retries} retries.")
