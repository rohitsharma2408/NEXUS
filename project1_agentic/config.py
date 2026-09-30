import os
import time
from dotenv import load_dotenv

load_dotenv()

READONLY_DATABASE_URL = os.environ.get("READONLY_DATABASE_URL") or os.environ["DATABASE_URL"]
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")

# Optional second provider. When set, call_llm() automatically retries here if the
# primary provider fails with a rate-limit / quota error, instead of failing the request.
# Set both to run e.g. Gemini as primary and Groq as fallback (or vice versa) so a
# multi-call batch job like the BI-Bench runner can burn through two separate free-tier
# daily quotas instead of stopping when the first one runs out.
LLM_FALLBACK_PROVIDER = os.environ.get("LLM_FALLBACK_PROVIDER", "")
LLM_FALLBACK_MODEL = os.environ.get("LLM_FALLBACK_MODEL", "")

MODEL_DIR = os.environ.get("MODEL_DIR", "models")

SQL_ROW_LIMIT = int(os.environ.get("SQL_ROW_LIMIT", "500"))
SQL_TIMEOUT_MS = int(os.environ.get("SQL_TIMEOUT_MS", "5000"))

RATE_LIMIT_MARKERS = ("429", "rate limit", "quota", "resource_exhausted")


def _call_provider(provider: str, model: str, system: str, prompt: str, max_tokens: int) -> str:
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")
    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
    elif provider == "groq":
        # Groq's API is OpenAI-compatible, so we reuse the OpenAI SDK with a different base_url.
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
    elif provider == "nvidia":
        # NVIDIA NIM is also OpenAI-compatible; same SDK, different base_url + key.
        # Some NVIDIA models (e.g. the nemotron reasoning family) narrate their chain of
        # thought before the actual answer, wrapped in <think>...</think> or similar. Since
        # NEXUS needs the raw answer (SQL, JSON, etc.) not a narration, strip that out here
        # rather than depend on every caller's prompt alone to suppress it.
        from openai import OpenAI
        import re as _re
        client = OpenAI(api_key=os.environ["NVIDIA_API_KEY"], base_url="https://integrate.api.nvidia.com/v1")
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system + " Respond with ONLY the final answer, no reasoning or narration."},
                      {"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content or ""
        text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL | _re.IGNORECASE).strip()
        return text
    elif provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        gm = genai.GenerativeModel(model_name=model, system_instruction=system)
        resp = gm.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens),
        )
        return resp.text
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(marker in msg for marker in RATE_LIMIT_MARKERS)


def call_llm(system: str, prompt: str, max_tokens: int = 1500) -> str:
    """Thin wrapper so agents don't care which provider is configured. Falls back to
    LLM_FALLBACK_PROVIDER automatically on a rate-limit/quota error, if one is configured."""
    try:
        return _call_provider(LLM_PROVIDER, LLM_MODEL, system, prompt, max_tokens)
    except Exception as e:
        if LLM_FALLBACK_PROVIDER and _is_rate_limit_error(e):
            print(f"[call_llm] {LLM_PROVIDER} rate-limited ({e}); falling back to {LLM_FALLBACK_PROVIDER}")
            time.sleep(1)
            return _call_provider(LLM_FALLBACK_PROVIDER, LLM_FALLBACK_MODEL, system, prompt, max_tokens)
        raise
