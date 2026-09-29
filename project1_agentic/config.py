import os
from dotenv import load_dotenv

load_dotenv()

READONLY_DATABASE_URL = os.environ.get("READONLY_DATABASE_URL") or os.environ["DATABASE_URL"]
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")
MODEL_DIR = os.environ.get("MODEL_DIR", "models")

SQL_ROW_LIMIT = int(os.environ.get("SQL_ROW_LIMIT", "500"))
SQL_TIMEOUT_MS = int(os.environ.get("SQL_TIMEOUT_MS", "5000"))


def call_llm(system: str, prompt: str, max_tokens: int = 1500) -> str:
    """Thin wrapper so agents don't care which provider is configured."""
    if LLM_PROVIDER == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = client.messages.create(
            model=LLM_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")
    elif LLM_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
    elif LLM_PROVIDER == "groq":
        # Groq's API is OpenAI-compatible, so we reuse the OpenAI SDK with a different base_url.
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
    elif LLM_PROVIDER == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel(model_name=LLM_MODEL, system_instruction=system)
        resp = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens),
        )
        return resp.text
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")
