# evals/judge.py
"""
Groq-backed judge for DeepEval.

The original suite judged with gpt-4o-mini. This project has no OpenAI key, so
every DeepEval metric is pointed at Groq through DeepEvalBaseLLM, using the
separate JUDGE_GROQ key (falls back to GROQ_API_KEY) so eval runs never eat the
production key's rate limit.

Env knobs:
  JUDGE_MODEL          default llama-3.3-70b-versatile (use llama-3.1-8b-instant
                       if you hit the 6k-TPM on-demand ceiling; expect noisier scores)
  JUDGE_MIN_INTERVAL_S minimum seconds between judge calls (default 2)

Caveat: the generator is also a Llama model, so the judge shares blind spots with
the system under test. Treat absolute scores with suspicion; the regression
DELTAS (baseline vs candidate, same judge) are what the suite is built to use.
"""
import json
import os
import re
import time

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI, RateLimitError, APIStatusError

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile")
MIN_INTERVAL_S = float(os.getenv("JUDGE_MIN_INTERVAL_S", "2"))
MAX_RETRIES = 6


def _extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in judge output: {text[:200]!r}")
    return json.loads(text[start : end + 1])


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model: str = DEFAULT_JUDGE_MODEL):
        self.model_name = model
        key = os.getenv("JUDGE_GROQ") or os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("Set JUDGE_GROQ (or GROQ_API_KEY) in .env to run LLM-judged evals.")
        self.client = OpenAI(api_key=key, base_url=GROQ_BASE_URL)
        self._last_call = 0.0

    def load_model(self):
        return self.client

    def get_model_name(self):
        return self.model_name

    def _chat(self, prompt: str, json_mode: bool) -> str:
        for attempt in range(MAX_RETRIES):
            wait = MIN_INTERVAL_S - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    **kwargs,
                )
                self._last_call = time.time()
                return resp.choices[0].message.content or ""
            except (RateLimitError, APIStatusError) as e:
                status = getattr(e, "status_code", None)
                if status not in (429, 500, 502, 503) or attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(min(2 ** attempt * 5, 60))   # 5s, 10s, 20s, 40s, 60s
        raise RuntimeError("unreachable")

    def generate(self, prompt: str, schema=None):
        if schema is None:
            return self._chat(prompt, json_mode=False)
        full = (
            f"{prompt}\n\nRespond with ONLY a valid JSON object matching this JSON schema "
            f"(no markdown, no commentary):\n{json.dumps(schema.model_json_schema())}"
        )
        last_err = None
        for _ in range(3):                       # re-ask if the model emits malformed JSON
            try:
                return schema.model_validate(_extract_json(self._chat(full, json_mode=True)))
            except Exception as e:
                last_err = e
        raise last_err

    async def a_generate(self, prompt: str, schema=None):
        import asyncio
        return await asyncio.to_thread(self.generate, prompt, schema)


_judge = None


def get_judge() -> GroqJudge:
    """Singleton so rate-limit pacing is shared across every metric."""
    global _judge
    if _judge is None:
        _judge = GroqJudge()
    return _judge
