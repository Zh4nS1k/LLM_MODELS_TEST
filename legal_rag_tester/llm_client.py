"""Unified LLM Client module."""
import time
import re
from typing import Dict, Any
from logger import pipeline_logger
from openai import OpenAI, RateLimitError
from models.schemas import LLMResult
from config import settings

class LLMClient:
    """Unified client for calling Groq models via OpenAI-compatible API."""

    def __init__(self):
        """Initializes the LLMClient."""
        self.client = OpenAI(
            api_key=settings.groq_api_key, 
            base_url=settings.groq_base_url, 
            timeout=settings.request_timeout
        )

    def _call_groq(self, model: str, system: str, user: str) -> Dict[str, Any]:
        """Calls the Groq API with custom rate limit handling."""
        delays = [2, 4, 8]
        
        for attempt, delay in enumerate(delays + [0]):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    temperature=0.0
                )
                latency_ms = (time.time() - start_time) * 1000
                
                prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                completion_tokens = response.usage.completion_tokens if response.usage else 0
                
                return {
                    "answer": response.choices[0].message.content or "",
                    "latency_ms": latency_ms,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "error": ""
                }
            except RateLimitError as e:
                if attempt < len(delays):
                    pipeline_logger.log_warning(f"[{model}] Rate limited (429). Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    pipeline_logger.log_error("Unknown", model, "Rate limited (429) after retries.")
                    return {"answer": "", "latency_ms": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "error": "rate_limited"}
            except Exception as e:
                pipeline_logger.log_error("Unknown", model, f"API Error: {str(e)}")
                return {"answer": "", "latency_ms": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "error": str(e)}

    def call(self, model: str, system: str, user: str) -> LLMResult:
        """Routes the call to the Groq API."""
        result = self._call_groq(model, system, user)
        
        answer_raw = result["answer"]
        answer = answer_raw
        
        # DeepSeek-R1 special handling
        if model == "deepseek-r1-distill-70b":
            # Strip everything inside <think>...</think>
            answer = re.sub(r'<think>.*?</think>', '', answer_raw, flags=re.DOTALL).strip()
            
        return LLMResult(
            model=model,
            answer=answer,
            answer_raw=answer_raw,
            error=result["error"],
            latency_ms=result["latency_ms"],
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"]
        )