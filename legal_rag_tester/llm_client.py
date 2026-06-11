"""Unified LLM Client module."""
import time
import re
from typing import Dict, Any
from logger import pipeline_logger
from openai import OpenAI, RateLimitError, APITimeoutError, APIError
from google import genai
from google.genai import types
import os
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

    def _strip_think_tags(self, text: str) -> str:
        """Remove <think>...</think> blocks from any model response."""
        import re
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return cleaned.strip()

    def _call_llm(self, model: str, system: str, user: str, q_id: str) -> Dict[str, Any]:
        """Calls the LLM API with custom rate limit handling."""
        delays = [2, 4, 8]
        import os
        from config import settings
        
        for attempt, delay in enumerate(delays + [0]):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    temperature=0.0,
                    timeout=float(os.getenv("QWEN_TIMEOUT", "45")) if "qwen" in model.lower() else settings.request_timeout,
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
                    pipeline_logger.log_error(q_id, model, "Rate limited (429) after retries.")
                    raise
            except Exception as e:
                pipeline_logger.log_error(q_id, model, f"API Error: {str(e)}")
                raise

    def _call_google(self, model: str, system: str, user: str) -> LLMResult:
        """Call Google AI Studio via new google-genai SDK."""
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        # Suppress thinking for Flash models to cut latency
        thinking_cfg = None
        if "flash" in model.lower():
            try:
                thinking_cfg = types.ThinkingConfig(thinking_budget=0)
            except Exception:
                pass  # SDK version may not support it — ignore

        response = client.models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.1,
                max_output_tokens=4096,
                thinking_config=thinking_cfg,
            ),
        )

        # Detect truncation
        finish_reason = None
        if response.candidates:
            finish_reason = response.candidates[0].finish_reason
        truncated = str(finish_reason) in ("FinishReason.MAX_TOKENS", "MAX_TOKENS", "2")
        if truncated:
            pipeline_logger.log_warning(
                f"[{model}] Response truncated at max_output_tokens. "
                f"Consider increasing limit or reducing context."
            )

        # Extract text safely
        try:
            answer_text = response.text
        except (ValueError, AttributeError):
            return LLMResult(
                model=model, answer="Контекстте жауап жоқ.",
                error="safety_blocked", latency_ms=0, chunks_used=0,
            )

        answer_text = self._strip_think_tags(answer_text)

        # Token counts
        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            prompt_tokens = getattr(
                response.usage_metadata, 'prompt_token_count', 0) or 0
            completion_tokens = getattr(
                response.usage_metadata, 'candidates_token_count', 0) or 0

        return LLMResult(
            model=model,
            answer=answer_text,
            answer_raw=answer_text,
            error="truncated" if truncated else "",
            latency_ms=0,
            chunks_used=0,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def call(self, model: str, system: str, user: str, q_id: str = "unknown") -> LLMResult:
        """Routes the call to the appropriate API."""
        start = time.perf_counter()
        try:
            if model.startswith("models/"):
                result = self._call_google(model, system, user)
            else:
                raw_res = self._call_llm(model, system, user, q_id)
                answer_raw = raw_res["answer"]
                answer = self._strip_think_tags(answer_raw)
                
                if '<think>' in answer:
                    answer = answer[:answer.index('<think>')].strip()
                    
                result = LLMResult(
                    model=model,
                    answer=answer,
                    answer_raw=answer_raw,
                    error=raw_res["error"],
                    latency_ms=raw_res["latency_ms"],
                    prompt_tokens=raw_res["prompt_tokens"],
                    completion_tokens=raw_res["completion_tokens"]
                )
            
            result.latency_ms = (time.perf_counter() - start) * 1000
            return result
        except RateLimitError as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResult(
                model=model,
                answer="",
                error=f"rate_limited: {str(e)[:120]}",
                latency_ms=elapsed,
            )
        except APITimeoutError as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResult(
                model=model,
                answer="",
                error=f"timeout: {str(e)[:120]}",
                latency_ms=elapsed,
            )
        except APIError as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResult(
                model=model,
                answer="",
                error=f"api_error_{getattr(e, 'status_code', 'unknown')}: {str(e)[:120]}",
                latency_ms=elapsed,
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return LLMResult(
                model=model,
                answer="",
                error=f"unexpected: {type(e).__name__}: {str(e)[:120]}",
                latency_ms=elapsed,
            )