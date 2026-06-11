"""Answer quality scoring."""
from config import settings
from llm_client import LLMClient

class AnswerScorer:
    def __init__(self):
        self.llm_client = LLMClient()
        # Default to llama-3.3 for high quality judgment
        self.model = "llama-3.3-70b-versatile"
        
        self.system_prompt = """You are a legal answer quality evaluator. Score the given answer
on a scale of 0–10 based on these criteria:

10 — Perfect: directly answers the question, cites correct legal
     articles from context, structured clearly, no hallucination
7-9 — Good: answers the question, correct law cited, minor gaps
4-6 — Partial: some correct info but incomplete or missing key articles
1-3 — Poor: vague, misses the point, or cites wrong laws
0   — Refused or completely wrong

Respond with ONLY: <score>|<one sentence reason>
Example: 8|Correctly cites Article 35 and 42-4 but misses Article 634."""

    def score(self, question: str, chunks_context: str, answer: str) -> tuple[int, str]:
        """Returns (score 0-10, one-line reason)."""
        # If the model explicitly gave no answer, auto-score 0
        if answer.strip() == "Контекстте жауап жоқ.":
            return 0, "Model determined context did not contain answer."
            
        user_prompt = f"Question: {question}\nContext chunks: {chunks_context}\nAnswer to evaluate: {answer}\nScore:"
        
        try:
            result = self.llm_client.call(self.model, self.system_prompt, user_prompt)
            if result.error:
                return 0, f"Error scoring: {result.error}"
                
            out = result.answer.strip()
            if "|" in out:
                score_str, reason = out.split("|", 1)
                try:
                    score_val = int(score_str.strip())
                    return score_val, reason.strip()
                except ValueError:
                    pass
            return 0, f"Invalid scorer output: {out}"
        except Exception as e:
            return 0, f"Exception during scoring: {str(e)}"
