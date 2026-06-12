"""Pipeline orchestration for the Legal RAG Tester."""
import os
import sys
import signal
import time
from datetime import datetime
from tqdm import tqdm
from logger import pipeline_logger
from timer import StepTimer
from token_counter import TokenCounter
import excel_io
from embedder import Embedder
from retriever import PineconeRetriever
from prompt_builder import PromptBuilder
from llm_client import LLMClient
from models.schemas import LLMResult, TestRow
from config import settings
from query_rewriter import QueryRewriter
from answer_scorer import AnswerScorer
from checkpoint import CheckpointManager


def _get_provider(model: str) -> str:
    """Classify model by which API key / rate-limit pool it uses."""
    if model.startswith("models/"):
        return "google"
    if model.startswith("gpt-5") or model.startswith("gpt-4"):
        return "openai"
    return "groq"  # everything else (llama, qwen, deepseek, openai/gpt-oss-*)


def _inter_request_delay(prev_model: str, next_model: str) -> None:
    """Sleep using the SLOWER of the two adjacent providers' configured delays."""
    delays = {
        "groq":   float(os.getenv("DELAY_GROQ",   "3.0")),
        "google": float(os.getenv("DELAY_GOOGLE", "6.0")),
        "openai": float(os.getenv("DELAY_OPENAI", "5.0")),
    }
    prev_delay = delays.get(_get_provider(prev_model), 3.0)
    next_delay = delays.get(_get_provider(next_model), 3.0)
    time.sleep(max(prev_delay, next_delay))


class TestPipeline:
    """Orchestrates the entire evaluation process."""

    def __init__(self, override_models: str = None):
        """Initializes the pipeline components."""
        self.embedder = Embedder()
        self.retriever = PineconeRetriever(settings)
        self.prompt_builder = PromptBuilder()
        self.llm_client = LLMClient()
        self.token_counter = TokenCounter()
        self.query_rewriter = QueryRewriter()
        self.answer_scorer = AnswerScorer()

        self.models_to_test = settings.llm_models
        if override_models:
            self.models_to_test = [m.strip() for m in override_models.split(",") if m.strip()]

    def run(self, limit: int = None, dry_run: bool = False,
            skip_low_limit: bool = False, resume: bool = False):
        """Runs the complete RAG test pipeline with crash-safe checkpointing."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint = CheckpointManager(
            output_dir=settings.output_dir,
            session_id=session_id
        )

        # ── Graceful signal handlers ────────────────────────────────────
        def _signal_handler(sig, frame):
            pipeline_logger.log_warning(
                f"⌨️  Interrupted! Saving {len(checkpoint.rows)} rows..."
            )
            checkpoint.finalize()
            sys.exit(0)

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        try:
            questions = excel_io.read_questions()
            if limit:
                questions = questions[:limit]

            if not questions:
                pipeline_logger.log_warning("No questions found. Exiting.")
                return []

            maverick = "meta-llama/llama-4-scout-17b-16e-instruct"
            if maverick in self.models_to_test and (len(questions) > 400 or skip_low_limit):
                pipeline_logger.log_simple_info(f"Skipping {maverick} due to low limit constraints.")
                self.models_to_test.remove(maverick)

            # ── Resume: skip already-completed questions ────────────────
            if resume:
                ckpt_file = CheckpointManager.find_latest_checkpoint(settings.output_dir)
                if ckpt_file:
                    import pandas as pd
                    already_done = set(pd.read_excel(ckpt_file)["id"].astype(str).unique())
                    before = len(questions)
                    questions = [q for q in questions if str(q.id) not in already_done]
                    pipeline_logger.log_simple_info(
                        f"▶️  Resuming from {ckpt_file.name}: "
                        f"{before - len(questions)} done, {len(questions)} remaining"
                    )
                    # Load existing rows into the checkpoint so the final file is complete
                    existing_rows = excel_io.read_results_from_checkpoint(ckpt_file)
                    if existing_rows:
                        checkpoint.rows.extend(existing_rows)
                else:
                    pipeline_logger.log_warning("--resume requested but no checkpoint found. Starting fresh.")

            pipeline_logger.log_pipeline_start(len(questions), self.models_to_test)

            for q_index, question in enumerate(tqdm(questions, desc="Processing questions"), 1):
                pipeline_logger.log_question(question.id, question.text, q_index, len(questions))

                rewritten_query = self.query_rewriter.rewrite(question.text)
                pipeline_logger.log_simple_info(f"🔍 Original: {question.text[:100]}...")
                pipeline_logger.log_simple_info(f"🔍 Rewritten: {rewritten_query}")

                with StepTimer("embed") as t_embed:
                    embedding = self.embedder.embed(rewritten_query)
                pipeline_logger.log_timing("embed", t_embed.elapsed_ms)

                with StepTimer("retrieve") as t_retrieve:
                    chunks = self.retriever.query(embedding, query_text=question.text)
                pipeline_logger.log_retrieval(question.id, chunks, t_retrieve.elapsed_ms)

                if len(chunks) < 2:
                    pipeline_logger.log_warning(
                        f"Q#{question.id} — Retrieved {len(chunks)} chunks. "
                        f"Question: '{question.text[:80]}...' "
                        f"Embed dim: {len(embedding)} | Retrieve time: {t_retrieve.elapsed_ms:.0f}ms"
                    )

                if dry_run:
                    continue

                prompt = self.prompt_builder.build(question.text, chunks)

                prompt_token_count = self.token_counter.count_prompt(prompt["system"], prompt["user"])
                pipeline_logger.log_simple_info(
                    f"🧮 Q#{question.id} prompt built: {prompt_token_count} tokens "
                    f"({len(chunks)} chunks, system={self.token_counter.count(prompt['system'])} "
                    f"user={self.token_counter.count(prompt['user'])})"
                )

                total_llm_ms = 0.0
                model_results = []

                prev_model = None

                for i, model_name in enumerate(self.models_to_test):
                    # ── Inter-request delay ─────────────────────────────
                    if prev_model:
                        _inter_request_delay(prev_model, model_name)
                    prev_model = model_name

                    with StepTimer("llm_call") as t_llm:
                        try:
                            result = self.llm_client.call(
                                model_name, prompt["system"], prompt["user"], q_id=question.id
                            )
                        except KeyboardInterrupt:
                            raise
                        except Exception as e:
                            pipeline_logger.log_error(question.id, model_name, f"unhandled: {e}")
                            result = LLMResult(model=model_name, answer="", error=str(e),
                                               latency_ms=0, chunks_used=0)

                    total_llm_ms += t_llm.elapsed_ms

                    result.chunks_used = len(chunks)
                    result.retrieved_scores = [c.score for c in chunks]
                    result.avg_score = sum(c.score for c in chunks) / len(chunks) if chunks else 0.0
                    result.embed_ms = t_embed.elapsed_ms
                    result.retrieve_ms = t_retrieve.elapsed_ms
                    result.llm_ms = t_llm.elapsed_ms
                    result.total_ms = t_embed.elapsed_ms + t_retrieve.elapsed_ms + t_llm.elapsed_ms

                    # Sanitize
                    def sanitize_answer(ans: str) -> str:
                        NO_ANSWER = "Контекстте жауап жоқ."
                        if ans.strip().startswith(NO_ANSWER):
                            return NO_ANSWER
                        ans = ans.replace(NO_ANSWER, "").strip()
                        return ans if ans else NO_ANSWER

                    if result.answer:
                        result.answer = sanitize_answer(result.answer)

                    prompt_toks = prompt_token_count
                    comp_toks = result.completion_tokens or self.token_counter.count(result.answer_raw)

                    if t_llm.elapsed_ms > 10000:
                        tok_s = comp_toks / (result.llm_ms / 1000.0) if result.llm_ms > 0 else 0.0
                        pipeline_logger.log_warning(
                            f"⚠️  [{model_name}] Q#{question.id} slow: {t_llm.elapsed_ms:.0f}ms "
                            f"({comp_toks} tokens, {tok_s:.1f} tok/s)"
                        )

                    total_toks = prompt_toks + comp_toks
                    result.prompt_tokens = prompt_toks
                    result.completion_tokens = comp_toks

                    # Auto-scoring + scorer RPM protection
                    score_val, reason = self.answer_scorer.score(
                        question.text, prompt["user"], result.answer, question.id
                    )
                    result.quality_score = score_val
                    result.quality_reason = reason
                    time.sleep(0.5)  # scorer shares Groq key — count against RPM

                    result.tokens_per_sec = comp_toks / (result.llm_ms / 1000.0) if result.llm_ms > 0 else 0.0
                    self.token_counter.record(model_name, prompt_toks, comp_toks)

                    model_results.append(TestRow(question=question, result=result))

                    # ✅ Print immediately
                    pipeline_logger.log_model_result(i + 1, len(self.models_to_test), model_name, result)

                    # Verbose logs
                    if not result.error:
                        pipeline_logger.log_request(question.id, model_name, prompt["system"], prompt["user"], prompt_toks)
                        pipeline_logger.log_response(question.id, model_name, result.answer, result.llm_ms)
                        pipeline_logger.log_tokens(question.id, model_name, prompt_toks, comp_toks, total_toks, result.llm_ms)

                # ── Rank models for this question ───────────────────────
                model_results_sorted = sorted(model_results, key=lambda r: r.result.quality_score, reverse=True)
                for rank, row in enumerate(model_results_sorted, 1):
                    row.result.quality_rank = rank

                pipeline_logger.log_question_best(
                    question.id, [r.result for r in model_results],
                    t_embed.elapsed_ms + t_retrieve.elapsed_ms + total_llm_ms
                )

                # ── 💾 Save after EVERY question ────────────────────────
                checkpoint.add_question_results(model_results)

            # ── Clean completion ────────────────────────────────────────
            checkpoint.finalize()
            pipeline_logger.log_pipeline_summary(checkpoint.rows)
            return checkpoint.rows

        except Exception as e:
            pipeline_logger.log_error("pipeline", "pipeline", str(e))
            pipeline_logger.log_warning("💾 Saving partial results before exit...")
            checkpoint.finalize()
            raise