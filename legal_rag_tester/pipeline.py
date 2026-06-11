"""Pipeline orchestration for the Legal RAG Tester."""
import time
from tqdm import tqdm
from logger import pipeline_logger
from timer import StepTimer
from token_counter import TokenCounter
import excel_io
from embedder import Embedder
from retriever import PineconeRetriever
from prompt_builder import PromptBuilder
from llm_client import LLMClient
from config import settings
from models.schemas import TestRow

class TestPipeline:
    """Orchestrates the entire evaluation process."""
    
    def __init__(self, override_models: str = None):
        """Initializes the pipeline components."""
        self.embedder = Embedder()
        self.retriever = PineconeRetriever(settings)
        self.prompt_builder = PromptBuilder()
        self.llm_client = LLMClient()
        self.token_counter = TokenCounter()
        
        self.models_to_test = settings.llm_models
        if override_models:
            self.models_to_test = [m.strip() for m in override_models.split(",") if m.strip()]

    def run(self, limit: int = None, dry_run: bool = False, skip_low_limit: bool = False):
        """Runs the complete RAG test pipeline."""
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

        pipeline_logger.log_pipeline_start(len(questions), self.models_to_test)
        
        results = []
        
        for q_index, question in enumerate(tqdm(questions, desc="Processing questions"), 1):
            pipeline_logger.log_question(question.id, question.text, q_index, len(questions))
            
            with StepTimer("embed") as t_embed:
                embedding = self.embedder.embed(question.text)
            pipeline_logger.log_timing("embed", t_embed.elapsed_ms)
            
            with StepTimer("retrieve") as t_retrieve:
                chunks = self.retriever.query(embedding)
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
            
            for model_name in self.models_to_test:
                time.sleep(settings.groq_rpm_delay)
                
                with StepTimer("llm_call") as t_llm:
                    result = self.llm_client.call(model_name, prompt["system"], prompt["user"])
                
                total_llm_ms += t_llm.elapsed_ms
                
                result.chunks_used = len(chunks)
                result.retrieved_scores = [c.score for c in chunks]
                result.avg_score = sum(c.score for c in chunks) / len(chunks) if chunks else 0.0
                result.embed_ms = t_embed.elapsed_ms
                result.retrieve_ms = t_retrieve.elapsed_ms
                result.llm_ms = t_llm.elapsed_ms
                result.total_ms = t_embed.elapsed_ms + t_retrieve.elapsed_ms + t_llm.elapsed_ms
                
                # Sanitize the final answer
                def sanitize_answer(ans: str) -> str:
                    NO_ANSWER = "Контекстте жауап жоқ."
                    if ans.strip().startswith(NO_ANSWER):
                        return NO_ANSWER
                    ans = ans.replace(NO_ANSWER, "").strip()
                    return ans if ans else NO_ANSWER
                    
                if result.answer:
                    result.answer = sanitize_answer(result.answer)
                    
                # Calculate tokens consistently
                prompt_toks = prompt_token_count
                comp_toks = result.completion_tokens or self.token_counter.count(result.answer_raw)
                
                if t_llm.elapsed_ms > 10000:
                    tok_s = comp_toks / (result.llm_ms / 1000.0) if result.llm_ms > 0 else 0.0
                    pipeline_logger.log_warning(
                        f"⚠️  [{model_name}] Q#{question.id} slow response: {t_llm.elapsed_ms:.0f}ms "
                        f"({comp_toks} tokens, {tok_s:.1f} tok/s)"
                    )
                
                total_toks = prompt_toks + comp_toks
                
                result.prompt_tokens = prompt_toks
                result.completion_tokens = comp_toks
                
                result.tokens_per_sec = comp_toks / (result.llm_ms / 1000.0) if result.llm_ms > 0 else 0.0
                
                self.token_counter.record(model_name, prompt_toks, comp_toks)
                
                results.append(TestRow(question=question, result=result))
                
                if result.error:
                    pipeline_logger.log_error(question.id, model_name, result.error)
                else:
                    pipeline_logger.log_request(question.id, model_name, prompt["system"], prompt["user"], prompt_toks)
                    pipeline_logger.log_response(question.id, model_name, result.answer, result.llm_ms)
                    pipeline_logger.log_tokens(question.id, model_name, prompt_toks, comp_toks, total_toks, result.llm_ms)
                    
            if not dry_run:
                pipeline_logger.log_question_done(question.id, t_embed.elapsed_ms + t_retrieve.elapsed_ms + total_llm_ms, len(self.models_to_test))
                    
        if not dry_run:
            excel_io.write_results(results)
            pipeline_logger.log_simple_success(f"Done. {len(results)} rows written.")
            pipeline_logger.log_pipeline_summary(results)
            
        return results