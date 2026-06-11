"""Entry point for the Legal RAG Tester CLI."""
import argparse
import sys
from logger import update_logger_verbose
from pipeline import TestPipeline
from config import settings

def main():
    """Parses arguments and runs the pipeline."""
    parser = argparse.ArgumentParser(description="Legal RAG Tester Pipeline")
    parser.add_argument("--input", type=str, help="Override INPUT_EXCEL")
    parser.add_argument("--output", type=str, help="Override OUTPUT_EXCEL")
    parser.add_argument("--models", type=str, help="Override LLM_MODELS (comma-separated)")
    parser.add_argument("--limit", type=int, help="Process only first N questions")
    parser.add_argument("--dry-run", action="store_true", help="Embed and retrieve only, skip LLM calls")
    parser.add_argument("--skip-low-limit", action="store_true", help="Skip low-limit models like llama-4-maverick")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--diagnose", action="store_true", help="Run Pinecone diagnostic tool")
    
    args = parser.parse_args()
    
    # Apply overrides
    if args.input:
        settings.input_excel = args.input
    if args.output:
        settings.output_excel = args.output
        
    verbose = args.verbose or settings.verbose
    if args.diagnose:
        import excel_io
        from embedder import Embedder
        from retriever import PineconeRetriever
        print("Running diagnostic...")
        questions = excel_io.read_questions()
        if not questions:
            print("No questions found!")
            return
        q = questions[0]
        embedder = Embedder()
        retriever = PineconeRetriever(settings)
        embedding = embedder.embed(q.text)
        print(f"🔍 Query embedding dim: {len(embedding)}")
        stats = retriever.index.describe_index_stats()
        print(f"📊 Index stats: dimension={stats.dimension}, total_vectors={stats.total_vector_count}, namespaces={list(stats.namespaces.keys())}")
        response = retriever.index.query(
            namespace=settings.pinecone_namespace,
            vector=embedding,
            top_k=5,
            include_values=False,
            include_metadata=True
        )
        print("📄 Top 5 raw matches (no threshold filter):")
        for i, match in enumerate(response.get("matches", []), 1):
            meta_keys = list(match.metadata.keys()) if match.metadata else []
            text_preview = match.metadata.get(settings.chunk_text_field, "")[:80] if match.metadata else ""
            print(f"   [{i}] id={match.id}  score={match.score:.3f}  metadata_keys={meta_keys}")
            print(f"       text preview: \"{text_preview}...\"")
        return
        
    pipeline = TestPipeline(override_models=args.models)
    results = pipeline.run(limit=args.limit, dry_run=args.dry_run, skip_low_limit=args.skip_low_limit)
    
if __name__ == "__main__":
    main()