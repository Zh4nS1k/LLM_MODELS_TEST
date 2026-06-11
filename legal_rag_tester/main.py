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
    
    args = parser.parse_args()
    
    # Apply overrides
    if args.input:
        settings.input_excel = args.input
    if args.output:
        settings.output_excel = args.output
        
    verbose = args.verbose or settings.verbose
    update_logger_verbose(verbose)
    
    pipeline = TestPipeline(override_models=args.models)
    results = pipeline.run(limit=args.limit, dry_run=args.dry_run, skip_low_limit=args.skip_low_limit)
    
if __name__ == "__main__":
    main()