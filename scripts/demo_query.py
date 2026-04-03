from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline import CitationAwareRAG


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Question to ask the system.")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve.")
    args = parser.parse_args()

    rag = CitationAwareRAG()
    result = rag.answer(args.query, top_k=args.top_k)
    print(result.answer)
    print("\nDetailed citations:")
    for citation in result.citations:
        print(citation)


if __name__ == "__main__":
    main()
