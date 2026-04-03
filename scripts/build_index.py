from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pipeline import CitationAwareRAG


def main() -> None:
    rag = CitationAwareRAG()
    stats = rag.build_index()
    print(f"Indexed {stats['documents']} document pages into {stats['chunks']} chunks.")


if __name__ == "__main__":
    main()
