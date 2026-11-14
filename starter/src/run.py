"""Run the three controlled ingestion and retrieval strategies."""
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf
from .store import PgVectorStore


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    pdf_path = Path(os.getenv("PDF_PATH", root.parent / "Poverty, Prosperity and Planet Report 2026 - Overview.pdf"))
    benchmark_path = Path(os.getenv("BENCHMARK_PATH", root / "data" / "benchmark.json"))
    if not benchmark_path.is_file():
        raise FileNotFoundError(
            f"Missing benchmark: {benchmark_path}. Create 20 verified questions with 15 dev/5 test."
        )
    questions = _load_benchmark(benchmark_path)
    pages = load_pdf(pdf_path, "poverty-prosperity-planet-2026-overview", str(pdf_path))
    content_pages = [page for page in pages if page.text]
    if len(content_pages) < 30:
        raise ValueError(f"Corpus has {len(content_pages)} non-empty pages; at least 30 are required")

    model_name = os.getenv("EMBEDDING_MODEL") or (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    embedder = MyEmbedder(model_name)
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required; copy starter/.env.example to starter/.env")
    store = PgVectorStore(database_url, embedder.dimensions)
    store.create_schema()
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "corpus_manifest.json").write_text(
        json.dumps({
            "source_id": pages[0].document_id,
            "title": "Poverty, Prosperity and Planet Report 2026 - Overview",
            "source_path": str(pdf_path),
            "license_or_permission": os.getenv("CORPUS_LICENSE", "REPLACE_WITH_VERIFIED_PERMISSION"),
            "accessed_at": os.getenv("CORPUS_ACCESSED_AT", "REPLACE_WITH_ACCESS_DATE"),
            "physical_page_count": len(pages),
            "non_empty_content_pages": len(content_pages),
            "pages": [
                {"page": page.page, "characters": len(page.text),
                 "section": page.metadata.get("section"),
                 "extraction_status": page.metadata.get("extraction_status")}
                for page in pages
            ],
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    all_traces: list[dict] = []
    strategy_stats: dict[str, dict[str, int]] = {}
    run_started = time.perf_counter()
    for strategy, chunk_builder in (
        ("A", lambda page: fixed_size(page, size=500, overlap=50)),
        ("B", lambda page: by_paragraph(page, max_size=800)),
        ("C", lambda page: by_paragraph(page, max_size=800)),
    ):
        chunks = [chunk for page in pages for chunk in chunk_builder(page)]
        strategy_stats[strategy] = {
            "chunk_count": len(chunks),
            "total_characters": sum(len(chunk.text) for chunk in chunks),
            "min_chunk_characters": min((len(chunk.text) for chunk in chunks), default=0),
            "max_chunk_characters": max((len(chunk.text) for chunk in chunks), default=0),
        }
        vectors = embedder.embed_documents([chunk.text for chunk in chunks])
        store.upsert(strategy, chunks, vectors)
        for question in questions:
            query_vector = embedder.embed_query(question["question"])
            filters = question.get("filter", {}) if strategy == "C" else {}
            trace = evaluate_query(question, store.search(strategy, query_vector, 3, filters))
            trace["strategy"] = strategy
            trace["filter"] = filters
            all_traces.append(trace)
    (results_dir / "retrieval_traces.json").write_text(
        json.dumps(all_traces, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (results_dir / "retrieval_summary.json").write_text(
        json.dumps(summarize(all_traces), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (results_dir / "strategy_stats.json").write_text(
        json.dumps(strategy_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    elapsed = time.perf_counter() - run_started
    print(json.dumps({"model": embedder.model_name, "dimensions": embedder.dimensions,
                      "pages": len(content_pages), "queries": len(questions),
                      "elapsed_seconds": round(elapsed, 3)}, indent=2))


def _load_benchmark(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) < 20:
        raise ValueError("benchmark must be a JSON list containing at least 20 questions")
    splits = {question.get("split") for question in data}
    if sum(question.get("split") == "dev" for question in data) < 15:
        raise ValueError("benchmark must contain at least 15 dev questions")
    if sum(question.get("split") == "test" for question in data) < 5:
        raise ValueError("benchmark must contain at least 5 test questions")
    required = {"id", "split", "question", "gold_source_id", "gold_page", "evidence_quote"}
    for question in data:
        missing = required - set(question)
        if missing or question["split"] not in {"dev", "test"}:
            raise ValueError(f"invalid benchmark question {question.get('id')}: missing {missing}")
    if sum(bool(question.get("filter")) for question in data) < 3:
        raise ValueError("benchmark must contain at least 3 metadata-filter questions")
    if not any(question.get("split") == "test" and question.get("filter")
               for question in data):
        raise ValueError("benchmark needs at least one test question with a metadata filter")
    return data


if __name__ == "__main__":
    main()
