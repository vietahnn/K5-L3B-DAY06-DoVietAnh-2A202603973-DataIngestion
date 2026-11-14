"""Persistent PostgreSQL/pgvector storage with parameterized metadata filters."""
import json
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row

from .models import Chunk


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.database_url = database_url
        self.dimensions = dimensions

    def create_schema(self) -> None:
        with psycopg.connect(self.database_url) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute(f"""
                CREATE TABLE IF NOT EXISTS day6_chunks (
                    strategy TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata JSONB NOT NULL,
                    embedding vector({self.dimensions}) NOT NULL,
                    PRIMARY KEY (strategy, chunk_id)
                )
            """)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS day6_chunks_strategy_idx ON day6_chunks(strategy)"
            )
            connection.commit()

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not strategy:
            raise ValueError("strategy must not be empty")
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have equal lengths")
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise ValueError("embedding dimension does not match store schema")
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM day6_chunks WHERE strategy = %s", (strategy,))
                for chunk, vector in zip(chunks, vectors):
                    cursor.execute(
                        """
                        INSERT INTO day6_chunks
                            (strategy, chunk_id, document_id, source, page, text, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                        """,
                        (strategy, chunk.chunk_id, chunk.document_id, chunk.source, chunk.page,
                         chunk.text, json.dumps(chunk.metadata), _vector_literal(vector)),
                    )
            connection.commit()

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if len(query_vector) != self.dimensions:
            raise ValueError("query dimension does not match store schema")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        allowed_filters = {"section", "year", "document_type", "language", "audience"}
        filters = filters or {}
        unknown = set(filters) - allowed_filters
        if unknown:
            raise ValueError(f"unsupported metadata filters: {sorted(unknown)}")
        clauses = ["strategy = %s"]
        params: list[Any] = [strategy]
        for key, value in filters.items():
            clauses.append("metadata ->> %s = %s")
            params.extend([key, str(value)])
        query = f"""
            SELECT chunk_id, document_id, source, page, text, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM day6_chunks
            WHERE {' AND '.join(clauses)}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        # The query vector occurs in SELECT and ORDER BY; duplicate it in parameters.
        vector = _vector_literal(query_vector)
        params = [vector, strategy]
        for key, value in filters.items():
            params.extend([key, str(value)])
        params.extend([vector, top_k])
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.10g}" for value in vector) + "]"
