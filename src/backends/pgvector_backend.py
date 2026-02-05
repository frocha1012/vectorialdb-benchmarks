import time
from typing import List, Dict, Any, Optional, Tuple

import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector


class PgVectorBackend:
    """
    PostgreSQL + pgvector backend.
    Stores embeddings in a table and supports similarity search.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5432,
        dbname: str = "vectordb",
        user: str = "postgres",
        password: str = "postgres",
        table_name: str = "documents",
    ):
        self.conn_params = dict(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
        )
        self.table_name = table_name

    def _connect(self):
        conn = psycopg2.connect(**self.conn_params)
        register_vector(conn)  # enables pgvector adapter
        return conn

    def create_schema(self, dim: int):
        """
        Creates table for embeddings (id, text, embedding).
        Uses VECTOR(dim) type.
        """
        sql = f"""
        DROP TABLE IF EXISTS {self.table_name};
        CREATE TABLE {self.table_name} (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            embedding VECTOR({dim}) NOT NULL
        );
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                conn.commit()

    def clear(self):
        """Deletes all rows in the table."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {self.table_name};")
                conn.commit()

    def upsert(self, rows: List[Dict[str, Any]]):
        """
        rows: [{"id": int, "text": str, "embedding": np.array/list[float]}]
        """
        values = [(r["id"], r["text"], r["embedding"]) for r in rows]

        sql = f"""
        INSERT INTO {self.table_name} (id, text, embedding)
        VALUES %s
        ON CONFLICT (id) DO UPDATE
        SET text = EXCLUDED.text,
            embedding = EXCLUDED.embedding;
        """

        with self._connect() as conn:
            with conn.cursor() as cur:
                execute_values(cur, sql, values)
                conn.commit()

    def search(
        self,
        query_vector,
        top_k: int = 5,
    ) -> List[Tuple[int, str, float]]:
        """
        Returns list of (id, text, distance) ordered by nearest.
        Uses L2 distance with <-> operator.
        """
        sql = f"""
        SELECT id, text, (embedding <-> %s) AS distance
        FROM {self.table_name}
        ORDER BY embedding <-> %s
        LIMIT %s;
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (query_vector, query_vector, top_k))
                return cur.fetchall()

    def benchmark_search(self, query_vector, top_k: int = 10, repeats: int = 100) -> Dict[str, float]:
        """
        Simple latency benchmark: run the same search many times and measure ms.
        Returns p50 and p95 approx + avg.
        """
        times_ms = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = self.search(query_vector, top_k=top_k)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

        times_ms.sort()
        p50 = times_ms[int(0.50 * (len(times_ms) - 1))]
        p95 = times_ms[int(0.95 * (len(times_ms) - 1))]
        avg = sum(times_ms) / len(times_ms)

        return {"avg_ms": avg, "p50_ms": p50, "p95_ms": p95, "repeats": repeats}
