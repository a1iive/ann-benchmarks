import subprocess
import sys

import pgvector.psycopg
import psycopg

from datetime import datetime
from ..base.module import BaseANN


class PGVector(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = metric
        self._query_probes = None
        self._cur = None

        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def fit(self, X):
        subprocess.run("service postgresql start", shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        conn = psycopg.connect(user="ann", password="ann", dbname="ann", autocommit=True)
        pgvector.psycopg.register_vector(conn)
        cur = conn.cursor()
        cur.execute("set statement_timeout to 300000") # 300s
        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute("CREATE TABLE items (id int, embedding vector(%d))" % X.shape[1])
        cur.execute("ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN")
        print("copying data...")
        with cur.copy("COPY items (id, embedding) FROM STDIN") as copy:
            for i, embedding in enumerate(X):
                copy.write_row((i, embedding))

        print("creating index...")
        index_start_time = datetime.now()
        print("{}: start create index!".format(index_start_time))
        if self._metric == "angular":
            cur.execute(
                "CREATE INDEX ON items USING ivfflat (embedding vector_cosine_ops) WITH (lists = 245)"
            )
        elif self._metric == "euclidean":
            cur.execute("CREATE INDEX ON items USING ivfflat (embedding vector_l2_ops) WITH (lists = 1000)")
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        print("{}: finish create index! build index in {}".format(datetime.now(), datetime.now()-index_start_time))
        print("done!")
        self._cur = cur

    def set_query_arguments(self, n_probes):
        self._query_probes = n_probes
        self._cur.execute("SET ivfflat.probes = %d" % n_probes)

    def query(self, v, n):
        self._cur.execute(self._query, (v, n), binary=True, prepare=True)
        return [id for id, in self._cur.fetchall()]

    def get_memory_usage(self):
        if self._cur is None:
            return 0
        self._cur.execute("SELECT pg_relation_size('items_embedding_idx')")
        return self._cur.fetchone()[0] / 1024

    def __str__(self):
        return f"PGVector(query_probes={self._query_probes})"
