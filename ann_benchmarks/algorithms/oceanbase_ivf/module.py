import subprocess
import sys

import pymysql as mysql # need to pip install

from datetime import datetime
from ..base.module import BaseANN

class OBVector(BaseANN):
    def __init__(self, metric, method_param):
        try:
            result = subprocess.run('/root/systemctl start oceanbase', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            print("错误：", e)
        self._metric = metric
        self._cur = None
        self._query_probes = None

        if metric == "angular":
            self._query = "SELECT /*+query_timeout(1000000000)*/id FROM items ORDER BY embedding <~> '[%s]' LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT /*+query_timeout(1000000000) */id FROM items ORDER BY embedding <-> '[%s]' LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def fit(self, X):
        conn = mysql.connect(host="127.0.0.1", user="root@perf", port=2881, passwd="", database="test")
        cur = conn.cursor()

        cur.execute("DROP TABLE IF EXISTS items")
        cur.execute("CREATE TABLE items (id int, embedding vector(%d), primary key (id)) partition by key(id) partitions 100" % X.shape[1])
        cur.execute("set autocommit=1")
        print("copying data: data size: %d..." % X.shape[0])
        for i, embedding in enumerate(X):
            cur.execute("insert into items values (%d, '[%s]')" % (i, ",".join(str(d) for d in embedding)))
            if i % 1000 == 0:
                print("%d copied" % i)

        cur.execute("alter system minor freeze")
        cur.execute("select sleep(5)")

        index_start_time = datetime.now()
        print("{}: start create index!".format(index_start_time))
        cur.execute("alter system set vector_ivfflat_elkan='True'")
        cur.execute("alter system set vector_ivfflat_iters_count=200")

        if self._metric == "angular":
            cur.execute("CREATE INDEX items_ivfflat_idx ON items (embedding cosine) USING ivfflat with(lists=245)")
        elif self._metric == "euclidean":
            cur.execute("CREATE /*+parallel(50)*/INDEX items_ivfflat_idx ON items (embedding l2) USING ivfflat with(lists=8)")
        print("{}: finish create index! build index in {}".format(datetime.now(), datetime.now()-index_start_time))

        cur.execute("alter system minor freeze")
        cur.execute("select sleep(5)")

        #print("prepare nothing")
        self._cur = cur

    def set_query_arguments(self, n_probes):
        self._query_probes = n_probes
        self._cur.execute("set @@vector_ivfflat_probes = %d" % n_probes)
        return

    def query(self, v, n):
        self._cur.execute(self._query % ((",".join(str(d) for d in v), n)))
        return [id for id, in self._cur.fetchall()]

    def get_memory_usage(self):
        # TODO: memory usage info
        return 0
        # if self._cur is None:
        #     return 0
        # self._cur.execute("SELECT pg_relation_size('items_embedding_idx')")
        # return self._cur.fetchone()[0] / 1024

    def __str__(self):
        return f"OBVector(query_probes={self._query_probes})"