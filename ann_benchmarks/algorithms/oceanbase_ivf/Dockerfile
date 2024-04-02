FROM ann-benchmarks

COPY ann_benchmarks/algorithms/oceanbase_ivf/deb /root/ob_deb
RUN cd /root/ob_deb && apt install -y ./oceanbase-ce_4.3.0.0_amd64.deb \
    && apt install -y ./oceanbase-ce-libs_4.3.0.0_amd64.deb 
COPY ann_benchmarks/algorithms/oceanbase_ivf/oceanbase.cnf /etc/oceanbase.cnf
RUN pip3 install pymysql \
    && apt-get install -y curl python2 mysql-server \
    && sed -i '228,258d' /home/admin/oceanbase/profile/oceanbase-service.sh \
    && /bin/bash /home/admin/oceanbase/profile/oceanbase-service.sh start \ 
    && mysql -uroot -P2881 -h127.1 -e "create resource unit unit_1 max_cpu 10, memory_size \"10G\", log_disk_size \"40G\";" \
    && mysql -uroot -P2881 -h127.1 -e "create resource pool pool_2 unit = 'unit_1', unit_num = 1, zone_list = ('zone1');" \
    && mysql -uroot -P2881 -h127.1 -e "create tenant perf replica_num = 1,primary_zone='zone1', resource_pool_list=('pool_2') set ob_tcp_invited_nodes='%';" \
    && /bin/bash /home/admin/oceanbase/profile/oceanbase-service.sh stop

