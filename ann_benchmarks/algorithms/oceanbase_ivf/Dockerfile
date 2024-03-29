FROM ann-benchmarks

COPY ann_benchmarks/algorithms/oceanbase_ivf/systemctl /root
RUN apt-get install -y python2 mysql-server
RUN cat /root/systemctl > /usr/bin/systemctl && \
    chmod a+x /usr/bin/systemctl

COPY ann_benchmarks/algorithms/oceanbase_ivf/deb /root/ob_deb
RUN cd /root/ob_deb && apt install -y ./oceanbase-ce_4.3.0.0_amd64.deb
RUN cd /root/ob_deb && apt install -y ./oceanbase-ce-libs_4.3.0.0_amd64.deb
RUN cd /root/ob_deb && apt install -y ./oceanbase-ce-sql-parser_4.3.0.0_amd64.deb
RUN cd /root/ob_deb && apt install -y ./oceanbase-ce-table_4.3.0.0_amd64.deb
RUN pip3 install pymysql
RUN systemctl start oceanbase

# RUN mysql -uroot -P2881 -h127.1 -e "select 1+1;"