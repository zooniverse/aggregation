FROM ubuntu:14.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get install -y python python-dev python-setuptools python-pip \
    python-numpy python-scipy python-pymongo python-networkx python-yaml \
    python-psycopg2 python-matplotlib python-shapely supervisor

WORKDIR /app/

ADD . /app/

RUN pip install cassandra-driver
RUN pip install .

ADD supervisord.conf /etc/supervisor/conf.d/cron.conf

RUN ln -s /app/config/crontab /etc/cron.d/aggregation

ENTRYPOINT ["/usr/bin/supervisord"]
