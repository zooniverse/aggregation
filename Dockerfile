FROM zooniverse/aggregation-dependencies

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get install -y python python-dev python-setuptools python-pip \
    python-numpy python-scipy python-pymongo python-networkx python-yaml \
    python-psycopg2 python-matplotlib python-shapely python-pandas supervisor \
    mafft vim python-wxgtk2.8

WORKDIR /app/

ADD . /app/

RUN pip install -U cassandra-driver Flask redis rq requests==2.4.3 rollbar termcolor boto3 setproctitle alignment scikit-learn boto descartes
RUN pip install .

ADD supervisord.conf /etc/supervisor/conf.d/cron.conf

RUN ln -s /app/config/crontab /etc/cron.d/aggregation

EXPOSE 5000

ENTRYPOINT ["/app/start.sh"]
