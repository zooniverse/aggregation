import os
from redis import Redis
from rq import Worker, Queue, Connection
from load_redis import configure_redis

listen = ['high', 'default', 'low']
env = os.getenv('FLASK_ENV', 'production')

conn = configure_redis(env)

with Connection(conn):
    worker = Worker(map(Queue, listen))
    worker.work()
