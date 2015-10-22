import os
from redis import Redis
from rq import Worker, Queue, Connection
from load_redis import configure_redis
from aggregation_api import base_directory
import yaml
import rollbar
from rollbar.contrib.rq import exception_handler

listen = ['high', 'default', 'low']
env = os.getenv('FLASK_ENV', 'production')

try:
    panoptes_file = open("config/aggregation.yml","rb")
except IOError:
    panoptes_file = open(base_directory+"/Databases/aggregation.yml","rb")

api_details = yaml.load(panoptes_file)
rollbar_token = api_details[env]["rollbar"]
# rollbar.init(rollbar_token, env, handler='blocking')

conn = configure_redis(env)
with Connection(conn):
    worker = Worker(map(Queue, listen))
    worker.push_exc_handler(rollbar.contrib.rq.exception_handler)
    worker.work()
