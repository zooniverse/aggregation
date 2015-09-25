import os
from redis import Redis
from rq import Worker, Queue, Connection
from load_redis import configure_redis
from aggregation_api import base_directory
import yaml
import rollbar


def rollbar_handler(job, exc_type, exc_value, traceback):
    try:
        panoptes_file = open("config/aggregation.yml","rb")
    except IOError:
        panoptes_file = open(base_directory+"/Databases/aggregation.yml","rb")
    api_details = yaml.load(panoptes_file)

    if "rollbar" in api_details["default"]:
        rollbar_token = api_details["default"]["rollbar"]
        rollbar.init(rollbar_token,"production")
        rollbar.report_exc_info()

listen = ['high', 'default', 'low']
env = os.getenv('FLASK_ENV', 'production')

conn = configure_redis(env)

with Connection(conn):
    worker = Worker(map(Queue, listen),exc_handler=rollbar_handler)
    worker.work()
