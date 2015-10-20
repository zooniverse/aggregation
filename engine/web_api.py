#! /usr/bin/env python

from flask import Flask, make_response, request
from rq import Queue
from load_redis import configure_redis
from jobs import aggregate
import os
import json
import logging

app = Flask(__name__)
env = os.getenv('FLASK_ENV', 'production')
#30 mins - http://python-rq.org/docs/results/
q = Queue('default', connection=configure_redis(env), default_timeout=7200)
apis = {
    'development': "http://"+str(os.getenv('HOST_IP', '172.17.42.1'))+":3000",
    'staging': "https://panoptes-staging.zooniverse.org",
    'production': "https://panoptes.zooniverse.org"
}

api_root = apis[env]

@app.route('/',methods=['POST'])
def start_aggregation():
    try:
        body = request.get_json()
        project = body['project_id']
        href = body['medium_href']
        metadata = body['metadata']
        token = body['token']
        q.enqueue(aggregate, project, token, api_root+"/api"+href, metadata, env)
        resp = make_response(json.dumps({'queued': True}), 200)
        resp.headers['Content-Type'] = 'application/json'
        return resp
    except KeyError:
        resp = make_response(json.dumps({'error': [{'messages': "Missing Required Key"}]}), 422)
        resp.headers['Content-Type'] = 'application/json'
        return resp

@app.before_first_request
def setup_logging():
    if not app.debug:
        import logging
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        app.logger.addHandler(handler)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
