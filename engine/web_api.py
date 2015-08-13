#!/usr/bin/env python

from flask import Flask, make_response
from rq import Queue
from load_redis import configure_redis
from jobs import aggregate
import os
import json

app = Flask(__name__)
env = os.getenv('FLASK_ENV', 'production')
q = Queue(connection=configure_redis(env))
apis = {
    'development': "http://"+str(os.getenv('HOST_IP', '172.17.42.1'))+":3000",
    'stagin': "https://panoptes-staging.zooniverse.org",
    'production': "https://panoptes.zooniverse.org"

}

api_root = apis[env]

@app.route('/',methods=['POST'])
def start_aggregation():
    try:
        body = request.get_json()
        project = body['project_id']
        href = body['media_href']
        metadata = body['metadata']
        token = body['token']
        q.enqueue(aggregate, project, token, api_root+"/api"+href, metadata)
        resp = make_response(json.dumps({'queued': True}), 200)
        resp.headers['Content-Type'] = 'application/json'
        return resp
    except KeyError:
        resp = make_response(json.dumps({error: [{messages: "Missing Required Key"}]}), 422)
        resp.headers['Content-Type'] = 'application/json'
        return resp
    except:
        app.logger.info("TEST")
        app.logger.info(sys.exc_info()[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0')
