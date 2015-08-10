#!/usr/bin/env python

from flask import Flask, make_response
from rq import Queue
from load_redis import configure_redis
from jobs import aggregate
import os
import json

app = Flask(__name__)
q = Queue(connection=configure_redis(os.getenv('FLASK_ENV', 'production')))

@app.route('/',methods=['POST'])
def start_aggregation():
    try:
        body = request.get_json()
        project = body['project_id']
        bucket = body['bucket']
        key = body['key']
        href = body['href']
        metadata = body['metadata']
        q.enqueue(aggregate, project, href, bucket, key, metadata)
        resp = make_response(json.dumps({'queued': True}), 200)
        resp.headers['Content-Type'] = 'application/json'
        return resp
    except KeyError:
        resp = make_response(json.dumps({error: [{messages: "Missing Required Key"}]}), 422)
        return resp

if __name__ == '__main__':
    app.run(host='0.0.0.0')
