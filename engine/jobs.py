import os
import requests
import json
from aggregation_api import AggregationAPI

def aggregate(project_id, token, href, metadata, environment="development"):
    project = AggregationAPI(project_id, environment=environment)
    project.__aggregate__()
    tarpath = project.__csv_output__(compress=True)
    response = send_uploading(metadata)
    url = response.json()["media"][0]["src"]
    with open(tarpath, 'rb') as tarball:
        requests.put(url, data=tarball)
    os.remove(tarpath)
    send_finished(metadata)

def send_uploading(metadata):
    metadata['state'] = 'uploading'
    body = { 'media': { 'metadata': metadata } }
    send_request(body)

def send_finished(metadata):
    metadata['state'] = 'finished'
    body = { 'media': { 'metadata': metadata } }
    send_request(body)

def send_request(body):
    headers = { 'Accept': 'application/vnd.api+json; version=1',
                'Content-Type': 'application/json',
                'Authorization': 'Bearer '+str(token) }
    requests.put(href, headers=headers, data=json.dumps(body))





