import os
import requests
import json
from csv_output import CsvOut


def aggregate(project_id, token, href, metadata, environment):
    from aggregation_api import AggregationAPI

    with AggregationAPI(project_id, environment=environment) as project:
        project.__setup__()
        project.__aggregate__()

        with CsvOut(project) as writer:
            tarpath = writer.__write_out__()
            response = send_uploading(metadata, token, href)
            url = response.json()["media"][0]["src"]
            with open(tarpath, 'rb') as tarball:
                requests.put(url, headers={'Content-Type': 'application/x-gzip'}, data=tarball)
            os.remove(tarpath)
            send_finished(metadata, token, href)

def get_etag(href, token):
    response = requests.get(href, headers=headers(token), params={'admin': True})
    return response.headers['ETag']

def send_uploading(metadata, token, href):
    metadata['state'] = 'uploading'
    body = { 'media': { 'metadata': metadata } }
    return send_request(body, token, href)

def send_finished(metadata, token, href):
    metadata['state'] = 'ready'
    body = { 'media': { 'metadata': metadata } }
    return send_request(body, token, href)

def headers(token, etag=False):
    headers = { 'Accept': 'application/vnd.api+json; version=1',
                'Content-Type': 'application/json',
                'Authorization': 'Bearer '+str(token) }
    if etag:
        headers['If-Match'] = get_etag(etag, token)
    return headers

def send_request(body, token, href):
    body['admin'] = True
    return requests.put(href, headers=headers(token, etag=href), data=json.dumps(body))
