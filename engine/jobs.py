import os
import requests
import json
from aggregation_api import base_directory,AggregationAPI
from csv_output import CsvOut
import yaml
import rollbar
import traceback

def aggregate(project_id, token, href, metadata, environment):
    try:
        panoptes_file = open("/app/config/aggregation.yml","rb")
        api_details = yaml.load(panoptes_file)
        rollbar_token = api_details["default"]["rollbar"]
    except IOError:
        panoptes_file = open(base_directory+"/Databases/aggregation.yml","rb")
        api_details = yaml.load(panoptes_file)
        rollbar_token = api_details["staging"]["rollbar"]

    print rollbar_token
    rollbar.init(rollbar_token,environment)
    rollbar.report_message("step1","info")

    try:
        with AggregationAPI(project_id, environment=environment) as project:
            project.__migrate__()
            project.__aggregate__()
            rollbar.report_message("step2","info")

            with CsvOut(project) as writer:
                rollbar.report_message("step3","info")
                tarpath = writer.__write_out__(compress=True)
                response = send_uploading(metadata, token, href)
                url = response.json()["media"][0]["src"]
                with open(tarpath, 'rb') as tarball:
                    requests.put(url, headers={'Content-Type': 'application/x-gzip'}, data=tarball)
                os.remove(tarpath)
                send_finished(metadata, token, href)
                rollbar.report_message("step4","info")
    except Exception, err:
        print traceback.format_exc()
        rollbar.report_exc_info()



def get_etag(href, token):
    response = requests.get(href, headers=headers(token), params={'admin': True})
    return response.headers['ETag']

def send_uploading(metadata, token, href):
    metadata['state'] = 'uploading'
    body = { 'media': { 'metadata': metadata } }
    return send_request(body, token, href)

def send_finished(metadata, token, href):
    metadata['state'] = 'finished'
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





