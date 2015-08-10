import boto3
import os
import requests
import json
from aggregation_api import AggregationAPI

def aggregate(project_id, href, bucket, key, metadata):
    s3 = boto3.resource('s3')
    project = AggregationAPI(project_id)
    project.__aggregate__()
    tarpath = project.__csv_output__(compress=True)
    with open(tarpath, 'rb') as tarball:
        s3.Bucket(bucket).put_objject(Key=key, Body=tarball)
    os.remove(tarpath)
    metadata['state'] = 'ready'
    body = { 'media': { 'metadata': metadata } }
    headers = { 'Accept': 'application/vnd.api+json; version=1',
                'Content-Type': 'application/json',
                'Authorization': 'Bearer '+str(token) }

    requests.put(href, headers=headers, data=json.dumps(body))





