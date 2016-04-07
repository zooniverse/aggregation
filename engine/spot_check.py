#!/usr/bin/env python
import aggregation_api
import yaml
import os
import boto3
import cassandra
__author__ = 'ggdhines'


def check_pid(pid):
    """ Check For the existence of a unix pid.
    http://stackoverflow.com/questions/568271/how-to-check-if-there-exists-a-process-with-a-given-pid
    """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def send_email(project_id):
    client = boto3.client('ses',region_name='us-east-1')
    response = client.send_email(
        Source='greg@zooniverse.org',
        Destination={
            'ToAddresses': [
                'greg@zooniverse.org'
            ]#,
        },
        Message={
            'Subject': {
                'Data': "project interrupted",
                'Charset': 'ascii'
            },
            'Body': {
                'Text': {
                    'Data': "Project " + str(project_id) + " was interrupted by development of new code.",
                    'Charset': 'ascii'
                }
            }
        },
        ReplyToAddresses=[
            'greg@zooniverse.org',
        ],
        ReturnPath='greg@zooniverse.org'
    )

param_file = open("/app/config/aggregation.yml","rb")
param_details = yaml.load(param_file)

cassandra_session = aggregation_api.AggregationAPI.__cassandra_connect__(param_details["production"]["cassandra"])

try:
    cassandra_session.execute("CREATE TABLE running_processes(pid int, project_id int, PRIMARY KEY(pid,project_id))")
except cassandra.AlreadyExists:
    pass

results = cassandra_session.execute("select pid,project_id from running_processes")

for pid,project_id in results:
    if not check_pid(pid):
        send_email(project_id)
        # and remove the project so we don't note it again
        cassandra_session.execute("delete from running_processes where pid = " + str(pid) + " and project_id = " + str(project_id))

