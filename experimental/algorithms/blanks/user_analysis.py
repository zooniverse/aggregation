#!/usr/bin/env python
__author__ = 'greg'
from cassandra.cluster import Cluster
import numpy
import matplotlib.pyplot as plt
import datetime
import csv
import bisect
import random
import json
import matplotlib.pyplot as plt

def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError

# load subject data from CSV
subjects_index = {}
with open('/home/greg/Documents/subject_species_all.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in reader:
        subjects_index[row[1]] = row[2]

def unix_time(dt):
    epoch = datetime.datetime.utcfromtimestamp(0)
    delta = dt - epoch
    return delta.total_seconds()

def unix_time_millis(dt):
    return long(unix_time(dt) * 1000.0)

cluster = Cluster()
cassandra_session = cluster.connect('serengeti')

ips = []

for ii,row in enumerate(cassandra_session.execute("select * from classifications where id =1")):
    try:
        index(ips,row.user_ip)
    except ValueError:
        bisect.insort(ips,row.user_ip)
    # ips.add(row.user_ip)

    if ii == 100000:
        break

animal_accuracy = []

for ip in random.sample(ips,500):
    true_blank = 0.
    false_blank = 0.
    true_animal = 0.
    false_animal = 0.

    for classification in cassandra_session.execute("select * from ip_classifications where id =1 and user_ip='"+str(ip)+"'"):
        zooniverse_id = classification.zooniverse_id
        annotatons = json.loads(classification.annotations)

        nothing = "nothing" in annotatons[-1]
        if subjects_index[zooniverse_id]=="blank":
            if nothing:
                true_blank += 1
            else:
                false_animal += 1
        else:
            if nothing:
                false_blank += 1
            else:
                true_animal += 1

    if (true_animal+false_blank) == 0:
        continue
    animal_accuracy.append(true_animal/(true_animal+false_blank))

plt.hist(animal_accuracy,50,cumulative=True,normed=1)
plt.show()