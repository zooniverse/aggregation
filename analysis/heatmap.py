#!/usr/bin/env python
__author__ = 'greg'
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import csv
import json
from aggregation_api import AggregationAPI
import matplotlib.cbook as cbook
import sys

# subject_id = int(sys.argv[1])
# minimum_users = int(sys.argv[2])

subject_id = 511723

project = AggregationAPI(348,public_panoptes_connection=True)
subject_image = project.__image_setup__(subject_id)

for minimum_users in [8]:
    print minimum_users

    fig, ax = plt.subplots()

    image_file = cbook.get_sample_data(subject_image)
    image = plt.imread(image_file)
    # fig, ax = plt.subplots()
    im = ax.imshow(image)

    all_vertices = []

    with open("/tmp/348/4_ComplexAMOS/vegetation_polygons_heatmap.csv","rb") as f:
        polygon_reader = csv.reader(f)
        next(polygon_reader, None)
        for row in polygon_reader:
            if int(row[1]) == minimum_users:
                vertices = json.loads(row[2])
                all_vertices.append(vertices)



    all_vertices = np.asarray(all_vertices)
    coll = PolyCollection(all_vertices,alpha=0.3)
    ax.add_collection(coll)
    ax.autoscale_view()

    plt.show()
