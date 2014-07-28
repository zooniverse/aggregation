#!/usr/bin/env python
__author__ = 'greghines'

import sys
from sklearn.cluster import DBSCAN
import numpy as np
import shapes.ellipse,shapes.point



shapeMap = {"ellipse": shapes.ellipse, "point": shapes.point}


def analyze(currLayers, currAnotations, currGoldStandard):
    avgShapesPerLayer = []
    clustersSoFar = 0
    goldStandardSimilarity = []

    for layerIndex, layer in enumerate(currAnotations):
        #for now, we are just interested in clustering based on ellipse centers
        X = np.array([shapeMap[layerTypes[layerIndex]].DBSCANmap(shape) for shape in layer])
        db = DBSCAN(eps=0.3, min_samples=10).fit(X)
        labels = db.labels_

        #now average
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        avgShapesPerLayer.append([])

        #skip 0 since that is noise

        for clusterIndex in range(1,n_clusters_+1):
            cluster = [shape for i,shape in enumerate(layer) if labels[i] == clusterIndex]
            avgShape = [sum([s[i] for s in cluster])/float(len(cluster)) for i in range(len(cluster[0]))]
            avgShapesPerLayer[-1].append(avgShape[:])

            #now measure the "similarity" of this ellipse against all gold standard ones

            for goldIndex, goldStandard in currGoldStandard[layerIndex]:

                similarity = shapeMap[layerTypes[layerIndex]].similarity(avgShape,goldStandard)

                if similarity > 0.:
                    #since we
                    goldStandardSimilarity.append((clusterIndex+clustersSoFar,goldIndex,similarity))

        clustersSoFar += n_clusters_

    return avgShapesPerLayer, goldStandardSimilarity

current_zooniverse_id = None
annotationLayers = []
layerTypes = []
goldenStandardLayers = []

# input comes from STDIN (standard input)
for line in sys.stdin:
    # remove leading and trailing whitespace
    words = line.split("\t")
    subject_zooniverse_id = words[0]

    if subject_zooniverse_id != current_zooniverse_id:
        if current_zooniverse_id is not None:
            avgShapesPerLayer,goldStandardSimilarity = analyze(layerTypes,annotationLayers,goldenStandardLayers)

            print subject_zooniverse_id + " " + str(avgShapesPerLayer) + " " + str(goldStandardSimilarity)

        current_zooniverse_id = subject_zooniverse_id
        annotationLayers = []
        layerTypes = []
        goldenStandardLayers = []

    #find out what shape we are dealing with
    param = words[1].split(",")

    #first param is the user name, second is the shape, third is the layer - the rest are param which describe the shape
    layerIndex = int(param[2])

    #if we need to add more layers - do it now so we keep everything the same length (and assume that is will
    #save us the effort later)
    if len(annotationLayers) < layerIndex:
        #since the layers are not necessarily in order, may have to add multiple layers at once
        numNewLayers = layerIndex-len(annotationLayers)+1
        annotationLayers.extend([[] for i in range(numNewLayers)])
        layerTypes.extend([[] for i in range(numNewLayers)])
        goldenStandardLayers.extend([[] for i in range(numNewLayers)])

    #may have been previously set but just easier to assume not
    layerTypes[layerIndex] = param[1]

    if param[1] is not "goldStandard":
        #copy all of the params - so we can be indifferent about what shape it is
        # (and how many params it takes to describe it)
        annotationLayers[layerIndex].append([float(i) for i in param[3:]])
    else:
        goldenStandardLayers[layerIndex].append([float(i) for i in param[3:]])

if current_zooniverse_id is not None:
    avgShapesPerLayer,goldStandardSimilarity = analyze(layerTypes,annotationLayers,goldenStandardLayers)
    print subject_zooniverse_id + " " + str(avgShapesPerLayer) + " " + str(goldStandardSimilarity)