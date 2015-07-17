import clustering
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
import networkx as nx
import itertools
import math

def findsubsets(S,m):
    return set(itertools.combinations(S, m))


def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

class BlobClustering(clustering.Cluster):
    def __init__(self,shape):
        assert shape != "point"
        clustering.Cluster.__init__(self,shape)

    def __inner_fit__(self,markings,user_ids,tools,fname=None):

        results = []
        if len(markings) > 1:
            blobs = []
            G=nx.Graph()

            # take markings from only the first 5 users
            useable_users = list(set(user_ids))[:15]
            all_combined = [(m,u,t) for m,u,t in zip(markings,user_ids,tools) if u in useable_users]
            markings,user_ids,tools = zip(*all_combined)

            # check for bad polygons and discard if necessary
            # do it here so we don't have gaps in the ordering later
            for i in range(len(markings)-1,-1,-1):
                if len(markings[i]) <= 2:
                    markings.pop(i)
                    user_ids.pop(i)
                    tools.pop(i)

            for j,pts in enumerate(markings):
                assert isinstance(pts,list) or isinstance(pts,tuple)
                try:
                    blobs.append(Polygon(pts))
                except TypeError:
                    print pts
                    raise
                G.add_node(j)

                for i,old_blob in enumerate(blobs[:-1]):
                    if not blobs[-1].intersection(old_blob).is_empty:
                        G.add_edge(i,j)

            for c in nx.connected_components(G):

                union_blob = None
                if len(c) >= 3:
                    overlapping_blobs = [blobs[k] for k in c]
                    num_blobs = 3
                    # while nCr(len(overlapping_blobs),num_blobs) > 20:
                    #     print "** " + str(num_blobs)
                    #     num_blobs += 1

                    for subset_count, subset in enumerate(itertools.combinations(overlapping_blobs,num_blobs)):

                        overlap = subset[0]
                        for b in subset[1:]:
                            overlap = overlap.intersection(b)

                        # if isinstance(overlap,MultiPolygon):
                        #     overlap = overlap.geoms[0]
                        # print type(overlap)
                        # assert isinstance(overlap,Polygon)

                        if not overlap.is_empty:
                            # # print type(overlap)
                            # x,y = overlap.exterior.xy
                            # plt.plot(x,y,color="red")

                            if union_blob is None:
                                union_blob = overlap
                            else:
                                union_blob = union_blob.union(overlap)

                if isinstance(union_blob,Polygon):
                    x,y = union_blob.exterior.xy
                    x1 = min(x)
                    x2 = max(x)
                    y1 = min(y)
                    y2 = max(y)

                    blob_results = dict()
                    blob_results["center"] = [(x1,y1),(x1,y2),(x2,y2),(x2,y1)]
                    blob_results["points"] = []
                    blob_results["users"] = []

                    # todo: must be a better way to do this
                    # find which users overlap with this blob
                    for i in c:
                        if union_blob.intersects(blobs[i]):
                            blob_results["users"].append(user_ids[i])
                            x,y = blobs[i].exterior.xy
                            blob_results["points"].append(zip(x,y)[:-1])

                    results.append(blob_results)

                    # plt.plot(x,y,color="red")
                elif isinstance(union_blob,MultiPolygon):
                    for blob in union_blob:
                        x,y = blob.exterior.xy
                        x1 = min(x)
                        x2 = max(x)
                        y1 = min(y)
                        y2 = max(y)

                        blob_results = dict()
                        blob_results["center"] = [(x1,y1),(x1,y2),(x2,y2),(x2,y1)]
                        blob_results["points"] = []
                        blob_results["users"] = []

                        # todo: must be a better way to do this
                        # find which users overlap with this blob
                        for i in c:
                            if union_blob.intersects(blobs[i]):
                                blob_results["users"].append(user_ids[i])
                                x,y = blobs[i].exterior.xy
                                blob_results["points"].append(zip(x,y)[:-1])

                        results.append(blob_results)

        return results,0


