import clustering
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
import networkx as nx
import itertools

def findsubsets(S,m):
    return set(itertools.combinations(S, m))

class BlobClustering(clustering.Cluster):
    def __init__(self,project_api,shape):
        clustering.Cluster.__init__(self,project_api,shape)
        self.algorithm_name = "blob clustering"

    def __inner_fit__(self,markings,user_ids,tools,fname=None):

        results = []
        if len(markings) > 1:
            blobs = []
            G=nx.Graph()

            for j,pts in enumerate(markings):
                assert isinstance(pts,list) or isinstance(pts,tuple)
                blobs.append(Polygon(pts))
                G.add_node(j)

                for i,old_blob in enumerate(blobs[:-1]):
                    if not blobs[-1].intersection(old_blob).is_empty:
                        G.add_edge(i,j)

            for c in nx.connected_components(G):

                union_blob = None
                if len(c) >= 3:
                    overlapping_blobs = [blobs[k] for k in c]
                    for subset in itertools.combinations(overlapping_blobs,3):
                        overlap = subset[0]
                        for b in subset[1:]:
                            overlap = overlap.intersection(b)
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


