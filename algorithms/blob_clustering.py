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

        if (len(markings) > 1) and (fname is not None):
            print user_ids
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            image_file = cbook.get_sample_data(fname)
            image = plt.imread(image_file)
            # fig, ax = plt.subplots()
            im = ax.imshow(image)

            blobs = []
            G=nx.Graph()

            for j,(x1,y1,x2,y2) in enumerate(markings):
                blobs.append(Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)]))
                G.add_node(j)

                for i,old_blob in enumerate(blobs[:-1]):
                    if not blobs[-1].intersection(old_blob).is_empty:
                        G.add_edge(i,j)

                plt.plot([x1,x2],[y1,y1],color="blue")
                plt.plot([x1,x2],[y2,y2],color="blue")
                plt.plot([x1,x1],[y1,y2],color="blue")
                plt.plot([x2,x2],[y1,y2],color="blue")

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

                            print union_blob
                            print overlap
                            if union_blob is not None:
                                print union_blob.union(overlap)
                            print
                            if union_blob is None:
                                union_blob = overlap
                            else:
                                union_blob = union_blob.union(overlap)
                if isinstance(union_blob,Polygon):
                    print union_blob.area
                    x,y = union_blob.exterior.xy
                    plt.plot(x,y,color="red")
                elif isinstance(union_blob,MultiPolygon):
                    for blob in union_blob:
                        print blob.area
                        x,y = blob.exterior.xy
                        plt.plot(x,y,color="red")
            plt.show()


            #     plt.plot([x1,x2],[y1,y1],color="blue")
            #     plt.plot([x1,x2],[y2,y2],color="blue")
            #     plt.plot([x1,x1],[y1,y2],color="blue")
            #     plt.plot([x2,x2],[y1,y2],color="blue")
            # plt.show()

        return {},0
