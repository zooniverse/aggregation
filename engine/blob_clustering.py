import clustering
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
import networkx as nx
import itertools
import math
import shapely
from shapely.validation import explain_validity
import random

def findsubsets(S,m):
    return set(itertools.combinations(S, m))


def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

class BlobClustering(clustering.Cluster):
    def __init__(self,shape,dim_reduction_alg):
        assert shape != "point"
        clustering.Cluster.__init__(self,shape,dim_reduction_alg)
        self.rectangle = (shape == "rectangle")

    def __fix_polygon__(self,points):
        fixed_polygons = None

        points = list(points)

        validity = explain_validity(Polygon(points))

        # x,y = zip(*Polygon(points).exterior.coords)
        # x = list(x)
        # y = list(y)
        # x.append(x[0])
        # y.append(y[0])
        # plt.plot(x,y)
        # plt.show()

        assert isinstance(validity,str)
        s,t = validity.split("[")
        x_0,y_0 = t.split(" ")
        x_0 = float(x_0)
        y_0 = float(y_0[:-1])

        # search for all of the line segments which touch the intersection point
        # we need to wrap around to the beginning to get all of the line segments
        splits = []
        for line_index in range(len(points)):
            (x_1,y_1) = points[line_index]
            (x_2,y_2) = points[(line_index+1)%len(points)]

            # the equation from a point to the nearest place on a line is from
            # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
            dist = math.fabs((y_2-y_1)*x_0-(x_2-x_1)*y_0+x_2*y_1-y_2*x_1)/math.sqrt((y_2-y_1)**2+(x_2-x_1)**2)
            if dist < 0.01:
                splits.append(line_index)

        # seems to be the easiest way to dealing with needing to extract
        # sublists which wrap around the end/beginning of the list

        points.extend(points)
        for intersection_index,line_index in enumerate(splits):
            # find the index for the next line segment with intersect (x_0,y_0)
            # if we have reached the end of the list then we need to wrap around
            if (intersection_index+1) < len(splits):
                line_index2 = splits[intersection_index+1]
            else:
                # keep in mind that we've doubled the length of points
                line_index2 = splits[0] + len(points)/2

            # always create the new polygon starting at the intersection point
            new_polygon_points = [(x_0,y_0)]
            new_polygon_points.extend(points[line_index+1:line_index2+1])

            if explain_validity(Polygon(new_polygon_points)) != "Valid Geometry":
                # if this is the first "sub"polygon - just accept it
                if fixed_polygons is None:
                    fixed_polygons = self.__fix_polygon__(new_polygon_points)
                # else try to merge the results in
                else:
                    fixed_polygons = fixed_polygons.union(self.__fix_polygon__(new_polygon_points))
            else:
                if fixed_polygons is None:
                    fixed_polygons = Polygon(new_polygon_points)
                else:
                    fixed_polygons = fixed_polygons.union(Polygon(new_polygon_points))

        return fixed_polygons


    def __inner_fit__(self,markings,user_ids,tools,reduced_markings):
        # convert markings to a list so that we can pop elements from it
        if len(set(user_ids)) > 15:
            selected_users = list(set(user_ids))
            random.shuffle(selected_users)
            selected_users = selected_users[:15]

            markings_and_users = [[m,u] for (m,u) in zip(markings,user_ids) if u in selected_users]
            markings,user_ids = zip(*markings_and_users)

        markings = list(markings)
        user_ids = list(user_ids)


        # markings = markings[:15]
        # user_ids = user_ids[:15]
        # tools = tools[:15]

        results = []
        if len(markings) > 1:
            blobs = []

            # graph is used to find cliques of overlapping polygon/blobs
            G=nx.Graph()

            # take markings from only the first 5 users
            # useable_users = list(set(user_ids))[:15]
            # all_combined = [(m,u,t) for m,u,t in zip(markings,user_ids,tools) if u in useable_users]
            # markings,user_ids,tools = zip(*all_combined)

            # hard code to a maximum of 15 users
            # markings = markings[:15]
            # user_ids = user_ids[:15]

            # check for bad polygons and discard if necessary
            # do it here so we don't have gaps in the ordering later
            for i in range(len(markings)-1,-1,-1):
                if len(markings[i]) <= 2:
                    markings.pop(i)
                    user_ids.pop(i)

            # convert to Shapely polygon objects
            for j,pts in enumerate(markings):
                assert isinstance(pts,list) or isinstance(pts,tuple)

                p = Polygon(pts)
                validity = explain_validity(p)

                if validity != "Valid Geometry":
                    # use below to see the before and after of the correction
                    #     t = pts[:]
                    #     t = list(t)
                    #     t.append(t[0])
                    #     t_x,t_y = zip(*t)
                    #     plt.plot(t_x,t_y,"-")
                    #     plt.show()
                    #
                    #
                    #     if isinstance(polygon_collection,Polygon):
                    #         x,y = polygon_collection.exterior.xy
                    #         x.append(x[0])
                    #         y.append(y[0])
                    #         plt.plot(x,y)
                    #     else:
                    #         for p in polygon_collection:
                    #             x,y = p.exterior.xy
                    #             x.append(x[0])
                    #             y.append(y[0])
                    #             plt.plot(x,y)
                    #
                    #     plt.show()
                    #     assert False
                    polygon_collection = self.__fix_polygon__(pts)
                    if isinstance(polygon_collection,Polygon):
                        blobs.append(polygon_collection)
                    else:
                        for p in polygon_collection:
                            blobs.append(p)
                else:
                    blobs.append(Polygon(list(pts)))

                G.add_node(j)



                for i,old_blob in enumerate(blobs[:-1]):
                    try:
                        if not blobs[-1].intersection(old_blob).is_empty:
                            G.add_edge(i,j)
                    except shapely.geos.TopologicalError:
                        print blobs[-1]
                        print old_blob
                        print explain_validity(blobs[-1])
                        print explain_validity(old_blob)
                        raise


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
                    blob_results = dict()

                    if self.rectangle:
                        x,y = union_blob.exterior.xy
                        x1 = min(x)
                        x2 = max(x)
                        y1 = min(y)
                        y2 = max(y)
                        blob_results["center"] = [(x1,y1),(x1,y2),(x2,y2),(x2,y1)]
                    else:
                        x,y = union_blob.exterior.xy
                        blob_results["center"] = zip(x,y)

                    blob_results["cluster members"] = []
                    blob_results["users"] = []
                    blob_results["tools"] = []

                    # todo: must be a better way to do this
                    # find which users overlap with this blob
                    for i in c:
                        if union_blob.intersects(blobs[i]):
                            blob_results["users"].append(user_ids[i])
                            x,y = blobs[i].exterior.xy
                            blob_results["cluster members"].append(zip(x,y)[:-1])
                            blob_results["tools"].append(tools[i])

                    results.append(blob_results)

                    # plt.plot(x,y,color="red")
                elif isinstance(union_blob,MultiPolygon):
                    for blob in union_blob:
                        blob_results = dict()
                        x,y = blob.exterior.xy
                        if self.rectangle:
                            x1 = min(x)
                            x2 = max(x)
                            y1 = min(y)
                            y2 = max(y)
                            blob_results["center"] = [(x1,y1),(x1,y2),(x2,y2),(x2,y1)]
                        else:
                            blob_results["center"] = zip(x,y)

                        blob_results["cluster members"] = []
                        blob_results["users"] = []
                        blob_results["tools"] = []

                        # todo: must be a better way to do this
                        # find which users overlap with this blob
                        for i in c:
                            if union_blob.intersects(blobs[i]):
                                blob_results["users"].append(user_ids[i])
                                x,y = blobs[i].exterior.xy
                                blob_results["cluster members"].append(zip(x,y)[:-1])
                                blob_results["tools"].append(tools[i])

                        results.append(blob_results)

        return results,0


