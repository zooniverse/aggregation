import clustering
import matplotlib.pyplot as plt
from shapely.geometry import Polygon,MultiPolygon
import itertools
import math
from shapely.validation import explain_validity
import matplotlib
from shapely.ops import cascaded_union
import numpy

def findsubsets(S,m):
    return set(itertools.combinations(S, m))


def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)


class QuadTree:
    def __init__(self,((lb_x,lb_y),(ub_x,ub_y)),parent=None):
        self.lb_x = lb_x
        self.ub_x = ub_x
        self.lb_y = lb_y
        self.ub_y = ub_y

        self.children = None
        self.parent = parent
        self.polygons = {}

        self.bounding_box = Polygon([(lb_x,lb_y),(ub_x,lb_y),(ub_x,ub_y),(lb_x,ub_y)])


        self.user_ids = []

    def __get_splits__(self):
        if (self.bounding_box.area < 5) or (len(self.polygons) <= 2):
            return []

        complete_agreement = 0

        for user in self.polygons:
            # extract just the polygons - don't worry about type
            u = cascaded_union(zip(*self.polygons[user])[0])
            if math.fabs(self.bounding_box.intersection(u).area - self.bounding_box.area) < 1:
                complete_agreement += 1

        if complete_agreement >= 3:
            return []

        # calculate the height and width of the new children nodes
        new_width = (self.lb_x+self.ub_x)/2. - self.lb_x
        new_height = (self.ub_y+self.lb_y)/2. - self.lb_y

        lower_left = (self.lb_x,self.lb_y),(self.lb_x+new_width,self.lb_y+new_height)
        lower_right = (self.lb_x+new_width,self.lb_y),(self.ub_x,self.lb_y+new_height)
        upper_left = (self.lb_x,self.lb_y+new_height),(self.lb_x+new_width,self.ub_y)
        upper_right = (self.lb_x+new_width,self.lb_y+new_height),(self.ub_x,self.ub_y)

        self.children = [QuadTree(lower_left,self) ,QuadTree(lower_right,self),QuadTree(upper_left,self),QuadTree(upper_right,self)]
        for c in self.children:
            assert isinstance(c,QuadTree)
            assert c.bounding_box.area == self.bounding_box.area/4.

        return self.children

    def __plot__(self,ax):
        if self.children is None:
            if len(self.polygons) >= 3:
                plt.plot((self.lb_x,self.ub_x,self.ub_x,self.lb_x,self.lb_x),(self.lb_y,self.lb_y,self.ub_y,self.ub_y,self.lb_y),color="red")
                # rect = plt.Rectangle((self.lb_x,self.lb_y),(self.ub_x-self.lb_x),(self.ub_y-self.lb_y),color="red")
                # print (self.lb_x,self.lb_y),(self.ub_x-self.lb_x),(self.ub_y-self.lb_y)
                # ax.add_artist(rect)
        else:
            for c in self.children:
                assert isinstance(c,QuadTree)
                c.__plot__(ax)

    def __aggregate__(self,total_area):
        """
        total_area is needed for calculating the "noise" area
        :param total_area:
        :return:
        """
        if self.children is None:
            if len(self.polygons) >= 3:
                # what is the majority vote for what type of "kind" this box outlines
                # for example, some people might say broad leave tree while others say it is a needle leaf tree
                # technically speaking, people could outline this region with different polygons
                # if so, we'll ignore such users, under the assumption that we don't really know
                vote_counts = {}
                for u in self.polygons:
                    polys, types = zip(*self.polygons[u])
                    if min(types) == max(types):
                        vote = min(types)
                        if vote not in vote_counts:
                            vote_counts[vote] = 1
                        else:
                            vote_counts[vote] += 1

                # extract the most likely type according to pluraity voting
                # ties will get resolved in an arbitrary fashion
                most_likely,num_votes = sorted(vote_counts.items(),key = lambda x:x[1],reverse=True)[0]
                percentage = num_votes/float(sum(vote_counts.values()))

                # return two sets of values - one with the bounding box
                # the other with the voting results
                # last value is the noise area

                polygons_by_user_density = {i:[self.bounding_box] for i in range(3,len(self.polygons)+1)}
                return {most_likely:self.bounding_box},{most_likely:(percentage,self.bounding_box.area)},0,polygons_by_user_density
            else:
                if len(self.polygons) == 0:
                    return {},{},0,{}
                else:
                    # we have come polygons inside this box which amount to noise
                    # find out the area of these polygons - with respect to inside this box
                    # for smaller boxes, we could just use the area of the box as an approximation
                    # but for larger boxes, especially if we are the root, we need to do better
                    if (self.bounding_box.area/float(total_area)) > 0.02:
                        noise_polygons = []
                        for poly_list in self.polygons.values():
                            noise_polygons.extend(zip(*poly_list)[0])
                        combined_polygon = cascaded_union(noise_polygons).intersection(self.bounding_box)

                        return {},{},combined_polygon.area,{}
                    else:
                        return {},{},self.bounding_box.area,{}
        else:
            return_polygons = {}
            return_stats = {}
            new_area = {}
            new_percentage = {}

            total_incorrect_area = 0

            total_polygons_user_density = {}

            for c in self.children:
                # get all the polygons that are in c's bounding boxes plus stats about which tools made
                # which reasons
                c_polygons,c_stats,incorrect_area,polygons_by_user_density = c.__aggregate__(total_area)

                for u in polygons_by_user_density:
                    if u not in total_polygons_user_density:
                        total_polygons_user_density[u] = polygons_by_user_density[u]
                    else:
                        total_polygons_user_density[u].extend(polygons_by_user_density[u])

                total_incorrect_area += incorrect_area

                # go through the
                for tool_id in c_polygons:
                    # first time we've seen this polygon
                    if tool_id not in return_polygons:
                        return_polygons[tool_id] = c_polygons[tool_id]
                        # return_stats[tool_id] = c_stats[tool_id]

                        new_area[tool_id] = [c_stats[tool_id][1],]
                        new_percentage[tool_id] = [c_stats[tool_id][0],]
                    else:
                        # we need to merge (yay!!)
                        # note this will probably result in more than one polygon
                        return_polygons[tool_id] = return_polygons[tool_id].union(c_polygons[tool_id])
                        # I think I could fold these values in as I go, but just to be careful (don't want to mess
                        # with the stats) save the values until end
                        new_area[tool_id].append(c_stats[tool_id][1])
                        new_percentage[tool_id].append(c_stats[tool_id][0])

            # now that we've gone through all of the children
            # calculate the weighted values
            for tool_id in new_area:
                # print new_percentage[tool_id]
                return_stats[tool_id] = numpy.average(new_percentage[tool_id],weights=new_area[tool_id]),sum(new_area[tool_id])

                assert not isinstance(return_polygons[tool_id],list)

            if self.parent == None:
                for u in total_polygons_user_density:
                    total_polygons_user_density[u] = cascaded_union(total_polygons_user_density[u])

            return return_polygons,return_stats,total_incorrect_area,total_polygons_user_density

    def __add_polygon__(self,user,polygon,poly_type):
        assert isinstance(polygon,Polygon)
        # don't add it if there is no intersection
        if self.bounding_box.intersection(polygon).is_empty:
            return

        if user not in self.polygons:
            self.polygons[user] = [(polygon,poly_type)]
            self.user_ids.append(user)
        else:
            self.polygons[user].append((polygon,poly_type))

    def __poly_iteration__(self):
        class Iterator:
            def __init__(self,node):
                assert isinstance(node,QuadTree)
                self.node = node
                self.user_index = None
                self.polygon_index = None

            def __iter__(self):
                return self

            def next(self):
                if self.user_index is None:
                    self.user_index = 0
                    self.polygon_index = 0
                else:
                    self.polygon_index += 1
                    if self.polygon_index == len(self.node.polygons[self.node.user_ids[self.user_index]]):
                        self.polygon_index = 0
                        self.user_index += 1

                    if self.user_index == len(self.node.user_ids):
                        raise StopIteration

                user_id = self.node.user_ids[self.user_index]
                return user_id,self.node.polygons[user_id][self.polygon_index]

        return Iterator(self)


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
                    fixed_polygons.extend(self.__fix_polygon__(new_polygon_points))
            else:
                if fixed_polygons is None:
                    fixed_polygons = [Polygon(new_polygon_points)]
                else:
                    fixed_polygons.append(Polygon(new_polygon_points))

        return fixed_polygons

    def __cluster__(self,markings,user_ids,tools,reduced_markings,dimensions):
        poly_dictionary = {}
        for polygon_pts,u,t in zip(markings,user_ids,tools):
            # we need at least 3 points to made a valid polygon
            if len(polygon_pts) < 3:
                continue

            poly = Polygon(polygon_pts)
            validity = explain_validity(poly)

            if validity != "Valid Geometry":
                corrected_polygon = self.__fix_polygon__(polygon_pts)
                if isinstance(corrected_polygon,Polygon):
                    if u not in poly_dictionary:
                        poly_dictionary[u] = [(corrected_polygon,t),]
                    else:
                        for p in corrected_polygon:
                            poly_dictionary[u].append((p,t))
            else:
                if u not in poly_dictionary:
                    poly_dictionary[u] = [(poly,t)]
                else:
                    poly_dictionary[u].append((poly,t))

        # if dimensions have been provided, use those as our initial bounding box
        # otherwise, use some default values - I have NO idea how well those will work in practice
        if dimensions is None:
            box = [[0,0],[800,0],[800,500],[0,500]]
        else:
            box = [[0,0],[dimensions[1],0],[dimensions[1],dimensions[0]],[0,dimensions[1]]]

        quad_root = QuadTree((box[0],box[2]))

        for user,polygon_list in poly_dictionary.items():
            for polygon,poly_type in polygon_list:
                quad_root.__add_polygon__(user,polygon,poly_type)

        to_process = [quad_root]

        while to_process != []:
            node = to_process.pop(-1)
            assert isinstance(node,QuadTree)

            # if (we have parent =>  !the root) => need to read in parent's polygons
            # some of which will become ours
            if node.parent is not None:
                for user,(poly,poly_type) in node.parent.__poly_iteration__():
                    node.__add_polygon__(user,poly,poly_type)

            new_children = node.__get_splits__()

            to_process.extend(new_children)

        # total incorrect area is the total area which at least one person marked/outlined by not enough people
        # so typically just 1 or 2 (unless we change the threshold)
        image_area = dimensions[0]*dimensions[1]

        aggregate_polygons,aggregate_stats,total_incorrect_area,polygons_by_user_density = quad_root.__aggregate__(image_area)

        for tool_id in aggregate_stats:
            vote_percentage,tool_area = aggregate_stats[tool_id]
            aggregate_stats[tool_id] = vote_percentage,tool_area/float(image_area)

        incorrect_area_as_percent = total_incorrect_area/image_area

        results = []
        # a lot of this stuff is done in the classification code for other tool types but it makes more
        # sense for polygons to do it here
        for tool_id in aggregate_polygons:
            # have one set of results per tool type
            # the center will be a list of all polygons
            next_result = dict()


            # we will either have a multi-polygon as our aggregation result - or if we are really
            # lucky, a single polygon
            if isinstance(aggregate_polygons[tool_id],Polygon):
                poly = aggregate_polygons[tool_id]
                next_result["center"] = [zip(poly.exterior.xy[0],poly.exterior.xy[0])]
            elif isinstance(aggregate_polygons[tool_id], MultiPolygon):
                next_result["center"] = []
                for poly in aggregate_polygons[tool_id]:

                    # go through each of the individual polygons making up this multipolygon
                    if isinstance(poly,Polygon):
                        next_result["center"].append(zip(poly.exterior.xy[0],poly.exterior.xy[0]))
            else:
                # unknown type
                print type(aggregate_polygons[tool_id])
                assert False

            # a value of None -> not really relevant to polygon aggregation
            # or really hard to keep track of
            next_result["users"] = None
            next_result["num users"] = None
            next_result["cluster_members"] = None
            next_result["tool classification"] = tool_id

            # these are global values which are not really specific to any one polygon
            # but this seems to be the best place to store the values
            next_result["incorrect area"] = incorrect_area_as_percent
            # todo - don't hard code this
            next_result["minimum users"] = 3
            next_result["area"] = aggregate_stats[tool_id][1]
            next_result["certainty"] = aggregate_stats[tool_id][0]

            results.append(next_result)

        # these results might get really big so I don't want to waste space repeatedly storing the values
        # so with slight abuse of setup ...
        for num_users,polygons in polygons_by_user_density.items():
            # probably never going to wind up a just a single polygon
            # but we can always hope
            if isinstance(polygons,Polygon):
                assert False
            else:
                for single_poly in polygons:
                    assert  isinstance(single_poly,Polygon)
                    x,y = single_poly.exterior.xy
                    pts = zip(x,y)

                    next_result = {"users":None,"num users": num_users,"tool classification" : None, "area":single_poly.area, "certainty":None}
                    next_result["center"] = pts

                    results.append(next_result)

        if results == []:
            # add in a dummy polygon so that we can report back the size of
            # the incorrect area
            results = [{"area":0,"incorrect area":incorrect_area_as_percent,"certainty": -1},]

        return results,0
