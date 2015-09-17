import clustering
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import itertools
import math
from shapely.validation import explain_validity
import matplotlib
from shapely.ops import cascaded_union

def findsubsets(S,m):
    return set(itertools.combinations(S, m))


def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)


class Controller:
    def __init__(self):
        self.generator = None

    def __register__(self,generator):
        assert isinstance(generator,SubsetGenerator)
        self.generator = generator

    def __skip__(self,branch_to_skip):
        self.generator.__skip__(branch_to_skip)


class SubsetGenerator:
    def __init__(self,elements,subset_size,controller):
        assert len(elements) >= subset_size
        assert isinstance(controller,Controller)
        self.elements = elements
        self.num_elements = len(elements)
        self.subset_size = subset_size

        self.curr_subset_indices = None

        # start off assuming that we can increment to another subset, but don't specify which one
        self.can_increment = -1
        self.saved_subset = None

        controller.__register__(self)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    # todo - REFACTOR!!!
    def __skip__(self,branch_to_skip):
        """
        skip a given branch - because of the way Python works, this change can happen immediately
        for example we could have
        print a => [0,1,2]
        __skip__(0)
        print a=> [1,2,3]
        without having to go through the loop again - don't want this
        so we will save the results and the next "next" call will return those saved results
        :param branch_to_skip:
        :return:
        """
        self.saved_subset = self.curr_subset_indices[:]

        self.can_increment = None
        for i in range(len(branch_to_skip)):
            if i == (len(branch_to_skip)-1):
                if branch_to_skip[i] < (self.num_elements -1):
                    self.can_increment = i
                else:
                    if (branch_to_skip[i]+1) < branch_to_skip[i]:
                        self.can_increment = i

        # if we have found a place to increment, do it
        # otherwise, the next call to "next" should raise a StopIncrement error
        if self.can_increment is not None:
            self.saved_subset[self.can_increment] += 1
            # reset all the indices to the right of the one we just updated (even if they weren't
            # explicity mentioned in the branch_to_skip
            for i in range(self.can_increment+1,len(self.curr_subset_indices)):
                self.saved_subset[i] = self.saved_subset[i-1] + 1

    def next(self):
        # check if a skip call resulted in us finishing the iteration
        # we can probably get away with just checking once after we've updated (in which case the update
        # is invalid and irrelevant) but I want to make sure that such an update doesn't do anything funny
        if self.can_increment is None:
            raise StopIteration()

        # if we have stored values as the results of skipping some branches
        if self.saved_subset is not None:
            self.curr_subset_indices = self.saved_subset[:]
            self.saved_subset = None
        else:
            if self.curr_subset_indices is None:
                self.curr_subset_indices = range(self.subset_size)
            else:
                can_increment = None
                for i in range(len(self.curr_subset_indices)):
                    if i == (len(self.curr_subset_indices)-1):
                        if self.curr_subset_indices[i] < (self.num_elements-1):
                            can_increment = i
                    else:
                        if (self.curr_subset_indices[i]+1) < self.curr_subset_indices[i+1]:
                            can_increment = i

                if can_increment is None:
                    raise StopIteration()

                self.curr_subset_indices[can_increment] += 1
                for i in range(can_increment+1,len(self.curr_subset_indices)):
                    self.curr_subset_indices[i] = self.curr_subset_indices[i-1] + 1

        return [self.elements[j] for j in self.curr_subset_indices]


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
            u = cascaded_union(self.polygons[user])
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



    def __add_polygon__(self,user,polygon,debug=False):
        assert isinstance(polygon,Polygon)
        # don't add it if there is no intersection
        if self.bounding_box.intersection(polygon).is_empty:

            return

        if debug:
            plt.plot((self.lb_x,self.ub_x,self.ub_x,self.lb_x,self.lb_x),(self.lb_y,self.lb_y,self.ub_y,self.ub_y,self.lb_y))
            x,y = polygon.exterior.xy
            plt.plot(x,y)
            # box = [[0,0],[800,0],[800,500],[0,500]]
            plt.xlim((0,800))
            plt.ylim((0,500))
            plt.show()

        if user not in self.polygons:
            self.polygons[user] = [polygon]
            self.user_ids.append(user)
        else:
            self.polygons[user].append(polygon)

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

    def __inner_fit__(self,markings,user_ids,tools,reduced_markings):
        poly_dictionary = {}
        for polygon_pts,u in zip(markings,user_ids):
            # we need at least 3 points to made a valid polygon
            if len(polygon_pts) < 3:
                continue

            poly = Polygon(polygon_pts)
            validity = explain_validity(poly)

            if validity != "Valid Geometry":
                corrected_polygon = self.__fix_polygon__(polygon_pts)
                # if isinstance(polygon_collection,Polygon):
                #     valid_polygons = polygon_collection]
                # else:
                #     # if the valid polygon is actually a group of polygon add them separately
                #     valid_polygons = []
                #     for p in polygon_collection:
                #         valid_polygons.append(p)
            else:
                corrected_polygon = [poly]

            if u not in poly_dictionary:
                poly_dictionary[u] = corrected_polygon
            else:
                # update the user's polygon by taking the union
                poly_dictionary[u].extend(corrected_polygon)

        box = [[0,0],[800,0],[800,500],[0,500]]

        quad_root = QuadTree((box[0],box[2]))

        for user,polygon_list in poly_dictionary.items():
            for polygon in polygon_list:
                quad_root.__add_polygon__(user,polygon)

        to_process = [quad_root]

        while to_process != []:
            node = to_process.pop(-1)
            assert isinstance(node,QuadTree)


            # if we have parent != the root => need to read in
            if node.parent is not None:
                for user,poly in node.parent.__poly_iteration__():
                    node.__add_polygon__(user,poly,debug=False)

            new_children = node.__get_splits__()

            to_process.extend(new_children)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        quad_root.__plot__(ax)
        plt.show()
        assert False

    def __put_markings_in_boxes__(self,bounding_box):
        for polygons in poly_dictionary.values():
            for p in polygons:
                print bounding_box.intersection(p).is_empty

                x,y = p.exterior.xy
                plt.plot(x,y)
        plt.show()
        assert False



if __name__ == "__main__":
    c = Controller()
    for a in SubsetGenerator(range(5),3,c):
        print a
        if a[0] == 0:
            c.__skip__([0])
        print a
        print