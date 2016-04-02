from __future__ import print_function
import clustering
import networkx
import numpy as np


class RectangleClustering(clustering.Cluster):
    def __init__(self,shape,project,additional_params):
        assert shape != "point"
        clustering.Cluster.__init__(self,shape,project,additional_params)
        self.rectangle = (shape == "rectangle") or (shape == "image")

    def __overlap__(self,l1,l2):
        """
        do two lines overlap? assume horizontal lines - if vertical then you need to flip coordinates before calling
        :param l1:
        :param l2:
        :return:
        """
        (l1_a,l1_b) = l1
        (l2_a,l2_b) = l2

        # l2 ends before l1 begins
        if l2_b < l1_a:
            return False
        # l2 starts after l1 ends
        elif l2_a > l1_b:
            return False
        else:
            return True

    def __overlap_graph__(self,markings):
        """
        given a set of rectangle markings return a graph where each node corresponds to a rectangle
        and an edge exists iff two rectangles overlap
        :param markings:
        :return:
        """
        g = networkx.Graph()
        g.add_nodes_from(range(len(markings)))

        # go through each pair of rectangles and see if they overlap
        for i,((x1,y1),_,(x2,y2),_) in enumerate(markings):
            for j,((m1,n1),_,(m2,n2),_) in list(enumerate(markings))[i+1:]:
                # do these rectangles overlap on the x axis?
                overlap_x = self.__overlap__((x1,x2),(m1,m2))
                if not overlap_x:
                    continue

                # and on the y axis?
                overlap_y = self.__overlap__((y1,y2),(n1,n2))
                if not overlap_y:
                    continue

                # we know that these rectangles overlap
                g.add_edge(i,j)

        return g

    def __median_rectangles__(self,markings):
        """
        given a set of rectangles (which should represent a clique)
        create a "representative" rectangle based on median corners
        :param markings:
        :return:
        """
        # don't assume that all rectangles will be in the same order
        # e.g. don't assume that the first point is the lower left hand corner
        maximum_x = [max(m[0][0],m[2][0]) for m in markings]
        minimum_x = [min(m[0][0],m[2][0]) for m in markings]

        maximum_y = [max(m[0][1],m[2][1]) for m in markings]
        minimum_y = [min(m[0][1],m[2][1]) for m in markings]

        x_top = np.median(maximum_x)
        x_bot = np.median(minimum_x)
        y_top = np.median(maximum_y)
        y_bot = np.median(minimum_y)

        return (x_top,y_top),(x_bot,y_bot)

    def __cluster__(self,markings,user_ids,tools,reduced_markings,dimensions,subject_id):
        """
        main clustering algorithm - works on a single per-subject basis
        for rectangles, doesn't make use of reduced_markings
        :param markings:
        :param user_ids:
        :param tools:
        :param reduced_markings:
        :param dimensions:
        :param subject_id:
        :return:
        """
        # if empty markings, just return nothing
        if markings == []:
            return [],0

        results = []

        overlap_graph = self.__overlap_graph__(markings)

        # each clique is a group of markings which all refer to the same region on the page
        # go through each clique
        for c in networkx.find_cliques(overlap_graph):
            # ignore any clique with less than 3 markings in it
            if len(c) < 0:
                continue

            # get the specific markings in this clique and their corresponding tools
            clique = [markings[i] for i in c]
            tools_in_clique = [tools[i] for i in c]

            # create the new cluster based on this clique
            new_cluster = dict()
            new_cluster["center"] = self.__median_rectangles__(clique)
            new_cluster["cluster members"] = clique
            new_cluster["users"] = [user_ids[i] for i in c]
            # the tools used by each person with a rectangle in this cluster
            new_cluster["tools"] = tools_in_clique
            new_cluster["image area"] = None

            results.append(new_cluster)
        return results,0
