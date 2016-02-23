import clustering
import networkx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

class RectangleClustering(clustering.Cluster):
    def __init__(self,shape,project,additional_params):
        assert shape != "point"
        clustering.Cluster.__init__(self,shape,project,additional_params)
        self.rectangle = (shape == "rectangle") or (shape == "image")

    def __overlap__(self,l1,l2):
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

    def __cluster__(self,markings,user_ids,tools,reduced_markings,dimensions,subject_id):
        if markings == []:
            return [],0

        g = networkx.Graph()
        g.add_nodes_from(range(len(markings)))

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

        # each clique is a group of markings which all refer to the same region on the page
        cliques = list(networkx.find_cliques(g))

        results = []

        for c in cliques:
            if len(c) < 3:
                continue
            # found = True
            # don't assume that all rectangles will be in the same order
            maximum_x = [max(markings[i][0][0],markings[i][2][0]) for i in c]
            minimum_x = [min(markings[i][0][0],markings[i][2][0]) for i in c]

            maximum_y = [max(markings[i][0][1],markings[i][2][1]) for i in c]
            minimum_y = [min(markings[i][0][1],markings[i][2][1]) for i in c]

            x_top = np.median(maximum_x)
            x_bot = np.median(minimum_x)
            y_top = np.median(maximum_y)
            y_bot = np.median(minimum_y)

            new_cluster = dict()
            new_cluster["center"] = [(x_top,y_top),(x_bot,y_bot)]

            new_cluster["cluster members"] = [markings[i] for i in c]
            new_cluster["users"] = [user_ids[i] for i in c]
            # todo - implement
            new_cluster["tools"] = None
            # todo - implement this - maybe based on voting weighted by fraction of rectangle inside aggregate?
            new_cluster["tool_classification"] = None
            new_cluster["image area"] = None

            results.append(new_cluster)

        return results,0