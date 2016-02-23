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
        # networkx.draw(g)
        cliques = list(networkx.find_cliques(g))

        found = False

        # image_fname = self.project.__image_setup__(subject_id)
        #
        # image_file = cbook.get_sample_data(image_fname[0])
        # image = plt.imread(image_file)
        # fig, ax = plt.subplots()
        # im = ax.imshow(image)

        for c in cliques:
            if len(c) < 3:
                for i in c:
                    a,b = markings[i][0]
                    c,d = markings[i][2]

                    x = [a,c,c,a,a]
                    y = [b,b,d,d,b]

                    plt.plot(x,y,color="red")
                continue
            # found = True
            # don't assume that all rectangles will be in the same order
            maximum_x = [max(markings[i][0][0],markings[i][2][0]) for i in c]
            minimum_x = [min(markings[i][0][0],markings[i][2][0]) for i in c]

            maximum_y = [max(markings[i][0][1],markings[i][2][1]) for i in c]
            minimum_y = [min(markings[i][0][1],markings[i][2][1]) for i in c]

            for i in c:
                a,b = markings[i][0]
                c,d = markings[i][2]

                x = [a,c,c,a,a]
                y = [b,b,d,d,b]

                plt.plot(x,y,color="green")

            x_top = np.median(maximum_x)
            x_bot = np.median(minimum_x)
            y_top = np.median(maximum_y)
            y_bot = np.median(minimum_y)

            # x_values = [x_top,x_bot,x_bot,x_top,x_top]
            # y_values = [y_top,y_top,y_bot,y_bot,y_top]
            # plt.plot(x_values,y_values,color="blue")

        # if found:
        #     plt.show()
        # else:
        #     plt.close()
        return [],0