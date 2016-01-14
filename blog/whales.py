__author__ = 'ggdhines'
from aggregation_api import AggregationAPI
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.spatial import ConvexHull
from shapely.ops import cascaded_union, polygonize
import shapely.geometry as geometry
from scipy.spatial import Delaunay
import numpy as np
import math
import shapely

from descartes import PolygonPatch

def plot_polygon(ax,polygon):
    # fig = plt.figure(figsize=(10,10))
    # ax = fig.add_subplot(111)
    # margin = .3
    # x_min, y_min, x_max, y_max = polygon.bounds
    # ax.set_xlim([x_min-margin, x_max+margin])
    # ax.set_ylim([y_min-margin, y_max+margin])
    patch = PolygonPatch(polygon, fc='#999999',
                         ec='#000000', fill=True,
                         zorder=-1)
    ax.add_patch(patch)
    return

def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])

    coords = np.array([point.coords[0] for point in points])
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)
        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points

with AggregationAPI(11,"development") as whales:
    whales.__setup__()

    postgres_cursor = whales.postgres_session.cursor()
    select = "SELECT classification_subjects.subject_id,annotations from classifications INNER JOIN classification_subjects ON classification_subjects.classification_id = classifications.id where workflow_id = 84"
    postgres_cursor.execute(select)

    for subject_id,annotations in postgres_cursor.fetchall():
        f_name = whales.__image_setup__(subject_id)

        image_file = cbook.get_sample_data(f_name[0])
        image = plt.imread(image_file)

        fig, ax1 = plt.subplots(1, 1)
        ax1.imshow(image)
        plt.show()

        inds_0 = image[:,:,0] >= 175
        inds_1 = image[:,:,1] >= 175
        inds_2 = image[:,:,2] >= 175
        inds_white = inds_0 & inds_1 & inds_2

        inds = image[:,:,2] >= 50
        image[inds] = [255,255,255]
        # image[inds_white] = [0,0,0]


        # imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # ret,thresh = cv2.threshold(imgray,127,255,0)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (thresh, im_bw) = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        im_bw = cv2.threshold(gray_image, thresh, 255, cv2.THRESH_BINARY)[1]

        fig, ax1 = plt.subplots(1, 1)
        ax1.imshow(im_bw)
        plt.show()



        fig, ax1 = plt.subplots(1, 1)
        image_file = cbook.get_sample_data(f_name[0])
        image = plt.imread(image_file)
        ax1.imshow(image)

        # edges = cv2.Canny(image,50,400)

        im2, contours, hierarchy = cv2.findContours(im_bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)


        for ii,cnt in enumerate(contours):
            if cnt.shape[0] > 20:
                cnt = np.reshape(cnt,(cnt.shape[0],cnt.shape[2]))
                cnt_list = cnt.tolist()
                X,Y = zip(*cnt_list)
                # plt.plot(X,Y)
                hull = ConvexHull(cnt)
                # plt.plot(cnt[hull.vertices,0], cnt[hull.vertices,1], 'r--', lw=2)

                shapely_points = [shapely.geometry.shape({"type": "Point", "coordinates": (x,y)}) for (x,y) in zip(X,Y)]
                concave_hull, edge_points = alpha_shape(shapely_points,alpha=0.01)

                # print edge_points

                if isinstance(concave_hull,shapely.geometry.Polygon):
                    # plot_polygon(ax1,concave_hull)
                    X,Y =  zip(*list(concave_hull.exterior.coords))
                    plt.plot(X,Y)
                else:

                    for p in concave_hull:
                        X,Y =  zip(*list(p.exterior.coords))
                        plt.plot(X,Y)
                # else:
                #     for p in concave_hull:
                #         plot_polygon(ax1,p)


                # hull_y = [Y[simplex[0]] for simplex in hull.simplices]
                # plt.plot(hull_x,hull_y)
                # if cv2.contourArea(cnt) > 0:
                #     print cv2.contourArea(cnt)
                #     cv2.drawContours(image, contours, ii, (0,255,0), 3)

        plt.ylim((image.shape[0],0))
        plt.xlim((0,image.shape[1]))
        plt.show()
