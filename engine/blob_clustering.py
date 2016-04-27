from __future__ import print_function

import clustering
import matplotlib.pyplot as plt
import itertools
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import stats


class EmptyPolygon(Exception):
    pass


class BlobClustering(clustering.Cluster):
    def __init__(self,shape,project,additional_params):
        assert shape != "point"
        clustering.Cluster.__init__(self,shape,project,additional_params)
        self.rectangle = (shape == "rectangle") or (shape == "image")

    def __find_positive_regions__(self,user_ids,markings,dimensions):
        """
        give a set of polygon markings made by people, determine the area(s) in the image which were outlined
        by enough people. "positive" => true positive as opposed to noise or false positive
        """
        unique_users = set(user_ids)

        aggregate_polygon_list = []

        for i in unique_users:
            user_polygons = [markings[j] for j,u in enumerate(user_ids) if u == i]

            template = np.zeros(dimensions,np.uint8)

            # start by drawing the outline of the area
            cv2.polylines(template,user_polygons,True,255)

            # now take the EXTERNAL contour
            # the docker image has an older version of opencv where findcontours only returns 2 values
            if cv2.__version__ == '2.4.8':
                contours, hierarchy = cv2.findContours(template,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            else:
                im2, contours, hierarchy = cv2.findContours(template,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            template2 = np.zeros(dimensions,np.uint8)
            cv2.drawContours(template2,contours,-1,1,-1)

            aggregate_polygon_list.append(template2)

        aggregate_polygon = np.sum(aggregate_polygon_list,axis=0,dtype=np.uint8)

        # the threshold determines the minimum number of people who have outlined an area
        threshold = int(0.25*len(set(user_ids)))
        ret,thresh1 = cv2.threshold(aggregate_polygon,threshold,255,cv2.THRESH_BINARY)

        return thresh1

    def __most_common_tool_array__(self,markings,user_ids,tools,dimensions):
        """
        return an array the same size as the original image where the value of each pixel corresponds to 1+ the
        most common tool used to select that pixel. 0 => no one selected that pixel
        :return:
        """
        # now go through on a tool by tool basis
        # this time, the brightness of the polygons corresponds to the tools used
        polygons_by_tools = []

        # do it user by user - so if a user outlines a region multiple times, their vote will only count once
        # if a user outlines a region using two different tools not really not sure what will happen there
        unique_users = set(user_ids)

        for i in unique_users:
            polygons_by_user = [j for j, u in enumerate(user_ids) if u == i]
            # convert the polygons into numpy arrays - makes opencv happy
            user_polygons = [markings[j] for j in polygons_by_user]

            # find the tool used to create each of these tools
            tools_per_polygons = [tools[j] for j in polygons_by_user]

            # this is where we will draw each of the polygons

            template2 = np.zeros(dimensions, np.uint8)
            for poly, t in zip(user_polygons, tools_per_polygons):
                # start by drawing the outline of the area
                template = np.zeros(dimensions, np.uint8)
                cv2.polylines(template, [poly], True, 255)
                # now take the EXTERNAL contour
                # the docker image has an older version of opencv where findcontours only returns 2 values
                if cv2.__version__ == '2.4.8':
                    contours, hierarchy = cv2.findContours(template,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                else:
                    im2, contours, hierarchy = cv2.findContours(template,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

                # +1 since I don't know if the tools are 1 indexed or 0 indexed
                # if 0-indexed, this might get confused with black - so just playing it safe
                cv2.drawContours(template2, contours, -1, t + 1, -1)

            polygons_by_tools.append(template2)

        # find the most common tool used to outline each pixel
        # most_common_tool = stats.mode(polygons_by_tools)[0][0]
        values,counts = np.unique(polygons_by_tools,return_counts=True)
        values_and_counts = zip(values,counts)
        # sort by the number of occurances
        values_and_counts.sort(reverse=True,key= lambda x:x[1])

        # if the most common tool was "nothing" return the second most common tool
        if values_and_counts[0][0] == 0:
            return values_and_counts[1][0]
        else:
            return values_and_counts[0][0]

    def __convert_to_numpy__(self,markings):
        """
        opencv needs the markings to be in numpy array format. We need this format several times, so it makes sense
        to do it once here
        :param markings:
        :return:
        """
        return [np.asarray(m,np.int) for m in markings]

    def __get_dimensions__(self,markings):
        """
        if image dimensions are not provided, use the markings to determine the template size
        :param markings:
        :return:
        """
        # images often swap vertical with horizontal and I right now I can't remember which is which
        # so to place it safe, I'll just set both dimensions to the maximum value
        a,b = zip(*([np.max(m,axis=0) for m in markings]))
        return max(max(a),max(b))+1,max(max(a),max(b))+1


    def __cluster__(self,markings,user_ids,tools,reduced_markings,dimensions,subject_id):
        """
        do polygon clustering looking for regions which have been highlighted/selected/outlined by enough people
        :param markings:
        :param user_ids:
        :param tools:
        :param reduced_markings:
        :param dimensions:
        :param subject_id:
        :return:
        """
        # start by converting to numpy array
        markings = self.__convert_to_numpy__(markings)

        if dimensions == (None,None):
            dimensions = self.__get_dimensions__(markings)

        positive_area = self.__find_positive_regions__(user_ids,markings,dimensions)

        most_common_tool = self.__most_common_tool_array__(markings,user_ids,tools,dimensions)

        clusters = []

        threshold = int(0.25 * len(set(user_ids)))

        for tool_index in sorted(set(tools)):
            area_by_tool = np.where((most_common_tool==(tool_index+1)) & (positive_area > 0))

            template = np.zeros(dimensions,np.uint8)
            template[area_by_tool] = 255

            # finally extract each (possible seperate) area which has been outlined by enough people
            # and is of the right tool
            # the docker image has an older version of opencv where findcontours only returns 2 values
            if cv2.__version__ == '2.4.8':
                contours, hierarchy = cv2.findContours(template,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            else:
                im2, contours, hierarchy = cv2.findContours(template,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                try:
                    clusters.append(self.__new_cluster__(cnt,tool_index,threshold,dimensions))
                except EmptyPolygon:
                    pass

        return clusters,0

    def __new_cluster__(self,polygon,tool_index,threshold,dimensions):
        """
        create the dictionary entry representing the new cluster
        :return:
        """
        s = polygon.shape
        cluster = {}
        cluster["center"] = polygon.reshape((s[0],s[2])).tolist()
        cluster["area"] = cv2.contourArea(polygon)/(dimensions[0]*dimensions[1])
        if cluster["area"] <= 0.001:
            raise EmptyPolygon()
        # todo - remember why i needed both tool_classification and most_likely_tool
        cluster["tool_classification"] = ({tool_index-1:1},-1)
        cluster["most_likely_tool"] = tool_index
        cluster["users"] = range(threshold)

        return cluster
