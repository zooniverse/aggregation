__author__ = 'greg'
import pymongo
import bisect
import sys
import os
import csv
import matplotlib.pyplot as plt
import urllib
import matplotlib.cbook as cbook
from collections import Iterator
import math
from scipy.stats.stats import pearsonr

from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# for Greg - which computer am I on?
if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"
sys.path.append(base_directory+"/github/pyIBCC/python")
import ibcc


def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError


class AnnotationIteration(Iterator):
    def __init__(self, x_max, y_max, scale=1):
        self.markings = None
        self.markingIndex = 0
        self.numMarkings = 0

        self.scale = scale

        self.x_min = 0
        self.y_min = 0
        self.x_max = x_max
        self.y_max = y_max

    def next(self):
        if self.markings is None:
            raise StopIteration

        # on the off chance that a marking is out of bounds, we should ignore it and move on
        while self.markingIndex < len(self.markings):
            # get that marking
            mark = self.markings[self.markingIndex]
            self.markingIndex += 1


            x = self.scale*float(mark["x"])
            y = self.scale*float(mark["y"])

            if (x >= self.x_min) and (x <= self.x_max) and (y >= self.y_min) and (x <= self.y_max) :
                # we have found a valid marking
                # create a special type of animal None that is used when the animal type is missing
                # thus, the marking will count towards not being noise but will not be used when determining the type
                if not("animal" in mark):
                    animal_type = None
                else:
                    animal_type = mark["animal"]

                return (x,y),animal_type

        raise StopIteration






class Aggregation:
    def __init__(self, project, date, ann_iterate=None, to_skip=[]):
        self.project = project

        client = pymongo.MongoClient()
        db = client[project+"_"+date]
        self.classification_collection = db[project+"_classifications"]
        self.subject_collection = db[project+"_subjects"]
        self.user_collection = db[project+"condor_users"]

        # we a global list of logged in users so we use the index for the same user over multiple images
        self.all_users = []

        # we need a list of of users per subject (and a separate one for just those users who were not logged in
        # those ones will just have ip addresses
        self.users_per_subject = {}
        self.ips_per_subject = {}

        # dictionaries for the raw markings per image
        self.markings_list = {}
        self.user_list = {}
        # what did the user think they saw these coordinates?
        # for example, in penguin watch, it could be a penguin
        self.what_list = {}

        # the clustering results per image
        self.clusterResults = {}

        self.signal_probability = []

        # this is the iteration class that goes through all of the markings associated with each classification
        # for at least some of the projects the above code will work just fine
        if ann_iterate is None:
            self.ann_iterate = AnnotationIteration
        else:
            self.ann_iterate = ann_iterate

        self.to_skip = to_skip

        # image dimensions - used to throw silly markings not actually on the image
        self.dimensions = {}

        self.closet_neighbours = {}

    def __readin_users__(self):
        for user_record in self.user_collection.find():
            if "name" in user_record:
                user = user_record["name"]

                bisect.insort(self.all_users,user)

    def __get_completed_subjects__(self):
        id_list = []
        for subject in self.subject_collection.find({"state": "complete"}):
            zooniverse_id = subject["zooniverse_id"]
            id_list.append(zooniverse_id)

            self.dimensions[zooniverse_id] = subject["metadata"]["original_size"]

        return id_list

    def __cluster_overlap__(self, c1, c2):
        return [c for c in c1 if c in c2]

    def __find_closest_neighbour__(self,zooniverse_id):
        cluster_results = self.clusterResults[zooniverse_id]
        # if there is only one cluster NN doesn't make sense
        if len(cluster_results[0]) == 1:
            return

        assert zooniverse_id in self.clusterResults
        self.closet_neighbours[zooniverse_id] = []



        assert len(cluster_results[0]) == len(cluster_results[1])
        assert len(cluster_results[1]) == len(cluster_results[2])



        for i1 in range(len(cluster_results[0])):
            center1,pts1,users1 = cluster_results[0][i1],cluster_results[1][i1],cluster_results[2][i1]

            minimum_distance = float("inf")
            overlap = None
            closest_neighbour = None
            for i2 in range(len(cluster_results[0])):
                if i1 != i2:
                    try:
                        center2,pts2,users2 = cluster_results[0][i2],cluster_results[1][i2],cluster_results[2][i2]
                    except IndexError:
                        print i2
                        print len(cluster_results[0])
                        print len(cluster_results[1])
                        print len(cluster_results[2])
                        raise

                    dist = math.sqrt((center1[0]-center2[0])**2+(center1[1]-center2[1])**2)
                    if dist < minimum_distance:
                        minimum_distance = dist
                        overlap = self.__cluster_overlap__(users1, users2)
                        closest_neighbour = center2[:]

            assert overlap is not None
            assert closest_neighbour is not None

            self.closet_neighbours[zooniverse_id].append((closest_neighbour, minimum_distance, len(overlap)))

    def __plot_closest_neighbours__(self,zooniverse_id_list):
        totalY = []
        totalDist = []

        for zooniverse_id in zooniverse_id_list:
            if zooniverse_id in self.closet_neighbours:
                pt_l,dist_l = zip(*self.closet_neighbours[zooniverse_id])
                X_pts,Y_pts = zip(*pt_l)

                # find to flip the image
                Y_pts = [-p for p in Y_pts]

                plt.plot(dist_l,Y_pts,'.',color="red")

                totalDist.extend(dist_l)
                totalY.extend(Y_pts)

        print pearsonr(dist_l,Y_pts)
        plt.show()

    def __find_one__(self,zooniverse_id_list):
        # for zooniverse_id in zooniverse_id_list:
        #     if zooniverse_id in self.closet_neighbours:
        #         self.__display_image__(zooniverse_id)
        #         for cluster_index in range(len(self.clusterResults[zooniverse_id][0])):
        #             center = self.clusterResults[zooniverse_id][0][cluster_index]
        #             nearest_neigbhour = self.closet_neighbours[zooniverse_id][cluster_index][0]
        #             overlap = self.closet_neighbours[zooniverse_id][cluster_index][2]
        #
        #             plt.plot([center[0],], [center[1],],'o',color="blue")
        #             if overlap == 1:
        #                 plt.plot([center[0],nearest_neigbhour[0]],[center[1],nearest_neigbhour[1]],color="red")
        #
        #         plt.show()
        # return


        #  sort of barnes interpolation
        totalY = []
        totalDist = []

        scale = 1

        Ypixel_range = np.arange(0,1,0.005)
        dist_range = np.arange(0,scale,0.01)
        X,Y = np.meshgrid(Ypixel_range,dist_range)

        Z = []

        #convert into one big list
        for zooniverse_id in zooniverse_id_list:
            if zooniverse_id in self.closet_neighbours:
                cluster_centers = self.clusterResults[zooniverse_id][0]
                X_pts,Y_pts = zip(*cluster_centers)

                distance_to_nn = zip(*self.closet_neighbours[zooniverse_id])[1]

                totalDist.extend(distance_to_nn)
                totalY.extend(Y_pts)

        #scale
        minY,maxY = min(totalY),max(totalY)
        minD,maxD = min(totalDist),max(totalDist)

        totalY = [scale*(y-minY)/(maxY-minY) for y in totalY]
        totalDist = [scale*(d-minD)/(maxD-minD) for d in totalDist]

        # now search for all of the clusters whose neighbour has one or zero users in common
        # convert into one big list
        to_show = []
        for zooniverse_id in zooniverse_id_list:
            if zooniverse_id in self.closet_neighbours:
                for cluster_index,center in enumerate(self.clusterResults[zooniverse_id][0]):
                    y_pixel = center[1]

                    # find out about the nearest neighbour
                    neighbour,dist,overlap_size = self.closet_neighbours[zooniverse_id][cluster_index]

                    if overlap_size == 1:
                        #normalize the y pixel height
                        y_pixel = scale*(y_pixel-minY)/(maxY-minY)
                        #normalize the distance
                        dist = scale*(dist-minD)/(maxD-minD)
                        z = sum([math.exp(-(Y-y_pixel)**2) for Y,d in zip(totalY,totalDist) if d <= dist])

                        z_max = sum([math.exp(-(Y-y_pixel)**2) for Y,d in zip(totalY,totalDist) if d <= scale])
                        to_show.append((z/z_max,zooniverse_id,center[:],neighbour[:]))

        to_show.sort(key = lambda x:x[0])
        for p,zooniverse_id,pt1,pt2 in to_show:
            self.__display_image__(zooniverse_id)
            plt.plot([pt1[0],pt2[0]],[pt1[1],pt2[1]],"o-",color="blue")
            plt.plot([pt1[0],],[pt1[1],],"o",color="red")
            plt.show()

    def __display_image__(self,zooniverse_id):
        subject = self.subject_collection.find_one({"zooniverse_id": zooniverse_id})
        url = subject["location"]["standard"]

        slash_index = url.rfind("/")
        object_id = url[slash_index+1:]

        if not(os.path.isfile(base_directory+"/Databases/"+self.project+"/images/"+object_id)):
            urllib.urlretrieve(url, base_directory+"/Databases/"+self.project+"/images/"+object_id)

        image_file = cbook.get_sample_data(base_directory+"/Databases/"+self.project+"/images/"+object_id)
        image = plt.imread(image_file)

        fig, ax = plt.subplots()
        im = ax.imshow(image)

    def __barnes_interpolation__(self,zooniverse_id_list):
        # sort of barnes interpolation
        totalY = []
        totalDist = []

        scale = 1

        Ypixel_range = np.arange(0,1,0.005)
        dist_range = np.arange(0,scale,0.01)
        X,Y = np.meshgrid(Ypixel_range,dist_range)

        Z = []

        #convert into one big list
        for zooniverse_id in zooniverse_id_list:
            if zooniverse_id in self.closet_neighbours:
                closest_neighbours = self.closet_neighbours[zooniverse_id]
                pt_l,pt_2,dist_l,overlap_size_l = zip(*closest_neighbours)
                X_pts,Y_pts = zip(*pt_l)

                totalDist.extend(dist_l)
                totalY.extend(Y_pts)

        #scale
        minY,maxY = min(totalY),max(totalY)
        minD,maxD = min(totalDist),max(totalDist)

        totalY = [scale*(y-minY)/(maxY-minY) for y in totalY]
        totalDist = [scale*(d-minD)/(maxD-minD) for d in totalDist]


        #now search for all of the clusters whose neighebour has one or zero users in common
        #convert into one big list
        P = []
        for zooniverse_id in zooniverse_id_list:
            if zooniverse_id in self.closet_neighbours:
                closest_neighbours = self.closet_neighbours[zooniverse_id]
                pt_l,pt_2,dist_l,overlap_size_l = zip(*closest_neighbours)
                for pts,dist,overlap_size in zip(pt_l,dist_l,overlap_size_l):
                    if overlap_size == 1:
                        y_pixel = scale*(pts[1]-minY)/(maxY-minY)
                        dist = scale*(dist-minD)/(maxD-minD)
                        z = sum([math.exp(-(Y-y_pixel)**2) for Y,d in zip(totalY,totalDist) if d <= dist])

                        z_max = sum([math.exp(-(Y-y_pixel)**2) for Y,d in zip(totalY,totalDist) if d <= 1])
                        P.append(z/z_max)
        print len(P)
        plt.hist(P,bins=20,normed=1,cumulative=True)
        plt.xlabel("Percentile for Clusters with |NN|=1")
        plt.show()



        # y_pixel = 0.1
        # Z = []
        # for dist in dist_range:
        #     Z.append(sum([math.exp(-(Y-y_pixel)**2) for Y,d in zip(totalY,totalDist) if d <= dist]))
        #
        # Z = [z/max(Z) for z in Z]
        # plt.plot(dist_range,Z)
        #
        # y_pixel = 0.9
        # Z = []
        # for dist in dist_range:
        #     Z.append(sum([math.exp(-(Y-y_pixel)**2) for Y,d in zip(totalY,totalDist) if d <= dist]))
        #
        # Z = [z/max(Z) for z in Z]
        # plt.plot(dist_range,Z)
        # plt.xlim((0,0.2))
        # plt.title("CDF of Nearest Neighbour Distance")
        # plt.legend(("Upper row", "Lower row"), "lower right")
        # plt.show()


    def __plot_cluster_size__(self,zooniverse_id_list):
        data = {}

        for zooniverse_id in zooniverse_id_list:
            if self.clusterResults[zooniverse_id] is not None:
                centers,pts,users = self.clusterResults[zooniverse_id]

                Y = [700-c[1] for c in centers]
                X = [len(p) for p in pts]

                plt.plot(X,Y,'.',color="blue")

                for x,y in zip(X,Y):
                    if not(x in data):
                        data[x] = [y]
                    else:
                        data[x].append(y)

        print pearsonr(X,Y)

        X = sorted(data.keys())
        Y = [np.mean(data[x]) for x in X]
        plt.plot(X,Y,'o-')
        plt.xlabel("Cluster Size")
        plt.ylabel("Height in Y-Pixels")
        plt.show()

    def __cluster_subject__(self,zooniverse_id,clustering_alg,correction_alg=None):
        assert zooniverse_id in self.markings_list

        if self.markings_list[zooniverse_id] != []:
            # cluster results will be a 3-tuple containing a list of the cluster centers, a list of the points in each
            # cluster and a list of the users who marked each point
            self.clusterResults[zooniverse_id] = clustering_alg(self.markings_list[zooniverse_id],self.user_list[zooniverse_id])

            # make sure we got a 3 tuple and since there was a least one marking, we should have at least one cluster
            # pruning will come later
            if not(len(self.clusterResults[zooniverse_id]) == 3):
                print self.clusterResults[zooniverse_id]
            assert len(self.clusterResults[zooniverse_id]) == 3
            assert self.clusterResults[zooniverse_id][0] != []

            # fix the cluster if desired
            if correction_alg is not None:
                self.clusterResults[zooniverse_id] = correction_alg(self.clusterResults[zooniverse_id])
        else:
            self.clusterResults[zooniverse_id] = None

        return self.clusterResults[zooniverse_id] is None

    def __signal_ibcc__(self):
        # run ibcc on each cluster to determine if it is a signal (an actual animal) or just noise
        # run ibcc on all of the subjects that have been processed (read in and clustered) so far
        # each cluster needs to have a universal index
        cluster_count = -1

        # need to give the ip addresses unique indices, so update ip_index after every subject
        ip_index = 0

        # needed for determining priors for IBCC
        real_animals = 0
        fake_animals = 0
        true_pos = 0
        false_neg = 0
        false_pos = 0
        true_neg = 0

        # intermediate holder variable
        # because ibcc needs indices to be nice and ordered with no gaps, we have to make two passes through the data
        to_ibcc = []

        for zooniverse_id in self.clusterResults:
            for cluster_center,cluster_markings,user_per_cluster in self.clusterResults[zooniverse_id]:
                # moving on to the next animal so increase counter
                # universal counter over all images
                cluster_count += 1

                # needed for determining priors for IBCC
                pos = 0
                neg = 0

                # check whether or not each user marked this cluster
                for u in self.users_per_subject[zooniverse_id]:
                    # was this user logged in or not?
                    # if not, their user name will be an ip address
                    try:
                        i = self.ips_per_subject[zooniverse_id].index(u) + ip_index


                        if u in user_per_cluster:
                            to_ibcc.append((-i,cluster_count,1))
                            pos += 1
                        else:
                            to_ibcc.append((-i,cluster_count,0))
                            neg += 1
                    # ifNone a ValueError was thrown, the user name was not in the list of ip addresses
                    # and therefore, the user name was not an ip address, which means the user was logged in
                    except ValueError as e:

                        if u in user_per_cluster:
                            to_ibcc.append((index(self.all_users,u),cluster_count,1))
                            pos += 1
                        else:
                            to_ibcc.append((index(self.all_users,u),cluster_count,0))
                            neg += 1

                if pos > neg:
                    real_animals += 1

                    true_pos += pos/float(pos+neg)
                    false_neg += neg/float(pos+neg)
                else:
                    fake_animals += 1

                    false_pos += pos/float(pos+neg)
                    true_neg += neg/float(pos+neg)

            ip_index += len(self.ips_per_subject[zooniverse_id])

        # now run through again - this will make sure that all of the indices are ordered with no gaps
        ibcc_user_list = []
        for u,animal_index,found in to_ibcc:
            # can't use bisect or the indices will be out of order
            if not(u in ibcc_user_list):
                ibcc_user_list.append(u)

        # write out the input file for IBCC
        with open(base_directory+"/Databases/condor_ibcc.csv","wb") as f:
            f.write("a,b,c\n")
            for u,animal_index,found in to_ibcc:
                i = ibcc_user_list.index(u)
                f.write(str(i)+","+str(animal_index)+","+str(found)+"\n")

        # create the prior estimate and the default confusion matrix
        prior = real_animals/float(real_animals + fake_animals)
        confusion = [[max(int(true_neg),1),max(int(false_pos),1)],[max(int(false_neg),1),max(int(true_pos),1)]]

        # create the config file
        with open(base_directory+"/Databases/"+self.project+"_ibcc.py","wb") as f:
            f.write("import numpy as np\n")
            f.write("scores = np.array([0,1])\n")
            f.write("nScores = len(scores)\n")
            f.write("nClasses = 2\n")
            f.write("inputFile = \""+base_directory+"/Databases/"+self.project+"_ibcc.csv\"\n")
            f.write("outputFile = \""+base_directory+"/Databases/"+self.project+"_signal.out\"\n")
            f.write("confMatFile = \""+base_directory+"/Databases/"+self.project+"_ibcc.mat\"\n")
            f.write("nu0 = np.array(["+str(max(int((1-prior)*100),1))+","+str(max(int(prior*100),1))+"])\n")
            f.write("alpha0 = np.array("+str(confusion)+")\n")

        # start by removing all temp files
        try:
            os.remove(base_directory+"/Databases/"+self.project+"_signal.out")
        except OSError:
            pass

        try:
            os.remove(base_directory+"/Databases/"+self.project+"_ibcc.mat")
        except OSError:
            pass

        try:
            os.remove(base_directory+"/Databases/"+self.project+"_ibcc.csv.dat")
        except OSError:
            pass

        # pickle.dump((big_subjectList,big_userList),open(base_directory+"/Databases/tempOut.pickle","wb"))
        ibcc.runIbcc(base_directory+"/Databases/"+self.project+"_ibcc.py")

    def __process_signal__(self):
        self.signal_probability = []

        with open(base_directory+"/Databases/"+self.project+"_signal.out","rb") as f:
            results = csv.reader(f, delimiter=' ')

            for row in results:
                self.signal_probability.append(float(row[2]))

    def __get_subjects_per_site__(self,zooniverse_id):
        pass

    def __display__markings__(self, zooniverse_id):
        assert zooniverse_id in self.clusterResults
        subject = self.subject_collection.find_one({"zooniverse_id": zooniverse_id})
        zooniverse_id = subject["zooniverse_id"]
        print zooniverse_id
        url = subject["location"]["standard"]

        slash_index = url.rfind("/")
        object_id = url[slash_index+1:]

        if not(os.path.isfile(base_directory+"/Databases/"+self.project+"/images/"+object_id)):
            urllib.urlretrieve(url, base_directory+"/Databases/"+self.project+"/images/"+object_id)

        image_file = cbook.get_sample_data(base_directory+"/Databases/"+self.project+"/images/"+object_id)
        image = plt.imread(image_file)

        fig, ax = plt.subplots()
        im = ax.imshow(image)

        for (x, y), pts, users in zip(*self.clusterResults[zooniverse_id]):
            plt.plot([x, ], [y, ], 'o', color="red")


        plt.show()
        plt.close()

    def __display_signal_noise(self):
        for ii,zooniverse_id in enumerate(self.clusterResults):
            print zooniverse_id
            subject = self.subject_collection.find_one({"zooniverse_id":zooniverse_id})
            zooniverse_id = subject["zooniverse_id"]
            url = subject["location"]["standard"]

            slash_index = url.rfind("/")
            object_id = url[slash_index+1:]

            if not(os.path.isfile(base_directory+"/Databases/condors/images/"+object_id)):
                urllib.urlretrieve (url, base_directory+"/Databases/condors/images/"+object_id)

            image_file = cbook.get_sample_data(base_directory+"/Databases/condors/images/"+object_id)
            image = plt.imread(image_file)

            fig, ax = plt.subplots()
            im = ax.imshow(image)

            for center,animal_index,users_l,user_count in results_dict[zooniverse_id]:
                if ibcc_v[animal_index] >= 0.5:
                    print center[0],center[1],1
                    plt.plot([center[0],],[center[1],],'o',color="green")
                else:
                    print center[0],center[1],0
                    plt.plot([center[0],],[center[1],],'o',color="red")

            plt.show()
            plt.close()

    def __readin_subject__(self, zooniverse_id):
        # records relating to the individual annotations
        # first - the actual XY markings, then what species are associated with the annotations,
        # then who made each marking
        self.markings_list[zooniverse_id] = []
        self.user_list[zooniverse_id] = []
        self.what_list[zooniverse_id] = []

        # keep track of which users annotated this. Users per subject will contain all users - ip_addresses
        # will contain just those users who were not logged in so we only have the ip address to identify them
        # we need to deal with non-logged in users slightly differently
        self.ips_per_subject[zooniverse_id] = []
        self.users_per_subject[zooniverse_id] = []


        x_max = self.dimensions[zooniverse_id]["width"]
        y_max = self.dimensions[zooniverse_id]["height"]


        for user_index, classification in enumerate(self.classification_collection.find({"subjects.zooniverse_id":zooniverse_id})):
            # get the name of this user
            if "user_name" in classification:
                user = classification["user_name"]
            else:
                user = classification["user_ip"]
                self.ips_per_subject[zooniverse_id].append(user)

            # check to see if we have already encountered this subject/user pairing
            # due to some double classification errors
            if user in self.users_per_subject[zooniverse_id]:
                continue
            self.users_per_subject[zooniverse_id].append(user)


            # read in all of the markings this user made - which might be none
            for pt, animal_type in self.ann_iterate(classification,x_max,y_max):
                if not(animal_type in self.to_skip):
                    self.markings_list[zooniverse_id].append(pt)
                    # print annotation_list
                    self.user_list[zooniverse_id].append(user)
                    self.what_list[zooniverse_id].append(animal_type)

