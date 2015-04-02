__author__ = 'greg'
import bisect
import math
import ouroboros_api
import abc
import numpy as np
import os
import re
import ibcc
import csv
import matplotlib.pyplot as plt

def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError






class Cluster:
    __metaclass__ = abc.ABCMeta

    def __init__(self, project_api,min_cluster_size=1):
        """
        :param project_api: how to talk to whatever project we are clustering for (Panoptes/Ouroboros shouldn't matter)
        :param min_cluster_size: minimum number of points in a cluster to not be considered noise
        :return:
        """
        assert isinstance(project_api,ouroboros_api.MarkingProject)
        self.project_api = project_api
        self.min_cluster_size = min_cluster_size
        self.clusterResults = {}

        self.correct_pts = {}
        # for gold points which we have missed
        self.missed_pts = {}
        # for false positives (according to the gold standard)
        self.false_positives = {}
        # we also need to map between user points and gold points
        self.user_gold_mapping = {}

        # the distance between a user cluster and what we believe to do the corresponding goal standard point
        self.user_gold_distance = {}

        # the list of every subject we have processed so far - a set just in case we accidentally process the same
        # one twice. Probably should never happen but just in case
        self.processed_subjects = set([])

        # needed for when we want use ibcc for stuff
        current_directory = os.getcwd()
        slash_indices = [m.start() for m in re.finditer('/', current_directory)]
        self.base_directory = current_directory[:slash_indices[2]+1]

        # for creating names of ibcc output files
        self.alg = None



    def __display__markings__(self, subject_id):
        """
        display the results of clustering algorithm - the image must already have been downloaded
        :param subject_id:
        :param fname: the file name of the downloaded image
        :return:
        """
        x,y = zip(*self.clusterResults[subject_id][0])
        args = [x,y,'o']
        kwargs = {"color":"red"}

        ax = self.project_api.__display_image__(subject_id,[args],[kwargs])

    def __check_correctness__(self,subject_id):
        """
        display any and all pairs of users/gold standard points which we think are too far apart
        :param subject_id:
        :return:
        """
        args = []
        for u_pt,g_pt,dist in self.user_gold_distance[subject_id]:
            if dist > 5:
                print dist
                args.extend([[u_pt[0],g_pt[0]],[u_pt[1],g_pt[1]],"o-"])

        if args != []:
            ax = self.project_api.__display_image__(subject_id,[args],[{},])

    def __display_results__(self,subject_id):
        """
        plot found clusters, missed clusters and false positives clusters
        :param subject_id:
        :return:
        """
        # green is for correct points
        x,y = zip(*self.correct_pts[subject_id])
        args_l = [[x,y,'o'],]
        kwargs_l = [{"color":"green"},]

        # yellow is for missed points
        x,y = zip(*self.missed_pts[subject_id])
        args_l.append([x,y,'o'])
        kwargs_l.append({"color":"yellow"})

        # red is for false positives
        x,y = zip(*self.false_positives[subject_id])
        args_l.append([x,y,'o'])
        kwargs_l.append({"color":"red"})

        ax = self.project_api.__display_image__(subject_id,args_l,kwargs_l)

    @abc.abstractmethod
    def __fit__(self,markings,user_ids,jpeg_file=None):
        """
        the main function for clustering
        :param user_ids:
        :param markings:
        :param jpeg_file:
        :return cluster_centers: the center of each cluster - probably just take the average along each dimension
        feel free to try something else but the results might not mean as much
        :return markings_per_cluster: the markings in each cluster
        :return users_per_cluster: the user id of each marking per cluster
        :return time_to_cluster: how long it took to cluster
        """
        cluster_centers = []
        markings_per_cluster = []
        users_per_cluster = []
        time_to_cluster = 0

        return (cluster_centers , markings_per_cluster, users_per_cluster), time_to_cluster

    def __cluster_subject__(self,subject_id,jpeg_file=None):
        """
        the function to call from outside to do the clustering
        override but call if you want to add additional functionality
        :param subject_id: what is the subject (in Ouroboros == zooniverse_id)
        :param jpeg_file: for debugging - to show step by step what is happening
        :return:
        """
        # start by calling the api to get the annotations along with the list of who made each marking
        # so for this function, we know that annotations = markings
        users, markings = self.project_api.__get_markings__(subject_id)

        # if there are any markings - cluster
        # otherwise, just set to empty
        if markings != []:
            cluster_results,time_to_cluster = self.__fit__(markings,users,jpeg_file)
        else:
            cluster_results = [],[],[]
            time_to_cluster = 0

        self.clusterResults[subject_id] = cluster_results
        self.processed_subjects.add(subject_id)
        return time_to_cluster

    def __find_correct_markings__(self,subject_id):
        """
        find which user clusters we think are correct points
        :param subject_id:
        :return:
        """
        # get the markings made by the experts
        gold_markings = self.project_api.__get_markings__(subject_id,expert_markings=True)
        cluster_centers = self.clusterResults[subject_id][0]

        # init the set of correct markings, the missed ones, the false positives and the distance
        # between a correct marking and its corresponding gold standard pt - used for debugging
        self.correct_pts[subject_id] = []
        self.missed_pts[subject_id] = []
        self.user_gold_distance[subject_id] = []
        self.false_positives[subject_id] = []

        # if there are no gold markings, technically everything is a false positive
        if gold_markings == ([],[]):
            self.false_positives[subject_id] = cluster_centers
            return

        # so we know that there is at least one gold standard pt - but are there are any user clusters?
        # extract the actual points
        gold_pts = zip(*gold_markings[1][0])[0]

        # if there are no user markings, we have missed everything
        if cluster_centers == []:
            self.missed_pts = gold_pts
            return

        # we know that there are both gold standard points and user clusters - we need to match them up
        # user to gold - for a gold point X, what are the user points for which X is the closest gold point?
        userToGold = [[] for i in range(len(gold_pts))]

        # find which gold standard pts, the user cluster pts are closest to
        # this will tell us which gold points we have actually found
        for local_index, u_pt in enumerate(cluster_centers):
            # dist = [math.sqrt((float(pt["x"])-x)**2+(float(pt["y"])-y)**2) for g_pt in gold_pts]
            min_dist = float("inf")
            closest_gold_index = None

            # find the nearest gold point to the cluster center
            # doing this in a couple of lines so that things are simpler - need to allow
            # for an arbitrary number of dimensions
            for gold_index,g_pt in enumerate(gold_pts):
                dist = math.sqrt(sum([(u-g)**2 for (u,g) in zip(u_pt,g_pt)]))

                if dist < min_dist:
                    min_dist = dist
                    closest_gold_index = gold_index

            if min_dist < 20:
                userToGold[closest_gold_index].append(local_index)

        # and now find out which user clusters are actually correct
        # that will be the user point which is closest to the gold point
        distances_l =[]
        for gold_index,g_pt in enumerate(gold_pts):
            min_dist = float("inf")
            closest_user_index = None

            for u_index in userToGold[gold_index]:
                dist = math.sqrt(sum([(u-g)**2 for (u,g) in zip(cluster_centers[u_index],g_pt)]))

                if dist < min_dist:
                    min_dist = dist
                    closest_user_index = u_index

            # if none then we haven't found this point
            if closest_user_index is not None:
                u_pt = cluster_centers[closest_user_index]
                self.correct_pts[subject_id].append(u_pt)
                # todo: probably remove for production - only really useful for papers
                self.user_gold_distance[subject_id].append((u_pt,g_pt,min_dist))
                distances_l.append(min_dist)

                self.user_gold_mapping[(subject_id,tuple(u_pt))] = g_pt
            else:
                self.missed_pts[subject_id].append(g_pt)

        # print np.mean(distances_l)

        # what were the false positives?
        self.false_positives[subject_id] = [pt for pt in cluster_centers if not(pt in self.correct_pts[subject_id])]

    def __setup_gold_indices__(self):
        """
        if we want to use gold standard data for IBCC, we need to know which indices have gold standard data
        and also insert spots for gold standard points which all users missed
        """
        cluster_count = -1
        global_to_local = []
        global_gold_standard_indices = []

        # go through each subject and convert local indices into global ones
        for zooniverse_id in self.clusterResults:
            if not(self.clusterResults[zooniverse_id] is None):
                for local_index,center in enumerate(self.clusterResults[zooniverse_id][0]):
                    cluster_count += 1

                    # note that the cluster with index cluster_count is from subject zooniverse_id
                    global_to_local.append((zooniverse_id,local_index))
                    # does this result have a gold standard counterpart?
                    if (zooniverse_id,tuple(center)) in self.user_gold_mapping:
                        global_index = len(global_to_local)-1
                        global_gold_standard_indices.append(global_index)

            # include spots for animals which users missed
            for i in self.missed_pts[zooniverse_id]:
                cluster_count += 1
                global_to_local.append((zooniverse_id,None))

                # and add to the list
                global_index = len(global_to_local)-1
                global_gold_standard_indices.append(global_index)

        return global_to_local,global_gold_standard_indices


    def __signal_ibcc__(self,split_ip_address=True):
        """
        run ibcc to determine which clusters are signal or noise

        use majority voting to determine priors

        :param split_ip_address: for user ids which are ip addresses - when people are not logged in - should be
        treat each subject completely separate. That is if ip address X marked subjects A and B should we treat
        X as being two completely different people for those two classifications. There is no guarantee that they
        are the same person but seems like a lot of information to throw away. The param is to allow exploring the
        options and results.
        :return:
        """
        # todo: implement a middle ground for split_ip_address where we treat the same ip address as the same person
        # todo: as long as the classifications are close enough together time wise

        # get all users who have viewed any subjects which are processing - also get the list of those who
        # did so while not logged in
        all_users = list(self.project_api.__all_users__())
        all_ips = list(self.project_api.__all_ips__())

        # global cluster count - across all images/subjects
        cluster_count = -1

        # need to give the ip addresses unique indices, so update ip_offset after every subject
        ip_offset = 0

        # needed for determining priors for IBCC
        real_animals = 0
        fake_animals = 0

        # needed for prior confusion matrix
        true_pos = []
        true_neg = []

        # intermediate holder variable
        # because ibcc needs indices to be nice and ordered with no gaps, we have to make two passes through the data
        to_ibcc = []

        # for each global cluster index, store what image/subject it is from and what its local index is
        # wrt to that subject
        self.global_to_local = []

        # print out the classifications and set up the priors using majority voting
        for zooniverse_id in self.clusterResults:
            if self.clusterResults[zooniverse_id] is None:
                continue

            # get the list of all the users who viewed this subject
            # and the ip addresses of every user who was not logged in while viewing the subjects
            users_per_subject = self.project_api.__users__(zooniverse_id)
            ips_per_subject = self.project_api.__ips__(zooniverse_id)

            # process each cluster (possible animal), one at a time
            # only the names of users who marked this cluster matter - the specific x,y points are irrelevant right now
            for local_index,user_per_cluster in enumerate(self.clusterResults[zooniverse_id][2]):
                # moving on to the next animal so increase counter
                # universal counter over all images
                cluster_count += 1

                # needed for determining priors for IBCC
                pos = 0
                neg = 0

                # note that the cluster with index cluster_count is from subject zooniverse_id
                self.global_to_local.append((zooniverse_id,local_index))

                # for this cluster, go through each user and see if they marked this cluster
                # check whether or not each user marked this cluster
                for user_id in users_per_subject:
                    # if the user was not logged in
                    if user_id in ips_per_subject:
                        # if we are considering the ip addresses of each user (i.e. those that were not logged in)
                        # separately for each image - assign a user index based only this image
                        # use negative indices to differentiate ip addresses and users
                        # +1 assures that we don't have 0 - which is "both" positive and negative
                        if split_ip_address:
                            user_index = -(ips_per_subject.index(user_id)+ip_offset+1)
                        else:
                            # we are treating all occurances of this ip address as being from the same user
                            user_index = -all_ips.index(user_id)-1
                    else:
                        # user was logged in
                        # todo: use bisect to increase speed
                        user_index = all_users.index(user_id)

                    # did the user mark this cluster or not?
                    if user_id in user_per_cluster:
                        to_ibcc.append((user_id,user_index,cluster_count,1))
                        pos += 1
                    else:
                        to_ibcc.append((user_id,user_index,cluster_count,0))
                        neg += 1

                # if a majority of people say that there is an animal - use this for prior values
                if pos > neg:
                    real_animals += 1

                    # for estimating the confusion matrix
                    true_pos.append(pos/float(pos+neg))
                else:
                    fake_animals += 1

                    true_neg.append(neg/float(pos+neg))

            ip_offset += len(ips_per_subject)

        # now run through again - this will make sure that all of the indices are ordered with no gaps
        # since the user list is created by reading through all the users, even those which haven't annotated
        # of the specific images we are currently looking at
        ibcc_user_list = []

        # this is also for other functions to be able to interpret the results
        self.ibcc_users = []

        for user,user_index,animal_index,found in to_ibcc:
            # can't use bisect or the indices will be out of order
            if not(user_index in ibcc_user_list):
                ibcc_user_list.append(user_index)
                self.ibcc_users.append(user)

        # write out the input file for IBCC
        with open(self.base_directory+"/Databases/"+self.alg+"_ibcc.csv","wb") as f:
            f.write("a,b,c\n")
            for user,user_index,animal_index,found in to_ibcc:
                i = ibcc_user_list.index(user_index)
                f.write(str(i)+","+str(animal_index)+","+str(found)+"\n")

        # create the prior estimate and the default confusion matrix
        prior = real_animals/float(real_animals + fake_animals)

        t = np.mean(true_pos)
        f = np.mean(true_neg)
        # what the weight should be
        # todo: should this be hard coded or set as a param?
        weight = 10

        # the confusion matrix cannot have any zero values
        confusion = [[max(int(t*weight),1),max(int((1-t)*weight),1)],[max(int((1-f)*weight),1),max(int(f*weight),1)]]

        # create the config file
        with open(self.base_directory+"/Databases/"+self.alg+"_ibcc.py","wb") as f:
            f.write("import numpy as np\n")
            f.write("scores = np.array([0,1])\n")
            f.write("nScores = len(scores)\n")
            f.write("nClasses = 2\n")
            f.write("inputFile = \""+self.base_directory+"/Databases/"+self.alg+"_ibcc.csv\"\n")
            f.write("outputFile = \""+self.base_directory+"/Databases/"+self.alg+"_signal.out\"\n")
            f.write("confMatFile = \""+self.base_directory+"/Databases/"+self.alg+"_ibcc.mat\"\n")
            f.write("nu0 = np.array(["+str(max(int((1-prior)*100),1))+","+str(max(int(prior*100),1))+"])\n")
            f.write("alpha0 = np.array("+str(confusion)+")\n")

        # start by removing all temp files
        try:
            os.remove(self.base_directory+"/Databases/"+self.alg+"_signal.out")
        except OSError:
            pass

        try:
            os.remove(self.base_directory+"/Databases/"+self.alg+"_ibcc.mat")
        except OSError:
            pass

        try:
            os.remove(self.base_directory+"/Databases/"+self.alg+"_ibcc.csv.dat")
        except OSError:
            pass

        # pickle.dump((big_subjectList,big_userList),open(base_directory+"/Databases/tempOut.pickle","wb"))
        ibcc.runIbcc(self.base_directory+"/Databases/"+self.alg+"_ibcc.py")


    def __roc__(self,plot=False):
        """
        do a roc analysis of the ibcc results
        :param plot:
        :return:
        """
        # start by finding the correct markings for all of the images we have done
        for subject_id in self.clusterResults:
            self.__find_correct_markings__(subject_id)

        truePos = []
        falsePos = []

        with open(self.base_directory+"/Databases/"+self.alg+"_signal.out","rb") as f:
            reader = csv.reader(f,delimiter=" ")
            for r in reader:
                # t is 1-prob, so we can just ignore it t= temp or trash
                ii,t,prob = r
                # which subject does this cluster belong to and what is its local index?
                subject_id, local_index = self.global_to_local[int(float(ii))]
                center = self.clusterResults[subject_id][0][local_index]

                if center in self.correct_pts[subject_id]:
                    truePos.append(float(prob))
                else:
                    falsePos.append(float(prob))

        # create the ROC curve
        alphas = truePos[:]
        alphas.extend(falsePos)
        alphas.sort()
        X = []
        Y = []
        for a in alphas:
            X.append(len([x for x in falsePos if x >= a]))
            Y.append(len([y for y in truePos if y >= a]))


        plt.plot(X,Y)
        plt.xlabel("False Positive Count")
        plt.ylabel("True Positive Count")
        plt.show()