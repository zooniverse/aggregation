__author__ = 'ggdhines'
import pymongo
import abc
import cPickle as pickle
import os
import csv
import re
import matplotlib.pyplot as plt
import urllib
import matplotlib.cbook as cbook
import datetime
import warnings
import random
import clustering
import math
import numpy
import json
import itertools
import matplotlib.cm as cm

def findsubsets(S,m):
    return set(itertools.combinations(S, m))

class OuroborosAPI:
    __metaclass__ = abc.ABCMeta

    def __init__(self, project, date, experts=()):
        """
        :param project:
        :param date:
        :param experts: users to treat as experts - and skip over when we read stuff in normally
        :return:
        """
        self.project = project

        # connection to the MongoDB/ouroboros database
        client = pymongo.MongoClient()
        self.db = client[project+"_"+date]
        self.classification_collection = self.db[project+"_classifications"]
        self.subject_collection = self.db[project+"_subjects"]
        self.user_collection = self.db[project+"_users"]

        self.id_prefix = "subjects."

        self.experts = experts

        self.gold_standard_subjects = None

        # # use this to store classifications - only makes sense if we are using this data multiple times
        # # if you are just doing a straight run through of all the data, don't use this
        self.annotations = {}
        self.gold_annotations = {}

        current_directory = os.getcwd()
        slash_indices = [m.start() for m in re.finditer('/', current_directory)]
        self.base_directory = current_directory[:slash_indices[2]+1]

        # record which users viewed which subjects
        self.users_per_subject = {}
        # record which users were not logged in while viewing subjects
        # self.ips_per_subject = {}

        # all_ips is only really useful if we want to keep track on non-logged in users between subjects
        self.all_users = set()
        # self.all_ips = set()
        # used to keep track of the ip addresses of logged in users - so we can tell if there is ever
        # a case where users are only logged in some of the time
        self.user_ip_addresses = {}

    def __get_classifications__(self,subject_id,cluster_alg=None,gold_standard=False):
        pass

    def __evaluate__(self,candidates,ridings,results,clustering_alg=None):
        assert isinstance(ridings,dict)
        assert isinstance(results,dict)

        assert len(ridings) == len(results)

        errors = []
        percentage = []
        for subject_id in self.gold_standard_subjects:
            # print self.subject_collection.find_one({"zooniverse_id":subject_id})["location"]["standard"]
            if self.gold_annotations[subject_id] == []:
                continue

            # when none of the users have marked an image
            if not(subject_id in results):
                continue

            # map each cluster to its nearest gold standard point
            # None is when there are not enough markings in a cluster
            cluster_to_gold = []#None for i in clustering_alg.clusterResults[subject_id][0]]
            gold_to_cluster = []#[] for i in clusteringAlg.clusterResults[subject_id][0]]
            print clustering_alg.clusterResults[subject_id][0]
            print ridings[subject_id]

            # only go through  those clusters which have enough markings
            for cluster_center in ridings[subject_id]:
                # cluster_index = clustering_alg.clusterResults[subject_id][0].index(cluster_center)

                closest_distance = float("inf")
                closest_gold = None

                # go through each of the gold markings and determine the distance
                for gold_index,(gold_center,gold_classification) in enumerate(self.gold_annotations[subject_id][1][0]):
                    dist = math.sqrt((cluster_center[0]-gold_center[0])**2+(cluster_center[1]-gold_center[1])**2)

                    # if the distance is less then the smallest found so far, then we need to update
                    if dist < closest_distance:
                        closest_distance = dist
                        closest_gold = gold_index

                cluster_to_gold.append(closest_gold)

            # now map the gold standard markings to the user clusters
            # ie for each gold standard marking, find the closest user cluster
            # if the closest user cluster does not have enough users, then there is no mapping
            for gold_index,(gold_center,gold_classification) in enumerate(self.gold_annotations[subject_id][1][0]):
                closest_distance = float("inf")
                # closest_index = None
                closest_cluster = None

                # go though each of the user clusters and determine the distance
                for cluster_index,cluster_center in enumerate(clustering_alg.clusterResults[subject_id][0]):
                    dist = math.sqrt((cluster_center[0]-gold_center[0])**2+(cluster_center[1]-gold_center[1])**2)
                    if dist < closest_distance:
                        closest_distance = dist
                        # closest_index = cluster_index
                        closest_cluster = cluster_center

                # convert from the list of all clusters into the list of just those clusters which have enough users
                # if the closest cluster does not have enough  users, just set the value to none
                if closest_cluster in ridings[subject_id]:
                    gold_to_cluster.append(ridings[subject_id].index(closest_cluster))
                else:
                    gold_to_cluster.append(None)

            # print cluster_to_gold
            # print gold_to_cluster
            # print

            if (cluster_to_gold == []) or (gold_to_cluster == []):
                continue

            # assert len(clusteringAlg.clusterResults[subject_id][0]) == len(results[subject_id])
            for cluster_index,classification in enumerate(results[subject_id]):
                gold_index = cluster_to_gold[cluster_index]
                # we did not have enough markings for this cluster
                if gold_index is None:
                    continue

                assert isinstance(gold_index,int)

                # if this cluster does not match up with the gold standard
                if gold_to_cluster[gold_index] != cluster_index:
                    continue

                gold_pt,gold_classification = self.gold_annotations[subject_id][1][0][gold_index]
                # convert the gold standard label into an index
                candidate_index = candidates.index(gold_classification.lower())
                classification = results[subject_id][cluster_index]
                probability = classification[candidate_index]
                non_extreme_probability = max(min(probability,1-math.pow(10,-15)),math.pow(10,-15))

                individual_errors = -math.log(non_extreme_probability)
                errors.append(individual_errors)
                # else:
                #     errors.append(-math.log(1-math.pow(10,-15)))
                percentage.append(max(classification))

                if -math.log(non_extreme_probability) > 1:
                    print "***"
                    print subject_id
                    print gold_pt,gold_classification
                    # print -math.log(non_extreme_probability)
                    print candidates[classification.index(max(classification))]
                    print self.gold_annotations[subject_id]
                    individual_results = [candidates[r.index(max(r))] for r in results[subject_id]]
                    print zip(clustering_alg.clusterResults[subject_id][0],individual_results)
                    print max(results[subject_id][cluster_index]),probability
                    print "---=="
                    annotations_list = [item for sublist in self.annotations[subject_id][1] for item in sublist]

                    for pt in clustering_alg.clusterResults[subject_id][1][cluster_index]:
                        for ann in annotations_list:
                            if pt == ann[0]:
                                print ann
                    print
            continue

            for gold_pt,gold_classification in self.gold_annotations[subject_id][1][0]:
                closest_distance = float("inf")
                best_classification = None
                num_users = None
                best_index = None
                closest_center = None

                # print len(clusteringAlg.clusterResults[subject_id][0])
                # print len(results[subject_id])
                print "**"
                print  len(clusteringAlg.clusterResults[subject_id][0]), len(results[subject_id])
                assert len(clusteringAlg.clusterResults[subject_id][0]) == len(results[subject_id])
                for cluster_index in range(len(clusteringAlg.clusterResults[subject_id][0])):
                    center = clusteringAlg.clusterResults[subject_id][0][cluster_index]
                    classification = results[subject_id][cluster_index]
                    users = clusteringAlg.clusterResults[subject_id][2][cluster_index]
                # for center,users,classification in zip(clusteringAlg.clusterResults[subject_id][0],clusteringAlg.clusterResults[subject_id][2],results[subject_id]):
                    dist = math.sqrt((center[0]-gold_pt[0])**2+(center[1]-gold_pt[1])**2)
                    if dist < closest_distance:
                        closest_distance = dist
                        best_classification = classification
                        num_users = len(users)
                        best_index = cluster_index
                        closest_center = center

                # print (best_classification,gold_classification)
                if best_classification is not None:
                    gold_index = candidates.index(gold_classification.lower())
                    # if best_classification[0] != gold_classification:
                    non_extreme_probability = max(min(best_classification[gold_index],1-math.pow(10,-15)),math.pow(10,-15))


                    if -math.log(non_extreme_probability) > 5:
                        print "***"
                        print subject_id
                        print gold_pt,gold_classification
                        # print -math.log(non_extreme_probability)
                        print candidates[best_classification.index(max(best_classification))]
                        print self.gold_annotations[subject_id]
                        print clusteringAlg.clusterResults[subject_id][0]
                        print
                        print "---=="
                        annotations_list = [item for sublist in self.annotations[subject_id][1] for item in sublist]
                        for pt in clusteringAlg.clusterResults[subject_id][1][best_index]:
                            for ann in annotations_list:
                                if pt == ann[0]:
                                    print ann



                        # self.__display_image__(subject_id,[[[gold_pt[0],closest_center[0]],[gold_pt[1],closest_center[1]]]],[{"color":"black","linewidth":30}])

                    non_extreme_probability = min(max(0.01,non_extreme_probability),0.8)
                    individual_errors = -math.log(non_extreme_probability)
                    errors.append(individual_errors)
                    # else:
                    #     errors.append(-math.log(1-math.pow(10,-15)))
                    percentage.append(max(best_classification))
                    # print individual_errors,num_users

        print errors
        print numpy.mean(errors)

        return errors,percentage
            # if clusteringAlg is not None:
            #     assert isinstance(clusteringAlg,clustering.Cluster)
            #     for center,users,classification in zip(clusteringAlg.clusterResults[subject_id][0],clusteringAlg.clusterResults[subject_id][2],results[subject_id]):
            #
            #
            #         print center,users,classification
            # #     print clusteringAlg.clusterResults[subject_id][0]
            # # print results[subject_id]
            # print "----"
            # print self.gold_annotations[subject_id]
            # print

    # def __optimize_classification_collections__(self):
    #     """
    #     creates a new collection which has indices for the zooniverse_id of the subject
    #     :param gold_restrictions: if we only want a set of subjects for which we have gold standard data
    #     :return:
    #     """
    #     self.id_prefix = ""
    #     if "optimized_classifications" in self.db.collection_names():
    #         warnings.warn("The optimized version of the classifications collection already exists. You will need to manually delete this if you want to redo its creation.")
    #         self.classification_collection = self.db["optimized_classifications"]
    #         return
    #
    #     new_classification_collection = self.db["optimized_classifications"]
    #     new_classification_collection.create_index([("user_name",pymongo.ASCENDING),])
    #     new_classification_collection.create_index([("user_ip",pymongo.ASCENDING),])
    #     new_classification_collection.create_index([("zooniverse_id",pymongo.ASCENDING),])
    #
    #     for classification in self.classification_collection.find():
    #         post = dict()
    #         zooniverse_id = classification["subjects"][0]["zooniverse_id"]
    #
    #
    #         if (self.gold_standard_subjects != []) and (zooniverse_id not in self.gold_standard_subjects):
    #             continue
    #
    #         post["zooniverse_id"] = zooniverse_id
    #         try:
    #             post["user_name"] = classification["user_name"]
    #         except KeyError:
    #             pass
    #
    #         post["user_ip"] = classification["user_ip"]
    #         post["annotations"] = classification["annotations"]
    #
    #         post["subject_id"] = classification["subjects"][0]["id"]
    #
    #         new_classification_collection.insert(post)
    #
    #     self.classification_collection = self.db["optimized_classifications"]

    # def __classifications__per_gold_subject__(self):
    #     for subject_id in self.gold_standard_subjects:
    #         user_list,annotations_list = self.__get_annotations__(subject_id)

    def __users__(self,subject_id):
        """
        for a given subject_id return the set of all users - logged in or not - which viewed that subject
        if the user was not logged in, use the ip_address
        :param subject_id:
        :return:
        """
        return self.users_per_subject[subject_id]

    # def __ips__(self,subject_id):
    #     return self.ips_per_subject[subject_id]

    def __all_users__(self):
        return self.all_users

    # def __all_ips__(self):
    #     return self.all_ips

    def __display_image__(self,subject_id,args_l,kwargs_l,block=True,title=None):
        """
        return the file names for all the images associated with a given subject_id
        also download them if necessary
        :param subject_id:
        :return:
        """
        subject = self.subject_collection.find_one({"zooniverse_id": subject_id})
        url = subject["location"]["standard"]

        slash_index = url.rfind("/")
        object_id = url[slash_index+1:]

        if not(os.path.isfile(self.base_directory+"/Databases/"+self.project+"/images/"+object_id)):
            urllib.urlretrieve(url, self.base_directory+"/Databases/"+self.project+"/images/"+object_id)

        fname = self.base_directory+"/Databases/"+self.project+"/images/"+object_id

        image_file = cbook.get_sample_data(fname)
        image = plt.imread(image_file)

        fig, ax = plt.subplots()
        im = ax.imshow(image,cmap = cm.Greys_r)

        for args,kwargs in zip(args_l,kwargs_l):
            print args,kwargs
            ax.plot(*args,**kwargs)

        if title is not None:
            ax.set_title(title)
        plt.show(block=block)

    def __close_image__(self):
        plt.close()

    def __top_users__(self,num_users,overlap):
        """

        :param num_users:
        :param overlap:
        :return:
        """
        all_subjects = set()
        users = list(self.user_collection.find().sort("classification_count",-1).limit(10))
        users.append({"name":"yshish"})
        all = []
        self.experts = []
        for u in users:
            classifications = self.classification_collection.find({"user_name":u["name"],"tutorial":{"$ne":True}})
            subject_ids = [c["subjects"][0]["zooniverse_id"] for c in classifications]
            t = []

            all.append(set(subject_ids))
            self.experts.append(u["name"])
        print self.experts

        for subset in  findsubsets(range(len(self.experts)-1),2):
            old_len = len(list(all_subjects))
            # print subset
            s = all[-1]
            for i in subset:
                s = s.intersection(all[i])
            all_subjects = all_subjects.union(s)
            if len(list(all_subjects)) != old_len:
                # print len(list(all_subjects)), old_len
                print [self.experts[i] for i in subset], len(list(all_subjects)) - old_len
        # all_subjects = random.sample(all_subjects,100)

        t = []
        for zooniverse_id in list(all_subjects):
            subject = self.subject_collection.find_one({"zooniverse_id":zooniverse_id})
            if (subject["state"] == "complete") and (subject["metadata"]["retire_reason"] != "blank"):
                # all_subjects.remove(zooniverse_id)
                t.append(zooniverse_id)

        all_subjects = random.sample(t,400)
        for zooniverse_id in list(all_subjects):
            self.__store_annotations__(zooniverse_id,expert_markings=True)
        # print len(list(all_subjects))
        self.gold_standard_subjects = list(all_subjects)

        # print all

    def __set_subjects__(self,subject_ids):
        self.gold_standard_subjects = []
        for subject_id in subject_ids:
            self.gold_standard_subjects.append(subject_id)
            self.__store_annotations__(subject_id,expert_markings=True)

    def __gold_sample__(self,required_users,optional_users,max_subjects=float('inf')):
        base_classifications = list(self.classification_collection.find({"user_name": {"$in":required_users},"tutorial":{"$ne":True}},{"subjects.zooniverse_id":1}))
        print len(base_classifications)
        base_subjects = set([c["subjects"][0]["zooniverse_id"] for c in base_classifications])
        print len(base_subjects)
        other_subjects = {}
        for user in optional_users:
            other_classifications =  list(self.classification_collection.find({"user_name": user,"tutorial":{"$ne":True}}))
            other_subjects[user] = set([c["subjects"][0]["zooniverse_id"] for c in other_classifications])

        self.possible_subjects = set()

        for users in findsubsets(optional_users,2):
            subjects = base_subjects.copy()
            assert isinstance(subjects,set)
            for u in users:
                subjects = subjects.intersection(other_subjects[u])

            self.possible_subjects = self.possible_subjects.union(subjects)

        print len(self.possible_subjects)
        subject_ids = list(self.possible_subjects)
        random.shuffle(subject_ids)
        potential_subjects = []
        for subject_id in subject_ids:
            subject = self.subject_collection.find_one({"zooniverse_id":subject_id})
            if subject["classification_count"] >= 5:
                potential_subjects.append(subject_id)


        self.experts = required_users
        self.experts.extend(optional_users)
        self.gold_standard_subjects = []
        for subject_id in potential_subjects:
            self.__store_annotations__(subject_id)

            if len(self.annotations[subject_id][0]) >= 5:
                self.gold_standard_subjects.append(subject_id)
                self.__store_annotations__(subject_id,expert_markings=True)

            if len(self.gold_standard_subjects) >= max_subjects:
                break

        print len(self.gold_standard_subjects)


    def __random_gold_sample__(self,remove_blanks=True,max_subjects=float('inf'),maximum_number_gold_markings=float("inf"),minimum_number_gold_markings=0,minimum_users=0):
        """
        set up a list of subjects with gold standard data which we can use for testing

        NOTE - maximum_number_gold_markings and minimum_number_gold_markings only really make sense if the
        number of experts is 1, or if we are completely confident that all experts made the same markings

        :param remove_blanks: - do we want to discard blank images
        :param max_subjects: maximum number of subjects to return
        :param maximum_number_gold_markings: a maximum number of markings made by expert in a subject
        :param minimum_number_gold_markings: a minimum number of markings made by an expert
        :param minimum_users: minimum number of users to have see a subject
        :return:
        """

        # find all non-tutorial classifications made by experts - and randomly shuffle them
        expert_classifications = list(self.classification_collection.find({"user_name": {"$in": self.experts},"tutorial":{"$ne":True}}))
        random.shuffle(expert_classifications)

        self.gold_standard_subjects = []
        # expert_classifications = [{"zooniverse_id": id_} for id_ in [u'APK000185s', u'APK0000ndv', u'APK0000bu6', u'APK0001d8c', u'APK00012p5', u'APK00004gk', u'APK0000st7', u'APK0000u6z', u'APK000014u', u'APK00004a6', u'APK0000v1f', u'APK0000je7', u'APK0000yot', u'APK00011em', u'APK0001ed3', u'APK0000xiw', u'APK0000lrn', u'APK0000s3x', u'APK0002xk5', u'APK0000ppu', u'APK00004tu', u'APK00007n6', u'APK000003t', u'APK0000p5p', u'APK0000mx5', u'APK0001a7v', u'APK00009d7', u'APK0001bms', u'APK00018k1', u'APK0000r2d', u'APK0001bww', u'APK0000w2y', u'APK00011g6', u'APK0000day', u'APK0000q08', u'APK0000292', u'APK0000n0n', u'APK00004fc', u'APK0000c43', u'APK0000may', u'APK00011c7', u'APK0000rok', u'APK0000h87', u'APK00014d1', u'APK000150n', u'APK0000kb2', u'APK0000ji0', u'APK00018yv', u'APK0001b08', u'APK0000it7']]
        for classification in expert_classifications:

            # if we have reached the maximum desired number of subjects
            if len(self.gold_standard_subjects) == max_subjects:
                break

            try:
                zooniverse_id = classification["subjects"][0]["zooniverse_id"]
            except KeyError:
                zooniverse_id = classification["zooniverse_id"]

            # if we have provided a minimum or maximum number of markings
            if (maximum_number_gold_markings < float("inf")) or (minimum_number_gold_markings > 0):
                num_gold_markings =len(classification["annotations"][1]["value"])
                if num_gold_markings > maximum_number_gold_markings:
                    continue

                if num_gold_markings < minimum_number_gold_markings:
                    continue

            # we next filter based on some of the properties of the subject, not the classification
            subject = self.subject_collection.find_one({"zooniverse_id":zooniverse_id})
            # skip any subject which has not been retired
            if subject["state"] != "complete":
                continue

            # if we don't want any blank images
            if remove_blanks and (subject["metadata"]["retire_reason"] in ["blank"]):
                continue

            # if we haven't had enough users classify a subject
            # todo: don't count experts
            if subject["classification_count"] < minimum_users:
                continue

            # if we are reading in more than one expert's classifications, there is a chance we might
            # read in the same subject multiple times, make sure we only store it once
            if zooniverse_id not in self.gold_standard_subjects:
                print len(self.gold_standard_subjects)+1
                self.gold_standard_subjects.append(zooniverse_id)
                self.__store_annotations__(zooniverse_id,expert_markings=True)

        for subject_id in self.gold_standard_subjects:
            assert subject_id in self.gold_annotations
        print self.gold_standard_subjects


    def __get_completed_subjects__(self):
        """
        :return: the list of all subjects which been retired/completed
        """
        # todo: use the pickle directory to store results so we don't always reread in all of the db each time
        # todo: this will need to be dependent on the data given
        id_list = []

        find_conditions = {"state": "complete"}

        for subject in self.subject_collection.find(find_conditions):
            zooniverse_id = subject["zooniverse_id"]
            id_list.append(zooniverse_id)

        return id_list

    @abc.abstractmethod
    def __classification_to_annotations__(self,classification):
        """
        Convert a classification to an annotation - not ideal terminology but this is what we have in the
        Ouroboros MongoDB database. A classification is the result of one user viewing one subject. The annotation
        field of the classifications contains a lot of information that for aggregations (at least) is irrelevant -
        e.g. we don't need to know which browser the user was using.
        So for now, annotations means annotations which are specifically relevant to the aggregations. If you are
        trying to get other info such as how long it took the user to process a subject, that would probably be best
        done somewhere else. Maybe not completely efficient but will probably keep the sanity of the code.
        So for projects like Snapshot Serengeti the annotations will be just a "basic" classification (e.g.
        the user has a said the image contains a zebra). For Penguin Watch etc. the annotations will be a list of
        markings. Those markings should be filtered - i.e. scaled if necessary and checked to make sure that they
        fall into the ROI.
        """
        return []



    def __store_annotations__(self,zooniverse_id,max_users=float("inf"),expert_markings=False):
        """
        read through and return all of the relevant annotations associated with the given zooniverse_id
        :param zooniverse_id: id of the subject
        :param max_users: maximum number of classifications to read in
        :param expert_markings: do we want to read in markings from experts - either yes or no, shouldn't mix
        :return:
        """

        annotations_list = []
        user_list = []

        # print expert_markings

        # create a set of constraints for searching through the mongodb
        constraints = {"subjects.zooniverse_id":zooniverse_id}
        if expert_markings:
            constraints["user_name"] = {"$in": self.experts}
        else:
            constraints["user_name"] = {"$nin": self.experts}

        # print constraints

        # print zooniverse_id
        # subject = self.subject_collection.find_one({"zooniverse_id":zooniverse_id})
        # cutout = subject["metadata"]["cutout"]
        # roi = self.__get_roi__(zooniverse_id)

        for user_index, classification in enumerate(self.classification_collection.find(constraints)):
            # get the name of this user
            if "user_name" in classification:
                # add the _ just we know for certain whether a user was logged in
                # just in case a username is "10" for example, which is a valid ip address
                user_id = classification["user_name"]+"_"
            else:
                user_id = classification["user_ip"]

            # convert the classification into annotations/markings
            annotations = self.__classification_to_annotations__(classification)

            # if annotations != []:
            if user_id not in user_list:
                annotations_list.append(annotations)
                user_list.append(user_id)

            if len(user_list) == max_users:
                break

        assert len(user_list) == len(annotations_list)
        # print len(self.experts),len(annotations_list)
        # if expert_markings:
        #     if len(self.experts) != len(annotations_list):
        #
        #         for c in self.classification_collection.find(constraints):
        #             print c["created_at"]
        #             print json.dumps(c["annotations"], sort_keys=True,indent=4, separators=(',', ': '))
        #             print
        #     assert len(self.experts) == len(annotations_list)
        # store the classifications/annotations in the appropriate dictionary
        if expert_markings:
            self.gold_annotations[zooniverse_id] = (user_list,annotations_list)
        else:
            self.annotations[zooniverse_id] = (user_list,annotations_list)



class MarkingProject(OuroborosAPI):
    __metaclass__ = abc.ABCMeta

    def __init__(self, project, date, scale=1, experts=()):
        OuroborosAPI.__init__(self, project, date, experts=experts)
        # self.dimensions = dimensions
        self.scale = scale

        self.roi_dict = {}
        self.current_roi = None



    def __store_annotations__(self,zooniverse_id,max_users=float("inf"),expert_markings=False):
        """
        override the parent method so that we can apply ROIs
        read through and return all of the relevant annotations associated with the given zooniverse_id
        :param zooniverse_id: id of the subject
        :param max_users: maximum number of classifications to read in
        :param expert_markings: do we want to read in markings from experts - either yes or no, shouldn't mix
        :return:
        """
        if not(zooniverse_id in self.roi_dict):
            self.roi_dict[zooniverse_id] = self.__get_roi__(zooniverse_id)

        self.current_roi = self.roi_dict[zooniverse_id]
        OuroborosAPI.__store_annotations__(self,zooniverse_id,max_users,expert_markings)

        self.current_roi = None

    # @abc.abstractmethod
    # def __classification_to_markings__(self,classification,roi):
    #     """
    #     This is the main function projects will have to override - given a set of annotations, we need to return the list
    #     of all markings in that annotation
    #     """
    #     return []

    @abc.abstractmethod
    def __get_roi__(self,subject_id):
        pass

    def __get_markings__(self,subject_id,expert_markings=False):
        """
        just allows us to user different terminology so that we are clear about returning markings
        :param subject_id:
        :return:
        """
        if expert_markings:
            return self.gold_annotations[subject_id]
        else:
            return self.annotations[subject_id]

    def __get_classifications__(self,subject_id,cluster_alg=None,gold_standard=False):
        """
        Return a list of classifications for a given subject. For marking projects, we return a list of lists -
        each element in the "top" list corresponds to a different cluster (e.g. in Penguin Watch we return a list
        of classifications for different penguins)
        :param subject_id:
        :param cluster_alg:
        :param gold_standard:
        :return:
        """
        # for marking projects, a clustering algorithm must be provided
        assert isinstance(cluster_alg,clustering.Cluster)

        # l will be a list - each element will correspond to a different user and list all of the markings they made
        # along with their username/ip address
        # so for example - we could have [[], [((511.75, 237.5), (ip_address, u'doliolidWithoutTail'))]]
        # the blank indicates someone who saw the image but did not make any markings
        # the second element is for someone with ip_address, who made one marking
        # results is a 3 tuple - first is a list of the cluster centers, second is the a list, for every cluster
        # of the points in that cluster, finally a list of every user in a cluster
        if gold_standard:
            l = [[(m[0],(u,m[1])) for m in marking] for u,marking in zip(self.gold_annotations[subject_id][0],self.gold_annotations[subject_id][1])]
            results = cluster_alg.goldResults[subject_id]
        else:
            l = [[(m[0],(u,m[1])) for m in marking] for u,marking in zip(self.annotations[subject_id][0],self.annotations[subject_id][1])]
            results = cluster_alg.clusterResults[subject_id]

        # flatten that list
        l =  [item for sublist in l for item in sublist]

        if l == []:
            return [],[]

        # split points and classifications into two lists
        pts,classifications = zip(*l)

        classifications_per_cluster = []

        # iterate over the points making up each cluster
        for cluster in results[1]:
            classifications_per_cluster.append([])
            if cluster is None:
                continue
            for pt in cluster:
                # extract the user name and classification which correspond to this point
                # also add the point itself
                value = list(classifications[pts.index(pt)])
                value.append(pt)
                value = tuple(value)
                classifications_per_cluster[-1].append(value)

        return results[0],classifications_per_cluster

    # def __store_markings__(self,subject_id,max_users=None,expert_markings=False):
    #     OuroborosAPI.__store_annotations__(self,subject_id,max_users,expert_markings)

    # def __classification_to_annotations__(self,classification):
    #     annotations = classification["annotations"]
    #     markings = self.__annotations_to_markings__(annotations)
    #
    #     try:
    #         object_id = classification["subject_ids"][0]
    #     except KeyError:
    #         object_id = classification["subject_id"]
    #
    #     # go through the markings in reverse order and remove any that are outside of the ROI
    #     # also, scale as necessary
    #     assert isinstance(markings, list)
    #     for marking_index in range(len(markings)-1, -1, -1):
    #         marking = markings[marking_index]
    #
    #         # check to see if this marking is in the ROI
    #         if not(self.__in_roi__(object_id,marking)):
    #             markings.pop(marking_index)
    #
    #     return markings

    def __in_roi__(self,marking,lb_roi,ub_roi):
        """
        is this marking within the specific ROI
        even if there is specifically defined ROI (as in penguin watch), markings may occasionally
        be outside of the image (just plain error) so we need to exclude those
        :param marking:
        :param lb_roi: the marking must be above this multiline
        :param ub_roi: the marking must be below this multiline
        :return:

        """
        # todo: refactor a bit
        x,y = marking
        if (x < lb_roi[0][0]) or (x > lb_roi[-1][0]):
            return False

        # find the line segment that "surrounds" x and see if y is above that line segment (remember that
        # images are flipped)
        for segment_index in range(len(lb_roi)-1):
            if (lb_roi[segment_index][0] <= x) and (lb_roi[segment_index+1][0] >= x):
                rX1,rY1 = lb_roi[segment_index]
                rX2,rY2 = lb_roi[segment_index+1]

                m = (rY2-rY1)/float(rX2-rX1)
                rY = m*(x-rX1)+rY1

                if y >= rY:
                    # the point satisfies the lb_roi
                    break
                else:
                    return False

        if (x < ub_roi[0][0]) or (x > ub_roi[-1][0]):
            return False

        for segment_index in range(len(ub_roi)-1):
            if (ub_roi[segment_index][0] <= x) and (ub_roi[segment_index+1][0] >= x):
                rX1,rY1 = ub_roi[segment_index]
                rX2,rY2 = ub_roi[segment_index+1]

                m = (rY2-rY1)/float(rX2-rX1)
                rY = m*(x-rX1)+rY1

                if y <= rY:
                    # the point satisfies the lb_roi
                    return True
                else:
                    return False

        assert False


class PenguinWatch(MarkingProject):
    def __init__(self,date,roi_directory=None,pickle_directory="/tmp/"):
        """
        :param roi_directory:  the directory where roi.tsv is stored. if None we will try a standard directory
        :param date:
        :param to_pickle:
        :return:
        """
        MarkingProject.__init__(self, "penguin", date, ["x","y"],experts=["caitlin.black"],pickle_directory=pickle_directory)

        # init the ROI dictionary
        self.roi_dict = {}
        self.subject_to_site = {}
        # read in the ROI
        if roi_directory is None:
            roi_directory = self.base_directory + "github/Penguins/public/"

        with open(roi_directory+"roi.tsv","rb") as roi_file:
            roi_file.readline()
            reader = csv.reader(roi_file,delimiter="\t")
            for l in reader:
                path = l[0]
                t = [r.split(",") for r in l[1:] if r != ""]
                self.roi_dict[path] = [(int(x)/1.92,int(y)/1.92) for (x,y) in t]

    def __classification_to_annotations__(self,classification):
        """
        override so that we can read in which ROI we should use
        :param classification:
        :return:
        """
        # have we already found the ROI for this subject?
        try:
            object_id = classification["subject_ids"][0]
        except KeyError:
            object_id = classification["subject_id"]

        # if we haven't already read in this image find out what site it is from
        if not(object_id in self.subject_to_site):
            path = self.subject_collection.find_one({"_id":object_id})["metadata"]["path"]
            assert isinstance(path,unicode)
            slash_index = path.index("/")
            underscore_index = path.index("_")
            site_name = path[slash_index+1:underscore_index]

            # hard code some name changes in
            if site_name == "BOOTa2012a":
                site_name = "PCHAa2013"
            elif site_name == "BOOTb2013a":
                site_name = "PCHb2013"
            elif site_name == "DANCa2012a":
                site_namefit = "DANCa2013"
            elif site_name == "MAIVb2012a":
                site_name = "MAIVb2013"
            elif site_name == "NEKOa2012a":
                site_name = "NEKOa2013"
            elif site_name == "PETEa2013a":
                site_name = "PETEa2013a"
            elif site_name == "PETEa2013b":
                site_name = "PETEa2013a"
            elif site_name == "PETEb2012b":
                site_name = "PETEb2013"
            elif site_name == "SIGNa2013a":
                site_name = "SIGNa2013"

            if not(site_name in self.roi_dict.keys()):
                self.subject_to_site[object_id] = None
            else:
                self.subject_to_site[object_id] = site_name

        return MarkingProject.__classification_to_annotations__(self,classification)

    def __annotations_to_markings__(self,annotations):
        """
        find where markings are in the annotations and retrieve them
        :param annotations:
        :return: each marking is of two parts - the first are things we can cluster on - for example, x,y coordinates
        the second part is stuff we might want to use with classification - i.e. which animal species
        """
        for ann in annotations:
            if ("key" in ann) and (ann["key"] == "marking"):
                # try to extract and return the markings
                # if something is wrong with the list (usually shouldn't happen) - return an empty list
                try:
                    values = ann["value"].values()
                    return [((float(v["x"])*self.scale,float(v["y"])*self.scale),(v["value"],)) for v in values]
                except (AttributeError,KeyError,ValueError) as e:
                    return []

        # did not find any values corresponding to markings, so return an empty list
        return []

    def __display_image__(self,subject_id,args_l,kwargs_l,block=True,title=None):
        """
        overwrite so that we can display the ROI
        :param subject_id:
        :return:
        """
        # todo: add in displaying the ROI - if we want
        MarkingProject.__display_image__(self,subject_id,args_l,kwargs_l,block,title)



    def __in_roi__(self,object_id,marking):
        """
        check to see if the marking is within the roi - should be the case but sometimes weird things happen
        and users are able to give points outside of the roi.
        also - if there is no roi, then the whole image is valid
        :param object_id:
        :param marking:
        :return:
        """
        site = self.subject_to_site[object_id]
        if site is None:
            return True

        roi = self.roi_dict[site]

        x,y = marking[0]

        if y > 750:
            return False

        # find the line segment that "surrounds" x and see if y is above that line segment (remember that
        # images are flipped)
        for segment_index in range(len(roi)-1):
            if (roi[segment_index][0] <= x) and (roi[segment_index+1][0] >= x):
                rX1,rY1 = roi[segment_index]
                rX2,rY2 = roi[segment_index+1]

                m = (rY2-rY1)/float(rX2-rX1)
                rY = m*(x-rX1)+rY1

                if y >= rY:
                    # we have found a valid marking
                    # create a special type of animal None that is used when the animal type is missing
                    # thus, the marking will count towards not being noise but will not be used when determining the type

                    return True
                else:
                    return False

        # probably shouldn't happen too often but if it does, assume that we are outside of the ROI
        return False
