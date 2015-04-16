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

class OuroborosAPI:
    __metaclass__ = abc.ABCMeta

    def __init__(self, project, date, experts=[],pickle_directory="/tmp/"):
        """
        :param project:
        :param date:
        :param experts: users to treat as experts - and skip over when we read stuff in normally
        :param to_pickle: should results be saved for easy future use?
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

        self.pickle_directory = pickle_directory
        self.experts = experts

        self.gold_standard_subjects = None

        # use this to store classifications - only makes sense if we are using this data multiple times
        # if you are just doing a straight run through of all the data, don't use this
        self.classifications = {}

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

    def __optimize_classification_collections__(self,with_experts_only=False):
        """
        creates a new collection which has indices for the zooniverse_id of the subject
        :return:
        """
        self.id_prefix = ""
        if "optimized_classifications" in self.db.collection_names():
            warnings.warn("The optimized version of the classifications collection already exists. You will need to manually delete this if you want to redo its creation.")
            self.classification_collection = self.db["optimized_classifications"]
            return

        new_classification_collection = self.db["optimized_classifications"]
        new_classification_collection.create_index([("user_name",pymongo.ASCENDING),])
        new_classification_collection.create_index([("user_ip",pymongo.ASCENDING),])
        new_classification_collection.create_index([("zooniverse_id",pymongo.ASCENDING),])

        if with_experts_only:
            gold_subjects = list(self.__get_subjects_with_gold_standard__())
        else:
            gold_subjects = None

        for classification in self.classification_collection.find():
            post = dict()
            zooniverse_id = classification["subjects"][0]["zooniverse_id"]


            if with_experts_only and (zooniverse_id not in gold_subjects):
                continue

            post["zooniverse_id"] = zooniverse_id
            try:
                post["user_name"] = classification["user_name"]
            except KeyError:
                pass

            post["user_ip"] = classification["user_ip"]
            post["annotations"] = classification["annotations"]

            post["subject_id"] = classification["subjects"][0]["id"]

            new_classification_collection.insert(post)

        self.classification_collection = self.db["optimized_classifications"]


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
        im = ax.imshow(image)

        for args,kwargs in zip(args_l,kwargs_l):
            ax.plot(*args,**kwargs)

        if title is not None:
            ax.set_title(title)
        plt.show(block=block)

    def __close_image__(self):
        plt.close()

    def __get_subjects_with_gold_standard__(self,require_completed=True,remove_blanks=True,limit=-1,maximum_number_gold_markings=None):
        """
        find the zooniverse ids of all the subjects with gold standard data. Allows for more than one expert
        :param require_completed: return only those subjects which have been retired/completed
        :param remove_blanks: discard any subjects which were retired due to being blank
        :param limit - we want a limit on the number of subjects found
        :return:
        """
        # todo: use the pickle directory
        # todo: not sure if remove blanks is working properly
        # if we have provided a limit, use it
        classification_iterator = list(self.classification_collection.find({"user_name": {"$in": self.experts}}))
        random.shuffle(classification_iterator)
        # if limit != -1:
        #     classification_iterator = classification_iterator.limit(limit)
        subjects = set()
        count = 0
        for classification in classification_iterator:
            if (limit > 0) and (count >= limit):
                break

            try:
                zooniverse_id = classification["subjects"][0]["zooniverse_id"]
            except KeyError:
                zooniverse_id = classification["zooniverse_id"]

            if zooniverse_id in ["APZ00017op"]:
                continue

            try:
                num_gold_markings =len(classification["annotations"][1]["value"])
                if (maximum_number_gold_markings is not None) and (num_gold_markings > maximum_number_gold_markings):
                    continue
            except KeyError:
                continue
            # if we have additional constraints on the subject

            if require_completed or remove_blanks:
                # retrieve the subject first and then check conditions - this way we avoid having to search
                # through the whole DB
                subject = self.subject_collection.find_one({"zooniverse_id":zooniverse_id})

                if require_completed:
                    if subject["state"] != "complete":
                        print "a"
                        continue
                if remove_blanks:
                    if subject["metadata"]["retire_reason"] in ["blank"]:
                        print "b"
                        continue

            if zooniverse_id not in subjects:
                subjects.add(zooniverse_id)
                count += 1
                yield zooniverse_id


        # self.gold_standard_subjects = list(subjects)
        #
        # return self.gold_standard_subjects

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

    def __get_annotations__(self,zooniverse_id,max_users=None,expert_markings=False):
        """
        read through and return all of the relevant annotations associated with the given zooniverse_id
        :param zooniverse_id:
        :param expert_markings: do we want to read in markings from experts - either yes or no, shouldn't mix
        :return:
        """

        #mongo_results = list(self.classification_collection.find({self.id_prefix+"zooniverse_id":zooniverse_id}))
        # if we are storing values from previous runs
        if self.pickle_directory != "":
            fname = self.pickle_directory+zooniverse_id+".pickle"
            # check if we have read in this subject before
            if os.path.isfile(fname):
                mongo_results = pickle.load(open(fname,"rb"))
            else:
                mongo_results = list(self.classification_collection.find({self.id_prefix+"zooniverse_id":zooniverse_id}))
                pickle.dump(mongo_results,open(fname,"wb"))
        else:
            # just read in the results
            mongo_results = list(self.classification_collection.find({self.id_prefix+"zooniverse_id":zooniverse_id}))

        annotations_list = []
        user_list = []
        ip_list = []

        for user_index, classification in enumerate(mongo_results):
            # get the name of this user
            if "user_name" in classification:
                # add the _ just we know for certain whether a user was logged in
                # just in case a username is "10" for example, which is a valid ip address
                user_id = classification["user_name"]+"_"

                if not(user_id in self.user_ip_addresses):
                    self.user_ip_addresses[user_id] = set([classification["user_ip"]])
                else:
                    self.user_ip_addresses[user_id].add(classification["user_ip"])
            else:
                user_id = classification["user_ip"]
                logged_in_user = False

            # skip any users who are experts if we do not want experts
            # if we want experts, skip anyone who is not
            # != should be equal to XOR
            # need to remove the last character - "_" for testing if the user is an expert
            if (user_id[:-1] in self.experts) != expert_markings:
                continue

            annotations = self.__classification_to_annotations__(classification)
            if annotations != []:
                annotations_list.append(annotations)
                user_list.append(user_id)

            if max_users is not None:
                if len(user_list) == max_users:
                    break

        # print "num users " + str(len(user_list))
        # if we are not reading in the expert's classifications then we should update the ids
        if not expert_markings:
            self.users_per_subject[zooniverse_id] = set(user_list)
            # self.ips_per_subject[zooniverse_id] = set(ip_list)

            self.all_users.update(user_list)
            # self.all_ips.update(ip_list)

        return user_list,annotations_list


class MarkingProject(OuroborosAPI):
    __metaclass__ = abc.ABCMeta

    def __init__(self, project, date, dimensions, scale=1, experts=[],pickle_directory="/tmp/"):
        OuroborosAPI.__init__(self, project, date, experts=experts,pickle_directory=pickle_directory)
        self.dimensions = dimensions
        self.scale = scale

    # @abc.abstractmethod
    # def __get_cluster_annotations__(self,zooniverse_id):
    #     """
    #     get all aspects of the annotations that are relevant to clustering
    #     override parent class so that we can restrict the annotations to only the data we need for clustering
    #     so for example, we might have an X,Y point and an associated label "adult penguin". That label is not
    #     useful for clustering - at least with how the current set of clustering algorithms work; if we two users
    #     with close points but one says "adult penguin" and the other says "chick" then we assume that the users
    #     are talking about the same point, just confused about what kind of animal is at this point
    #     resolving what kind of animal we have is something that will be done at different point
    #     also things like PCA or such for converting higher dimensional markings down into lower dimensional ones
    #     should be done here
    #     :param zooniverse_id:
    #     :return:
    #     """
    #     return [],[]

    @abc.abstractmethod
    def __annotations_to_markings__(self,annotations):
        """
        This is the main function projects will have to override - given a set of annotations, we need to return the list
        of all markings in that annotation
        """
        return []

    def __get_markings__(self,subject_id,max_users=None,expert_markings=False):
        """
        just allows us to user different terminology so that we are clear about returning markings
        :param subject_id:
        :return:
        """
        return OuroborosAPI.__get_annotations__(self,subject_id,max_users,expert_markings)

    def __classification_to_annotations__(self,classification):
        annotations = classification["annotations"]
        markings = self.__annotations_to_markings__(annotations)

        try:
            object_id = classification["subject_ids"][0]
        except KeyError:
            object_id = classification["subject_id"]

        # go through the markings in reverse order and remove any that are outside of the ROI
        # also, scale as necessary
        assert isinstance(markings, list)
        for marking_index in range(len(markings)-1, -1, -1):
            marking = markings[marking_index]

            # check to see if this marking is in the ROI
            if not(self.__in_roi__(object_id,marking)):
                markings.pop(marking_index)

        return markings

    def __in_roi__(self,object_id,marking):
        """
        is this marking within the specific ROI (if one doesn't exist, then by default yes)
        only override if you want specific ROIs implemented
        :param marking:
        :return:

        """
        return True


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
