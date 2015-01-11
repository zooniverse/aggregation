__author__ = 'greg'
import pymongo


class Mongo:
    def __init__(self,project,date):
        self.project = project

        client = pymongo.MongoClient()
        db = client[project+"_"+date]
        self.classification_collection = db[project+"_classifications"]
        self.subject_collection = db[project+"_subjects"]
        self.user_collection = db[project+"condor_users"]

        #we a global list of logged in users so we use the index for the same user over multiple images
        self.all_users = []

        #we need a list of of users per subject (and a separate one for just those users who were not logged in
        #those ones will just have ip addresses
        self.users_per_subject = {}
        self.ips_per_subject = {}

        #dictionaries for the raw markings per image
        self.markings_list = {}
        self.user_list = {}
        #what did the user think they saw these coordinates?
        #for example, in penguin watch, it could be a penguin
        self.what_list = {}

        #the clustering results per image
        self.clusterResults = {}

        self.signal_probability = []

