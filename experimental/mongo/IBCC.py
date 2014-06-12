#!/usr/bin/env python
from __future__ import print_function
import csv
import pymongo
from itertools import chain, combinations
import shutil
import os
import sys
if os.path.exists("/home/ggdhines/github/pyIBCC/python"):
    sys.path.append("/home/ggdhines/github/pyIBCC/python")
else:
    sys.path.append("/Users/greghines/Code/pyIBCC/python")
import ibcc


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

class IBCC:
    def __init__(self):
        self.client = pymongo.MongoClient()
        self.db = self.client['serengeti_2014-05-13']

        self.species_groups = [["gazelleThomsons", "gazelleGrants"], ]
        self.species_groups = [["gazelleThomsons"], ["gazelleGrants"]]
        self.speciesList = ['elephant','zebra','warthog','impala','buffalo','wildebeest','gazelleThomsons','dikDik','giraffe','gazelleGrants','lionFemale','baboon','hippopotamus','ostrich','human','otherBird','hartebeest','secretaryBird','hyenaSpotted','mongoose','reedbuck','topi','guineaFowl','eland','aardvark','lionMale','porcupine','koriBustard','bushbuck','hyenaStriped','jackal','cheetah','waterbuck','leopard','reptiles','serval','aardwolf','vervetMonkey','rodents','honeyBadger','batEaredFox','rhinoceros','civet','genet','zorilla','hare','caracal','wildcat']

        self.cutoff = 5

        self.user_list = None
        self.subject_list = None

        if os.path.exists("/Users/greghines/Databases"):
            self.baseDir = "/Users/greghines/Databases/serengeti/"
        else:
            pass

    def __csv_in__(self):
        #check to see if this collection already exists (for this particular cutoff) - if so, skip
        db = self.client["system"]
        collection = db["namespace"]

        if ('merged_classifications'+str(self.cutoff)) in self.db.collection_names():
            print("mongoDB collection already exists")
            return

        reader = csv.reader(open(self.baseDir+"goldFiltered.csv", "rb"), delimiter=",")
        next(reader, None)

        curr_name = None
        curr_id = None
        species_list = []


        collection = self.db['merged_classifications'+str(self.cutoff)]

        zooniverse_id_count = {}

        count = 0

        for row in reader:
            user_name = row[1]
            subject_zooniverse_id = row[2]
            species = row[11]

            if (user_name != curr_name) or (subject_zooniverse_id != curr_id):
                if not(curr_name is None):
                    if curr_id in zooniverse_id_count:
                        zooniverse_id_count[curr_id] += 1
                    else:
                        zooniverse_id_count[curr_id] = 1

                    if zooniverse_id_count[curr_id] <= self.cutoff:
                        count += 1
                        document = {"user_name": curr_name, "subject_zooniverse_id": curr_id, "species_list": species_list}
                        collection.insert(document)

                curr_name = user_name[:]
                species_list = []
                curr_id = subject_zooniverse_id[:]

            species_list.append(species)

        document = {"user_name": curr_name, "subject_zooniverse_id": curr_id, "species_list": species_list}
        collection.insert(document)

    def __createConfigFile(self,counter,numClasses):
        f = open(self.baseDir+"ibcc/"+str(counter)+"config.py",'wb')
        print("import numpy as np\nscores = np.array("+str(range(numClasses))+")", file=f)
        print("nScores = len(scores)", file=f)
        print("nClasses = "+str(numClasses),file=f)
        print("inputFile = '"+self.baseDir+"ibcc/"+str(counter)+".in'", file=f)
        print("outputFile =  '"+self.baseDir+"ibcc/"+str(counter)+".out'", file=f)
        print("confMatFile = '"+self.baseDir+"ibcc/"+str(counter)+".mat'", file=f)
        if numClasses == 4:
            print("alpha0 = np.array([[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2,2, 2]])", file=f)
            print("nu0 = np.array([25.0, 25.0, 25.0, 1.0])", file=f)
        elif numClasses == 2:
            print("alpha0 = np.array([[2, 1], [1, 2],])", file=f)
            print("nu0 = np.array([50.,50.])", file=f)
        else:
            assert(False)
        f.close()

    def __analyze_results__(self):
        #to save having to repeatedly read through the experts' classifications, read them all in now
        expertClassifications = [[] for i in range(len(self.subject_list))]

        try:
            f = open("NA.csv", 'rb')
        except IOError:
            f = open(self.baseDir+"expert_classifications_raw.csv", "rU")

        expertReader = csv.reader(f, delimiter=',')
        next(expertReader, None)
        for row in expertReader:
            subjectID = row[2]
            subjectIndex = self.subject_list.index(subjectID)
            species = row[12]

            #has this species already been added to the list?
            if not(species in expertClassifications[subjectIndex]):
                expertClassifications[subjectIndex].append(species)

        #start off by assuming that we have classified all photos correctly
        correct_classification = [1 for i in range(len(self.subject_list))]

        counter = -1

        #go through each of the species groups, get the user predictions and compare them to the experts' predictions
        for speciesGroup in self.species_groups:
            #find all of the possible subgroups
            required_l = list(powerset(speciesGroup))
            prohibited_l = [[s for s in speciesGroup if not(s in r)] for r in required_l]

            #open up the prediction file corresponding to the next species group
            counter += 1
            ibcc_output_reader = csv.reader(open(self.baseDir+"ibcc/"+str(counter)+".out","rb"), delimiter=" ")

            #go through the predictions for each of the photos (subjects)
            for row in ibcc_output_reader:
                assert(len(row) == (len(required_l)+1))

                #get the subject ID and the predictions
                subjectIndex = int(float(row[0]))
                predictions = [float(r) for r in row[1:]]
                predicted_class = predictions.index(max(predictions))

                #now get the experts' classification (and subject/photo + species group)
                tagged = expertClassifications[subjectIndex]
                meet_required = [sorted(list(set(tagged).intersection(r))) == sorted(list(r)) for r in required_l]
                meet_prohibited = [tuple(set(tagged).intersection(p)) == () for p in prohibited_l]

                meet_overall = [r and p for (r, p) in zip(meet_required, meet_prohibited)]
                assert(sum([1. for o in meet_overall if o]) == 1)

                expert_class = meet_overall.index(True)
                if expert_class != predicted_class:
                    correct_classification[subjectIndex] = 0


        print(len(correct_classification) - sum(correct_classification))


    def __runIBCC__(self):
        collection = self.db['merged_classifications'+str(self.cutoff)]

        self.user_list = []
        self.subject_list = []

        shutil.rmtree(self.baseDir+"ibcc")
        os.makedirs(self.baseDir+"ibcc")

        counter = -1

        for speciesGroup in self.species_groups:
            required_l = list(powerset(speciesGroup))
            prohibited_l = [[s for s in speciesGroup if not(s in r)] for r in required_l]

            counter += 1

            self.__createConfigFile(counter,len(required_l))
            ibcc_input_file = open(self.baseDir+"ibcc/"+str(counter)+".in","wb")


            for document in collection.find():
                user_name = document["user_name"]
                subject_zooniverse_id = document["subject_zooniverse_id"]
                user_species_list = document["species_list"]

                #IBCC requires an int ID for both user and subject - so convert
                if user_name in self.user_list:
                    userID = self.user_list.index(user_name)
                else:
                    self.user_list.append(user_name)
                    userID = len(self.user_list)-1

                if subject_zooniverse_id in self.subject_list:
                    subjectID = self.subject_list.index(subject_zooniverse_id)
                else:
                    self.subject_list.append(subject_zooniverse_id)
                    subjectID = len(self.subject_list)-1

                #which class does this classification count as?
                meet_required = [sorted(list(set(user_species_list).intersection(r))) == sorted(list(r)) for r in required_l]
                meet_prohibited = [tuple(set(user_species_list).intersection(p)) == () for p in prohibited_l]
                meet_overall = [r and p for (r, p) in zip(meet_required, meet_prohibited)]
                assert(sum([1. for o in meet_overall if o]) == 1)

                class_id = meet_overall.index(True)
                print(str(userID) + "," + str(subjectID) + "," + str(class_id), file=ibcc_input_file)

            ibcc_input_file.close()

            #now run IBCC
            ibcc.runIbcc(self.baseDir+"ibcc/"+str(counter)+"config.py")

    def __merege_predictions__(self):
        counter = -1
        overall_predictions = [[] for i in range(len(self.subject_list))]

        for speciesGroup in self.species_groups:
            required_l = list(powerset(speciesGroup))

            counter += 1
            ibcc_output_file = open(self.baseDir+"ibcc/"+str(counter)+".out","rb")
            reader = csv.reader(ibcc_output_file, delimiter=' ')

            for row in reader:
                id = int(float(row[0]))
                predictions = [float(c) for c in row[1:]]
                predicted = required_l[predictions.index(max(predictions))]

                overall_predictions[id].extend(predicted)


        #find all of the possible subgroups
        required_l = list(powerset(self.species_groups2[0]))
        prohibited_l = [[s for s in self.species_groups2[0] if not(s in r)] for r in required_l]
        m = [0 for i in range(len(required_l))]

        for predict in overall_predictions:
            meet_required = [sorted(list(set(predict).intersection(r))) == sorted(list(r)) for r in required_l]
            meet_prohibited = [tuple(set(predict).intersection(p)) == () for p in prohibited_l]
            meet_overall = [r and p for (r, p) in zip(meet_required, meet_prohibited)]

            #print("===---")
            #print(predict)
            #print(required_l)
            #print(meet_required)
            #print(prohibited_l)
            #print(meet_prohibited)
            assert(sum([1. for o in meet_overall if o]) == 1)
            class_id = meet_overall.index(True)
            m[class_id] += 1
        print(required_l)
        print(m)

    def __merge_confusion_matrcies__(self):
        for speciesGroup in self.species_groups:
            pass




f = IBCC()
#f.__csv_in__()
#f.__IBCCoutput__([["gazelleThomsons","gazelleGrants"],])
f.__runIBCC__()
#f.__analyze_results__()
#f.__merege_predictions__()