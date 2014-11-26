#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import os
import pymongo
import sys
import cPickle as pickle
import bisect
import csv
import matplotlib.pyplot as plt
import random

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError

sys.path.append(base_directory+"/github/pyIBCC/python")
import ibcc

if os.path.exists("/home/ggdhines"):
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/clusteringAlg")
else:
    sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")
#from divisiveDBSCAN import DivisiveDBSCAN
from divisiveDBSCAN_multi import DivisiveDBSCAN
from divisiveKmeans import DivisiveKmeans

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

client = pymongo.MongoClient()
db = client['condor_2014-11-23']
classification_collection = db["condor_classifications"]
subject_collection = db["condor_subjects"]




big_userList = []
big_subjectList = []
animal_count = 0

f = open(base_directory+"/Databases/condor_ibcc.csv","wb")
f.write("a,b,c\n")
alreadyDone = []
animals_in_image = {}
animal_index = -1
global_user_list = []
animal_to_image = []
zooniverse_list = []
condor_votes = {}
animal_votes = []

#to_sample_from = list(subject_collection.find({"state":"complete"}))
to_sample_from = list(subject_collection.find({"classification_count":{"$gt":1},"state":{"$in":["active","complete"]}}))

animals = ["condor","turkeyVulture","goldenEagle","raven","coyote"]

#for subject_index,subject in enumerate(subject_collection.find({"classification_count":{"$gt":1},"state":{"$in":["active","complete"]}})):
for subject_index,subject in enumerate(random.sample(to_sample_from,5000)):
    print subject_index
    zooniverse_id = subject["zooniverse_id"]
    zooniverse_list.append(zooniverse_id)
    annotation_list = []
    user_list = []
    animal_list = []
    local_users = []

    condor_votes[zooniverse_id] = []

    for user_index,classification in enumerate(classification_collection.find({"subjects.zooniverse_id":zooniverse_id})):
        try:
            mark_index = [ann.keys() for ann in classification["annotations"]].index(["marks",])
            markings = classification["annotations"][mark_index].values()[0]

            if "user_name" in classification:
                user = classification["user_name"]
            else:
                user = classification["user_ip"]
            found_condor = False

            for animal in markings.values():
                scale = 1.875
                x = scale*float(animal["x"])
                y = scale*float(animal["y"])

                animal_type = animal["animal"]
                #if not(animal_type in ["carcassOrScale","carcass","other"]):
                if animal_type in animals:
                    annotation_list.append((x,y))
                    #print annotation_list
                    user_list.append(user_index)
                    animal_list.append(animal_type)
                    if not(user in global_user_list):
                        global_user_list.append(user)
                    local_users.append(user)

                if animal_type == "condor":
                    found_condor = True

            if found_condor:
                condor_votes[zooniverse_id].append(1)
            else:
                condor_votes[zooniverse_id].append(0)

        except (ValueError,KeyError):
            pass


    if annotation_list != []:
        user_identified,clusters = DivisiveKmeans(3).fit2(annotation_list,user_list,debug=True)

        if user_identified != []:
            user_identified,clusters = DivisiveKmeans(3).__fix__(user_identified,clusters,annotation_list,user_list,200)

        if user_identified != []:

            for c in clusters:
                animal_index += 1
                animal_votes.append([])
                animal_to_image.append(zooniverse_id)

                if not(zooniverse_id in animals_in_image):
                    animals_in_image[zooniverse_id] = [animal_index]
                else:
                    animals_in_image[zooniverse_id].append(animal_index)
                for pt in c:
                    pt_index = annotation_list.index(pt)
                    user_index = global_user_list.index(local_users[pt_index])
                    animal_type = animal_list[annotation_list.index(pt)]
                    #animal_index = animals.index(animal_type)
                    try:
                        f.write(str(user_index) + ","+str(animal_index) + ","+str(animals.index(animal_type))+"\n")
                    except ValueError:
                        print animal_type
                        raise
                    animal_votes[-1].append(animals.index(animal_type))
                    # if animal_type == "condor":
                    #     f.write(str(user_index) + ","+str(animal_index) + ",1\n")
                    #     animal_votes[-1].append(1)
                    # else:
                    #     f.write(str(user_index) + ","+str(animal_index) + ",0\n")
                    #     animal_votes[-1].append(0)


#print animal_votes
pluraity_vote = [v[np.argmax(v)] for v in animal_votes]
f.close()
priors = [max(int(sum([1 for p in pluraity_vote if p == i])/float(len(pluraity_vote))*100),1) for i in range(len(animals))]
#print priors
print priors
#print [sum([1 for p in pluraity_vote if p == i]) for i in range(len(animals))]
global_confusion = []

#now estimate the confusion matrix
for animal_type in range(len(animals)):
    confusion = [0 for i in range(len(animals))]
    for v_index,v in enumerate(animal_votes):
        if pluraity_vote[v_index] == animal_type:
            #convert into an overall count
            overall_count = [sum([1 for i in v if i == j]) for j in range(len(animals))]
            animal_confusion = [c/float(sum(overall_count)) for c in overall_count]
            confusion = [c+a for c,a in zip(confusion,animal_confusion)]

    global_confusion.append([max(int(c*(len(animals)**2)/float(sum(confusion))),1) for c in confusion])

print global_confusion
with open(base_directory+"/Databases/condor_ibcc.py","wb") as f:
    f.write("import numpy as np\n")
    f.write("scores = np.array("+str(range(len(animals)))+")\n")
    f.write("nScores = "+str(len(animals))+"\n")
    f.write("nClasses = " + str(len(animals))+"\n")
    f.write("inputFile = \""+base_directory+"/Databases/condor_ibcc.csv\"\n")
    f.write("outputFile = \""+base_directory+"/Databases/condor_ibcc.out\"\n")
    f.write("confMatFile = \""+base_directory+"/Databases/condor_ibcc.mat\"\n")
    f.write("nu0 = np.array("+str(priors)+")\n")
    f.write("alpha0 = np.array("+str(global_confusion)+")\n")
    #f.write("alpha0 = np.array([[4,1],[1,2]])\n")


#start by removing all temp files
try:
    os.remove(base_directory+"/Databases/condor_ibcc.out")
except OSError:
    pass

try:
    os.remove(base_directory+"/Databases/condor_ibcc.mat")
except OSError:
    pass

try:
    os.remove(base_directory+"/Databases/condor_ibcc.csv.dat")
except OSError:
    pass

#pickle.dump((big_subjectList,big_userList),open(base_directory+"/Databases/tempOut.pickle","wb"))
ibcc.runIbcc(base_directory+"/Databases/condor_ibcc.py")

#now analyze the data
#assume for starters that each image does not have a condor
X = []
Y = []
X_2 = []
Y_2 = []
contains_condor = {zooniverse_id:False for zooniverse_id in zooniverse_list}
condor_probabilities = {zooniverse_id:[] for zooniverse_id in zooniverse_list}
with open(base_directory+"/Databases/condor_ibcc.out","rb") as f:
    ibcc_results = csv.reader(f, delimiter=' ')

    for row in ibcc_results:
        animal_index = int(float(row[0]))
        condor_p = float(row[1])

        condor_probabilities[animal_to_image[animal_index]].append(condor_p)

print "=== " + str(len(condor_votes))

for zooniverse_id,votes in condor_votes.items():
    if votes == []:
        continue

    if condor_probabilities[zooniverse_id] == []:
        #X.append(0)
        x = 0
    else:
        #X.append(max(condor_probabilities[zooniverse_id]))
        x = max(condor_probabilities[zooniverse_id])

    #Y.append(np.mean(votes))
    y = np.mean(votes)

    subject = subject_collection.find_one({"zooniverse_id":zooniverse_id})

    if subject["state"] == "complete":
        X.append(x)
        Y.append(y)
    else:
        X_2.append(x)
        Y_2.append(y)

    state = subject["state"]
    if (state == "complete") and ((x>0.5) and (y<0.5)) or ((x<0.5) and (y>0.5)):
        print "error"
        print x,y
        print subject["location"]["standard"]

    #if state == "active":
    #    print "====----"
    #    print x,y
    #    print subject["location"]["standard"]
    #    print subject["classification_count"]

plt.plot(X,Y,'.',color="blue")
plt.plot(X_2,Y_2,'.',color="red")
plt.xlim((-0.05,1.05))
plt.ylim((-0.05,1.05))
plt.show()
