#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import os
import pymongo
import sys
import cPickle as pickle
import bisect
import random
import csv
import matplotlib.pyplot as plt

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
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

client = pymongo.MongoClient()
db = client['condor_2014-11-20']
classification_collection = db["condor_classifications"]
subject_collection = db["condor_subjects"]

gold = pickle.load(open(base_directory+"/condor_gold.pickle","rb"))
gold.sort(key = lambda x:x[1])
to_sample_from = (zip(*gold)[0])[1301:]
sample = random.sample(to_sample_from,500)

big_userList = []
big_subjectList = []
animal_count = 0

f = open(base_directory+"/Databases/condor_ibcc.csv","wb")
f.write("a,b,c\n")
alreadyDone = []

subjectVote = {}

gold_condor = []

for subject_index,zooniverse_id in enumerate(sample):
    print subject_index

    #gold standard
    gold_classification = classification_collection.find_one({"user_name":"wreness", "subjects.zooniverse_id":zooniverse_id})
    assert gold_classification["tutorial"] == False

    found_condor = False

    try:
        mark_index = [ann.keys() for ann in gold_classification["annotations"]].index(["marks",])
        markings = gold_classification["annotations"][mark_index].values()[0]


        try:
            for animal in markings.values():
                animal_type = animal["animal"]
                found_condor = (animal_type == "condor")
        except KeyError:
            continue
    except ValueError:
        pass

    if found_condor:
        gold_condor.append(1)
    else:
        gold_condor.append(0)

    alreadyDone = []
    classification_count = 0
    for classification in classification_collection.find({"subjects.zooniverse_id":zooniverse_id}):
        if ("user_name" in classification) and (classification["user_name"] == "wreness"):
            continue


        if "user_name" in classification:
            user = classification["user_name"]
        else:
            user = classification["user_ip"]

        if user in alreadyDone:
            continue

        alreadyDone.append(user)

        if not(user in big_userList):
            big_userList.append(user)
        if not(zooniverse_id in big_subjectList):
            big_subjectList.append(zooniverse_id)

        user_index = big_userList.index(user)
        subject_index = big_subjectList.index(zooniverse_id)

        classification_count += 1
        if classification_collection == 50:
            break


        try:
            mark_index = [ann.keys() for ann in classification["annotations"]].index(["marks",])
            markings = classification["annotations"][mark_index].values()[0]

            found = False
            for animal in markings.values():

                animal_type = animal["animal"]
                if animal_type in ["condor"]:
                    found = True
                    break




            if found:
                f.write(str(user_index) + ","+str(subject_index) + ",1\n")
                if not(zooniverse_id in subjectVote):
                    subjectVote[zooniverse_id] = [1]
                else:
                    subjectVote[zooniverse_id].append(1)
            else:
                f.write(str(user_index) + ","+str(subject_index) + ",0\n")
                if not(zooniverse_id in subjectVote):
                    subjectVote[zooniverse_id] = [0]
                else:
                    subjectVote[zooniverse_id].append(0)

        except (ValueError,KeyError):
            f.write(str(user_index) + ","+str(subject_index) + ",0\n")
            if not(zooniverse_id in subjectVote):
                subjectVote[zooniverse_id] = [0]
            else:
                subjectVote[zooniverse_id].append(0)



condor_count = 0.
total_count = 0.
false_positives = []
true_positives = []
false_negatives = []
true_negatives = []

for votes in subjectVote.values():
    if np.mean(votes) >= 0.5:
        condor_count += 1
        true_positives.append(np.mean(votes))
        #false_negatives.append(1-np.mean(votes))
    else:
        #false_positives.append(np.mean(votes))
        true_negatives.append(1-np.mean(votes))

    total_count += 1

pp = condor_count / total_count

print np.mean(true_positives)
print np.mean(true_negatives)

f.close()
with open(base_directory+"/Databases/condor_ibcc.py","wb") as f:
    f.write("import numpy as np\n")
    f.write("scores = np.array([0,1])\n")
    f.write("nScores = len(scores)\n")
    f.write("nClasses = 2\n")
    f.write("inputFile = \""+base_directory+"/Databases/condor_ibcc.csv\"\n")
    f.write("outputFile = \""+base_directory+"/Databases/condor_ibcc.out\"\n")
    f.write("confMatFile = \""+base_directory+"/Databases/condor_ibcc.mat\"\n")
    f.write("nu0 = np.array(["+str(int((1-pp)*100))+","+str(int(pp*100))+"])\n")
    #f.write("alpha0 = np.array([[20,1], [1,20]])\n")



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


values = []
errors = 0
low = 0
X_positive = []
X_negative = []
with open(base_directory+"/Databases/condor_ibcc.out","rb") as f:
    ibcc_results = csv.reader(f, delimiter=' ')

    for ii,row in enumerate(ibcc_results):
        if ii == 20000:
            break

        wreness_condor = gold_condor[ii]
        ibcc_condor = float(row[2])

        if wreness_condor == 0:
            X_negative.append(ibcc_condor)
        else:
            X_positive.append(ibcc_condor)



#print X_negative
# print X_positive
# plt.hist([X_positive,X_negative],10)
# plt.show()
alpha_list = X_negative[:]
alpha_list.extend(X_positive)
alpha_list.sort()

roc_X = []
roc_Y = []
for alpha in alpha_list:
    positive_count = sum([1 for x in X_positive if x >= alpha])
    positive_rate = positive_count/float(len(X_positive))

    negative_count = sum([1 for x in X_negative if x >= alpha])
    negative_rate = negative_count/float(len(X_negative))

    roc_X.append(negative_rate)
    roc_Y.append(positive_rate)

#print roc_X

plt.plot(roc_X,roc_Y)
#plt.xlim((0,1.05))
plt.plot((0,1),(0,1),'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.plot([0.058],[0.875],'o')
plt.show()