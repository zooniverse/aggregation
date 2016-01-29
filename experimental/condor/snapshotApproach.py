#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import os
import pymongo
import random
import cPickle as pickle
import matplotlib.pyplot as plt



if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

client = pymongo.MongoClient()
db = client['condor_2014-11-23']
classification_collection = db["condor_classifications"]
subject_collection = db["condor_subjects"]

gold = pickle.load(open(base_directory+"/condor_gold.pickle","rb"))
print len(gold)
gold.sort(key = lambda x:x[1])
gold = gold[1300:]
print len(gold)
ids = zip(*gold)[0]
train = random.sample(ids,500)
test = [zooniverse_id for zooniverse_id in ids if not(zooniverse_id in train)]

user_abilities = {}
user_score = {}

for ii,zooniverse_id in enumerate(train):
    print ii
    gold_classification = classification_collection.find_one({"subjects.zooniverse_id":zooniverse_id,"user_name":"wreness"})

    gold_condor = False
    try:
        for markings in gold_classification["annotations"][3].values()[0].values():
            try:
                if markings["animal"] == "condor":
                    gold_condor = True
                    break
            except KeyError:
                pass

    except IndexError:
        pass


    for classification in classification_collection.find({"subjects.zooniverse_id":zooniverse_id,"user_name":{"$ne":"wreness"}}):
        if "user_name" in classification:
            user_name = classification["user_name"]
        else:
            user_name = classification["user_ip"]

        if not(user_name in user_abilities):
            user_abilities[user_name] = [[0.,0.],[0.,0.]]
        user_condor = False
        try:
            for markings in classification["annotations"][3].values()[0].values():
                try:
                    if markings["animal"] == "condor":
                        user_condor = True
                        break
                except KeyError:
                    pass
        except IndexError:
            pass

        if gold_condor:
            if user_condor:
                user_abilities[user_name][1][1] += 1
            else:
                user_abilities[user_name][1][0] += 1
        else:
            if user_condor:
                user_abilities[user_name][0][1] += 1
            else:
                user_abilities[user_name][0][0] += 1
beta = 0.01
for user_name,record in user_abilities.items():
    user_score[user_name] = (beta*record[0][0]+record[1][1])/(beta*record[0][0]+record[1][1] + record[0][1]+record[1][0])

prior = 0.928

record = [[0.,0.],[0.,0.]]

X_false_positive = []
Y_true_positive = []

for ii,zooniverse_id in enumerate(test):
    if ii == 500:
        break

    print ii
    gold_classification = classification_collection.find_one({"subjects.zooniverse_id":zooniverse_id,"user_name":"wreness"})

    gold_condor = False
    try:
        for markings in gold_classification["annotations"][3].values()[0].values():
            try:
                if markings["animal"] == "condor":
                    gold_condor = True
                    break
            except KeyError:
                pass

    except IndexError:
        pass

    votes = []
    weights = []

    for classification in classification_collection.find({"subjects.zooniverse_id":zooniverse_id,"user_name":{"$ne":"wreness"}}):
        if "user_name" in classification:
            user_name = classification["user_name"]
        else:
            user_name = classification["user_ip"]

        user_condor = False
        try:
            for markings in classification["annotations"][3].values()[0].values():
                try:
                    if markings["animal"] == "condor":
                        user_condor = True
                        break
                except KeyError:
                    pass
        except IndexError:
            pass

        if user_name in user_score:
            individual_weight = max(0.1,user_score[user_name])
        else:
            individual_weight = prior

        assert(individual_weight > 0)

        if user_condor:
            votes.append(individual_weight)
        else:
            votes.append(0)

        weights.append(individual_weight)

    if len(votes) == 0:
        continue

    for j in range(5):
        #sampled = random.sample(votes,min(2,len(votes)))
        sampled = random.sample(range(len(votes)),min(2,len(votes)))
        a = sum([votes[k] for k in sampled])
        b = sum([weights[k] for k in sampled])

        if gold_condor:
            Y_true_positive.append(a/b)
        else:
            X_false_positive.append(a/b)

alpha_list = X_false_positive[:]
alpha_list.extend(Y_true_positive)
alpha_list.sort()

roc_X = []
roc_Y = []
for alpha in alpha_list:
    positive_count = sum([1 for x in Y_true_positive if x >= alpha])
    positive_rate = positive_count/float(len(Y_true_positive))

    negative_count = sum([1 for x in X_false_positive if x >= alpha])
    negative_rate = negative_count/float(len(X_false_positive))

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
