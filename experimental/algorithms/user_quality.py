#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import os
import pymongo
import datetime
from datetime import timedelta
import psycopg2
import matplotlib.pyplot as plt
# the directory to store the movie preview clips in
image_directory = "/home/greg/Databases/chimp/images/"

# connect to the mongodb server
client = pymongo.MongoClient()
db = client['serengeti_2015-06-27']
subjects = db["serengeti_subjects"]
classifications = db["serengeti_classifications"]
users = db["serengeti_users"]
# user = "yshish"

details = "dbname='zooniverse' user='greg' host='localhost' password='apassword'"
postgres_session = psycopg2.connect(details)
postgres_cursor = postgres_session.cursor()

length = 50

accuracy_dict = {}
subject_dict = {}

speciesList = ['elephant','zebra','warthog','impala','buffalo','wildebeest','gazelleThomsons','dikDik','giraffe','gazelleGrants','lionFemale','baboon','hippopotamus','ostrich','human','otherBird','hartebeest','secretaryBird','hyenaSpotted','mongoose','reedbuck','topi','guineaFowl','eland','aardvark','lionMale','porcupine','koriBustard','bushbuck','hyenaStriped','jackal','cheetah','waterbuck','leopard','reptiles','serval','aardwolf','vervetMonkey','rodents','honeyBadger','batEaredFox','rhinoceros','civet','genet','zorilla','hare','caracal','wildcat']

def user_accuracy(user,start_time,user_ip):
    delta_t = timedelta(days=2)
    end_time = start_time + delta_t
    correct_total = 0

    # if "user_name" in user:
    #     print users.find_one({"name":user["user_name"]})
    total = 0
    # print user
    # print user
    # print classifications.find(user).count()
    # print classifications.find_one(user)
    postgres_cursor.execute("select subject_id,annotations from Snapshot2 where user_name = '"+user+"' and user_ip = '" + user_ip + "' and created_at >= TIMESTAMP '"+ str(start_time) + "' and created_at <= TIMESTAMP '" + str(end_time) + "' order by created_at ASC")
    for subject_id,annotations in postgres_cursor.fetchall()[:100]:
        species = []
        for a in annotations:
            if "species" in a.keys():
                species.extend(a["species"].lower().split("-"))

        subject = subjects.find_one({"zooniverse_id":subject_id})

        try:
            retire_reason = subject["metadata"]["retire_reason"]
        except KeyError:
            print subject
            raise

        if retire_reason in ["blank","consensus"]:
            total += 1

            aggregate_species = []
            for s,c in subject["metadata"]["counters"].items():
                if (c >= 5) and (s != "blank"):
                    aggregate_species.extend(s.split("-"))
            # aggregate_species = [s.lower() for (s,c)  if (c >= 5) and (s != "blank")]
            correct = (sorted(aggregate_species) == (sorted(species)))
            if correct:
                correct_total += 1
        elif retire_reason == "blank_consensus":
            pass
        elif retire_reason == "complete":
            # print subject
            if subject_id not in subject_dict:
                c2 = postgres_session.cursor()
                c2.execute("select annotations from Snapshot where subject_id = '"+subject_id+"'")

                spotted = []

                # votes = {s:0 for s in speciesList}
                votes = {}

                for classification in c2.fetchall()[:25]:
                    spotted_animals = []
                    # print classification
                    for ann in classification[0]:
                        if "species" in ann:
                            s = ann["species"]
                            spotted_animals.append(ann["species"])
                            if s not in votes:
                                votes[s] = 1
                            else:
                                votes[s] += 1
                        else:
                            break
                    spotted.append(spotted_animals)

                median_animals = np.median([len(s) for s in spotted])
                subject_dict[subject_id] = sorted(zip(*sorted(votes.items(),key = lambda x:x[1],reverse=True)[:int(median_animals)])[0])

            if subject_dict[subject_id] == sorted(species):
                correct_total += 1
            total += 1

        else:
            # print retire_reason
            pass

    if total > 10:
        return correct_total/float(total)
    else:
        return -1
    # for ii,c in enumerate(classifications.find(user).limit(length)):
    #     # if (ii >0) and (ii %100 == 0):
    #     #     print correct_total
    #     #     correct_total = 0
    #
    #
    #     subject = c["subjects"][0]["zooniverse_id"]
    #
    #     species = []
    #     for a in  c["annotations"]:
    #         if "species" in a.keys():
    #             species.extend(a["species"].lower().split("-"))
    #
    #     subject = subjects.find_one({"zooniverse_id":subject})
    #     try:
    #         retire_reason = subject["metadata"]["retire_reason"]
    #     except KeyError:
    #         print subject
    #         raise
    #
    #     if retire_reason in ["blank","consensus"]:
    #         total += 1
    #
    #         aggregate_species = []
    #         for s,c in subject["metadata"]["counters"].items():
    #             if (c >= 5) and (s != "blank"):
    #                 aggregate_species.extend(s.split("-"))
    #         # aggregate_species = [s.lower() for (s,c)  if (c >= 5) and (s != "blank")]
    #         correct = (sorted(aggregate_species) == (sorted(species)))
    #         if correct:
    #             correct_total += 1
    #     elif retire_reason == "blank_consensus":
    #         pass
    #     elif retire_reason == "complete":
    #         # print subject
    #         pass
    #     else:
    #         # print retire_reason
    #         pass
    #     # else:
    #     #     print (sorted(aggregate_species),(sorted(species)))
    #
    #
    # # print total
    # # print
    #
    # if total > 10:
    #     return correct_total/float(total)
    # else:
    #     return -1


# accuracy_list = []
#
# for jj,c in enumerate(classifications.find({"tutorial":{"$ne":True},"created_at":{"$gte":datetime.datetime(2013, 1, 1)}}).sort("created_at",1).limit(5000)):
# # for jj,c in enumerate(classifications.find({"tutorial":{"$ne":True}}).skip(1000).limit(5000)):
#     print c["created_at"]
#     if (jj > 0) and ((jj % 100) == 0):
#         avg_acc = np.mean([a for a in accuracy_list if a >= 0])
#         num = len([a for a in accuracy_list if a >= 0])
#         print avg_acc,num
#         accuracy_list = []
#
#
#     user_constraints = [{"tutorial":{"$ne":True}}]
#     if "user_name" in c:
#         user_constraints.append({"user_name":c["user_name"]})
#     else:
#         user_constraints.append({"user_ip":c["user_ip"]})
#     user_constraints.append({"created_at":{"$gte":c["created_at"]}})
#     new_date = c["created_at"]+delta_t
#     user_constraints.append({"created_at":{"$lte":new_date}})
#
#     # user_dict["created_at"] = {"$and" : []}
#     accuracy_list.append(user_accuracy({"and":user_constraints}))
#
# avg_acc = np.mean([a for a in accuracy_list if a >= 0])
# num = len([a for a in accuracy_list if a >= 0])
# print avg_acc,num

X = []
accuracy_list = []
postgres_cursor.execute("select * from Snapshot2 ORDER BY created_at ASC")
for ii,r in enumerate(postgres_cursor.fetchall()[:1000000]):
    # print ii
    if (ii > 0) and (ii % 1000) == 0:
        print r[3]
        if [a for a in accuracy_list if a >= 0] != []:
            X.append(np.mean([a for a in accuracy_list if a >= 0]))
        accuracy_list = []
        accuracy_dict = {}

    user = r[2]
    if user not in accuracy_dict:
        accuracy_dict[user] = user_accuracy(r[2],r[3],r[1])

    accuracy_list.append(accuracy_dict[user])
print X

plt.plot(range(len(X)),X)
plt.show()