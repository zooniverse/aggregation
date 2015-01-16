#!/usr/bin/env python
import csv
import bisect
import os
import sys
import random

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


subjects = []
users = []
i = 0


with open(base_directory+"/Databases/galaxyZoo.csv","wb") as fOut:
    with open(base_directory+"/Databases/2014-08-31_galaxy_zoo_classifications.csv","rb") as f:
        reader = csv.reader(f)
        next(reader, None)

        for line in reader:
            i += 1

            subject_id = line[1]
            user_id = line[2]
            candels_0 = line[5]

            if candels_0 == "":
                continue

            if len(candels_0) != 3:
                continue

            fOut.write("\""+user_id+"\",\""+subject_id+"\",\""+candels_0+"\"\n")

            try:
                tt = index(subjects,subject_id)
            except ValueError:
                bisect.insort(subjects,subject_id)

            try:
                tt = index(users,user_id)
            except ValueError:
                bisect.insort(users,user_id)

                #if len(users) == 70000:
                #    break


print len(subjects)
print len(users)

s = random.sample(subjects,400)
sampled = {}

with open(base_directory+"/Databases/galaxy_zoo_ibcc.csv","wb") as ibccOut:
    ibccOut.write("a,b,c\n")

    with open(base_directory+"/Databases/galaxyZoo.csv","rb") as f:
        reader = csv.reader(f)

        for line in reader:
            user_id = line[0]
            subject_id = line[1]
            candels_0 = line[2]

            #print line
            #print user_id

            subject_index = index(subjects,subject_id)
            user_index = index(users,user_id)
            candels_index = candels_0[2]

            if subject_id in s:
                if not(subject_id in sampled):
                    sampled[subject_id] = [0,0,0]

                sampled[subject_id][int(candels_index)] += 1
            assert user_index <= len(users)
            ibccOut.write(str(user_index) + "," + str(subject_index) + "," + candels_index + "\n")

overallVotes = [0,0,0]
print overallVotes
confusion = [[0,0,0],[0,0,0],[0,0,0]]
for votes in sampled.values():
    F = [i for i,v in enumerate(votes) if v == max(votes)]
    mostLikely = random.sample(F,1)[0]
    overallVotes[mostLikely] += 1

    votes = [v/float(sum(votes)) for v in votes]
    for i,v in enumerate(votes):
        confusion[mostLikely][i] += v
#print overallVotes
#print sum(overallVotes)
overallVotes = [max(int(v/float(sum(overallVotes))*100),1) for v in overallVotes]
print overallVotes
#print confusion
confusion = [[int(c/(min(con)/1.)) for c in con] for con in confusion]

with open(base_directory+"/Databases/galaxy_zoo_ibcc.py","wb") as f:
    f.write("import numpy as np\n")
    f.write("scores = np.array([0,1,2])\n")
    f.write("nScores = len(scores)\n")
    f.write("nClasses = 3\n")
    f.write("inputFile = \""+base_directory+"/Databases/galaxy_zoo_ibcc.csv\"\n")
    f.write("outputFile = \""+base_directory+"/Databases/galaxy_zoo_ibcc.out\"\n")
    f.write("confMatFile = \""+base_directory+"/Databases/galaxy_zoo_ibcc.mat\"\n")
    f.write("goldFile = \""+base_directory+"/Databases/gold.csv\"\n")
    f.write("nu0 = np.array("+str(overallVotes)+")\n")
    f.write("alpha0 = np.array("+str(confusion)+")\n")

try:
    os.remove(base_directory+"/Databases/galaxy_zoo_ibcc.out")
except OSError:
    pass

try:
    os.remove(base_directory+"/Databases/galaxy_zoo_ibcc.mat")
except OSError:
    pass

try:
    os.remove(base_directory+"/Databases/galaxy_zoo_ibcc.csv.dat")
except OSError:
    pass

import datetime
print datetime.datetime.time(datetime.datetime.now())
print base_directory+"/Databases/galaxy_zoo_ibcc.py"
ibcc.runIbcc(base_directory+"/Databases/galaxy_zoo_ibcc.py")
print datetime.datetime.time(datetime.datetime.now())

#read in gold standard data
pos0 = []
pos1 = []
pos2 = []
with open(base_directory+"/Downloads/candels_t01_a00_positive.dat","rb") as f:
    for l in f.readlines():
        pos0.append(l[:-1])
with open(base_directory+"/Downloads/candels_t01_a01_positive.dat","rb") as f:
    for l in f.readlines():
        pos1.append(l[:-1])
with open(base_directory+"/Downloads/candels_t01_a02_positive.dat","rb") as f:
    for l in f.readlines():
        pos2.append(l[:-1])

found0 = 0
found1 = 0
found2 = 0
correct0 = 0
correct1 = 0
correct2 = 0
incorrect = [0,0,0]

confusion = [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]

Y_true_positive = []
X_false_positive = []

results = []

with open(base_directory+"/Databases/galaxy_zoo_ibcc.out","rb") as f:
    reader = csv.reader(f,delimiter=" ")

    for line in reader:
        subject_index = int(float(line[0]))
        subject_id = subjects[subject_index]
        probabilities = [float(p) for p in line[1:]]

        if (subject_id in pos0) or (subject_id in pos1) or (subject_id in pos2):
            results.append((probabilities,(subject_id in pos0),(subject_id in pos1),(subject_id in pos2)))

        if subject_id in pos0:
            found0 += 1
            if probabilities[0] == max(probabilities):
                correct0 += 1
            else:
                incorrect[probabilities.index(max(probabilities))] += 1

            confusion[0][probabilities.index(max(probabilities))] += 1

            X_false_positive.append(probabilities[2])

        elif subject_id in pos1:
            found1 += 1
            if probabilities[1] == max(probabilities):
                correct1 += 1
            else:
                incorrect[probabilities.index(max(probabilities))] += 1

            confusion[1][probabilities.index(max(probabilities))] += 1
            X_false_positive.append(probabilities[2])
        elif subject_id in pos2:
            found2 += 1
            if probabilities[2] == max(probabilities):
                correct2 += 1
            else:
                incorrect[probabilities.index(max(probabilities))] += 1

            confusion[2][probabilities.index(max(probabilities))] += 1
            Y_true_positive.append(probabilities[2])

print correct0,found0
print correct1,found1
print correct2,found2
print incorrect
print "====="
for c in confusion:
    print c

roc_X = []
roc_Y = []

alpha_list = X_false_positive[:]
alpha_list.extend(Y_true_positive)
alpha_list.sort()

for alpha in alpha_list:
    positive_count = sum([1 for x in Y_true_positive if x >= alpha])
    positive_rate = positive_count/float(len(Y_true_positive))

    negative_count = sum([1 for x in X_false_positive if x >= alpha])
    negative_rate = negative_count/float(len(X_false_positive))

    roc_X.append(negative_rate)
    roc_Y.append(positive_rate)

print roc_X
true_positive = len(Y_true_positive)/float(len(Y_true_positive)+len(X_false_positive))
false_postive = len(X_false_positive)/float(len(Y_true_positive)+len(X_false_positive))
true_positive = confusion[2][2]/sum(confusion[2])
false_positive = (confusion[2][0]+confusion[2][1])/sum(confusion[2])
import matplotlib.pyplot as plt
plt.plot(roc_X,roc_Y)
#plt.xlim((0,1.05))
plt.plot((0,1),(0,1),'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.plot([false_positive],[true_positive],'o')
plt.show()


import cPickle as pickle
pickle.dump(subjects,open(base_directory+"/Databases/subjects.pickle","wb"))