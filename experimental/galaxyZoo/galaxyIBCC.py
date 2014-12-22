#!/usr/bin/env python
import csv
import os
import sys
import cPickle as pickle

sys.path.append("/home/greg/github/pyIBCC/python")
import ibcc

subjects = []
users = []

if not(os.path.isfile("/home/greg/Databases/galaxy_zoo_ibcc.csv")):

    ibccOut = open("/home/greg/Databases/galaxy_zoo_ibcc.csv","wb")
    ibccOut.write("a,b,c\n")

    i = 0
    errorCount = 0
    with open("/home/greg/Databases/2014-08-31_galaxy_zoo_classifications.csv","rb") as f:
        reader = csv.reader(f)
        next(reader, None)

        for line in reader:
            subject_id = line[1]
            user_id = line[2]
            candels_0 = line[5]

            if candels_0 == "":
                continue



            try:
                subject_index = subjects.index(subject_id)
            except ValueError:
                subjects.append(subject_id)
                subject_index = len(subjects)-1

            try:
                user_index = users.index(user_id)
            except ValueError:
                users.append(user_id)
                user_index = len(users)-1

            if len(candels_0) == 3:
                candels_index = candels_0[2]
            else:
                errorCount += 1
                continue



            ibccOut.write(str(user_index) + "," + str(subject_index) + "," + candels_index + "\n")

    ibccOut.close()
    print len(subjects)
    print len(users)
    print i
    print "error count " + str(errorCount)
    pickle.dump(subjects,open("/home/greg/Databases/candels","wb"))


with open("/home/greg/Databases/galaxy_zoo_ibcc.py","wb") as f:
    f.write("import numpy as np\n")
    f.write("scores = np.array([0,1,2])\n")
    f.write("nScores = len(scores)\n")
    f.write("nClasses = 3\n")
    f.write("inputFile = \"/home/greg/Databases/galaxy_zoo_ibcc.csv\"\n")
    f.write("outputFile = \"/home/greg/Databases/galaxy_zoo_ibcc.out\"\n")
    f.write("confMatFile = \"/home/greg/Databases/galaxy_zoo_ibcc.mat\"\n")
    f.write("nu0 = np.array([40,40,10])\n")
    f.write("alpha0 = np.array([[5, 2, 2], [2, 5, 2], [3, 3, 3]])\n")

try:
    os.remove("/home/greg/Databases/galaxy_zoo_ibcc.out")
except OSError:
    pass

try:
    os.remove("/home/greg/Databases/galaxy_zoo_ibcc.mat")
except OSError:
    pass

try:
    os.remove("/home/greg/Databases/galaxy_zoo_ibcc.csv.dat")
except OSError:
    pass

import datetime
print datetime.datetime.time(datetime.datetime.now())
ibcc.runIbcc("/home/greg/Databases/galaxy_zoo_ibcc.py")
print datetime.datetime.time(datetime.datetime.now())

