#!/usr/bin/env python
__author__ = 'greg'
from cassandra.cluster import Cluster
import numpy
import matplotlib.pyplot as plt
import datetime
import csv

# load subject data from CSV
subjects_index = {}
with open('/home/greg/Documents/subject_species_all.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in reader:
        subjects_index[row[1]] = row[2]

def unix_time(dt):
    epoch = datetime.datetime.utcfromtimestamp(0)
    delta = dt - epoch
    return delta.total_seconds()

def unix_time_millis(dt):
    return long(unix_time(dt) * 1000.0)

cluster = Cluster()
cassandra_session = cluster.connect('serengeti')

total_blanks = 0
num_blanks = {}
session_length = {}
current_sessions = {}
total = 0.
X = []
Y = []
blank_list = []
#2012,6,11
#2013,1,11
for ii,row in enumerate(cassandra_session.execute("select * from classifications where id =1 and created_at>="+str(unix_time_millis(datetime.datetime(2013,6,6))))):
    zooniverse_id = row.zooniverse_id
    if row.user_name == "":
        continue

    id = row.user_ip
    time = row.created_at

    if subjects_index[zooniverse_id]=="blank":
        try:
            num_blanks[id] += 1
        except KeyError:
            num_blanks[id] = 1
        total_blanks += 1
        blank_list.append(1)
    else:
        blank_list.append(0)

    # increment the session length
    try:
        session_length[id] += 1
    except KeyError:
        session_length[id] = 1

    try:
        time_delta = time - current_sessions[id]
        if time_delta.seconds >= 60*30: # note, session length currently set to 60 minutes. Change second number to 10 for a 10 min session
            # store data and reset counters

            # if num_blanks[id]/float(session_length[id]) in [0,0.5,1]:
            #     print " ++ " + str((session_length[id],num_blanks[id]))
            # else:
            if (session_length[id] > 4):#: and (num_blanks[id] != session_length[id]):
                X.append(num_blanks[id]/float(session_length[id]))
                Y.append(session_length[id])
            session_length[id] = 0
            num_blanks[id] = 0



    except KeyError:
        pass

    current_sessions[id] = time
    total += 1

    if ii % 10000 == 0:
        print ii

    if ii >= 400000:
        break

# plt.plot(X,Y,'.')
# plt.hist(X, 50,weights=[0.75 for i in X],histtype='step')
# plt.xlabel("percentage of images per session which are blank")
# plt.ylabel("session length - blank + non-blank")
# print total
# print numpy.mean(Y)
p_blank = total_blanks/total
print p_blank
print numpy.mean(blank_list)
print numpy.median(blank_list)
# print numpy.median(Y)

f, (ax1, ax2) = plt.subplots(2, 1)



X2 = []
Y2 = []

n,bins,patches = plt.hist(X,bins=50)


for l in range(5):
    for y in Y:
        X2.append(numpy.random.binomial(y,p_blank)/float(y))
        Y2.append(y)


X3 = []
Y3 = []

t1 = []
t2 = []

t1_x = []
t2_x = []

for j in range(len(bins[:-1])):
    lb = bins[j]
    ub = bins[j+1]

    in_bin = [y for x,y in zip(X,Y) if (x >= lb) and (x <= ub)]
    in_bin2 = [y for x,y in zip(X2,Y2) if (x >= lb) and (x <= ub)]

    mid_pt = (bins[j]+bins[j+1])/2.
    if (in_bin != []) and (in_bin2 != []):

        diff = numpy.mean(in_bin)-numpy.mean(in_bin2)

        X3.append(mid_pt)
        Y3.append(diff)

    if in_bin != []:
        t1.append(numpy.mean(in_bin))
        t1_x.append(mid_pt)
    if in_bin2 != []:
        t2.append(numpy.mean(in_bin2))
        t2_x.append(mid_pt)

ax1.plot(t1_x,t1,color="blue")
ax1.plot(t2_x,t2,color="green")
ax2.plot(X3,Y3)
plt.show()