#!/usr/bin/env python
__author__ = 'greg'
import pymongo
import matplotlib.pyplot as plt
import numpy
import math
import random
import csv
import scipy.stats as stats
# import numpy as np
import pylab as pl
import scipy.special as ss
from scipy.stats import beta as beta_func

# load subject data from CSV
subjects_index = {}
with open('/home/greg/Documents/subject_species_all.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in reader:
        subjects_index[row[1]] = row[2]

# connect to the mongo server
client = pymongo.MongoClient()
db = client['serengeti_2015-02-22']
classification_collection = db["serengeti_classifications"]
subject_collection = db["serengeti_subjects"]
user_collection = db["serengeti_users"]

# for storing the time stamp of the last classification made by each user
current_sessions = {}
# how many subjects each user has classified during the current session
session_length = {}
# how many blank each user has classified during the current session
num_blanks = {}

# X - percentage of blanks per session
X = []
# Y - session length
Y = []

max_classifications = 0
total_blanks = 0
total = 0.
# scan through *ALL* classifications (add a .skip or .limit to look at subset)
#{"created_at":{"$gte":datetime.datetime(2012,12,10)}}
for ii, classification in enumerate(classification_collection.find().skip(0).limit(500000)):
    if ii % 10000 == 0:
        print ii


    # use the ip address to identify the user - that way we track non-logged in users as well
    if "user_name" not in classification:
        continue
    #print classification["user_name"]
    # id =  classification["user_ip"]
    id_ =  classification["user_name"]
    # what time was the classification made at?
    time = classification["updated_at"]

    # skip tutorial classifications
    if "tutorial" in classification:
        continue

    total += 1

    # get the id for the subject and find out whether it was retired as a blank
    zoonvierse_id= classification["subjects"][0]["zooniverse_id"]
    # you can include blank consensus as well but I found those to be pretty error prone
    #subject = subject_collection.find_one({"zooniverse_id":zoonvierse_id})
    #if subject["metadata"]["retire_reason"] in ["blank"]:
    if subjects_index[zoonvierse_id]=="blank":
        try:
            num_blanks[id_] += 1
        except KeyError:
            num_blanks[id] = 1
        total_blanks += 1

    # increment the session length
    try:
        session_length[id] += 1
    except KeyError:
        session_length[id] = 1

    # how long has it been since that person made their previous classification
    # if it has been 10 minutes or more, count as a new session
    try:
        time_delta = time - current_sessions[id]
        if time_delta.seconds >= 60*30: # note, session length currently set to 60 minutes. Change second number to 10 for a 10 min session
            # store data and reset counters

            # if num_blanks[id]/float(session_length[id]) in [0,0.5,1]:
            #     print " ++ " + str((session_length[id],num_blanks[id]))
            # else:
            if (session_length[id] > 4):#: and (num_blanks[id] != session_length[id]):
                max_classifications = max(max_classifications,session_length[id])
                X.append(num_blanks[id]/float(session_length[id]))
                Y.append(session_length[id])
            session_length[id] = 0
            num_blanks[id] = 0



    except KeyError:
        pass

    current_sessions[id] = time

# basic plot of results
plt.plot(X,Y,'.')
plt.hist(X, 50,weights=[0.75 for i in X],histtype='step')
plt.xlabel("percentage of images per session which are blank")
plt.ylabel("session length - blank + non-blank")
print numpy.mean(Y)
# print numpy.median(Y)
plt.show()

X2 = []

n,bins,patches = plt.hist(X,bins=50)



for y in Y:
    X2.append(numpy.random.binomial(y,0.704)/float(y))

n2,bins,patches = plt.hist(X2,bins=bins)

X3 = []
Y3 = []

for j in range(len(bins[:-1])):
    mid_pt = (bins[j]+bins[j+1])/2.
    diff = n2[j]-n[j]

    X3.append(mid_pt)
    Y3.append(diff)

plt.plot(X3,Y3)
plt.show()

print bins
print len(bins)
assert False

# R = [1.4135802469135803, 1.4135802469135803, 1.4135802469135803, 1.4135802469135803, 1.4135802469135803, 1.4135802469135803, 1.4135802469135803, 1.4817073170731707, 1.536144578313253, 1.536144578313253, 1.7613636363636365, 1.8022598870056497, 1.8022598870056497, 1.8764044943820224, 2.2842105263157895, 2.5282051282051281, 2.5282051282051281, 3.540192926045016, 3.873065015479876, 4.6408839779005522, 5.171361502347418, 6.0, 6.7172284644194757, 7.6954314720812187, 9.5975794251134641, 9.3083807973962571, 11.430453879941435, 14.360606060606061, 17.403136064744562, 21.211841599384851, 26.183418512208974, 32.933232823605429, 40.266754704522945, 45.811599467247852, 53.044514168055692, 57.076967213114756, 58.346408485797681, 58.841384330333511, 58.48618993567915, 57.620581493545664, 56.608156131663272, 56.136938483547922, 55.437664889138368, 55.052423838114301, 54.667822926990418, 54.479230600613768, 54.375, 54.317489768076399, 54.316108261486413, 54.316108261486413, 52.148479147068045]
# R=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 5.0, 5.0, 6.0, 7.0, 6.0, 6.0, 9.0, 11.0, 14.0, 17.0, 22.0, 28.0, 32.0, 38.0, 43.0, 44.0, 45.0, 45.0, 44.0, 43.0, 42.0, 41.0, 41.0, 41.0, 40.0, 40.0, 40.0, 40.0, 40.0, 38.0]
# print total_blanks/float(total)
# R = [1.1774332909783989, None, None, None, None, None, 8.21875, 7.0, 6.1904761904761907, 12.666666666666666, 5.1960784313725492, 9.2083333333333339, 14.333333333333334, 15.0, 7.9140049140049138, 11.66225165562914, 17.68, 4.2898089171974521, 13.257396449704142, 10.490328820116055, 6.9649305555555552, 10.449305847707194, 13.388640714741545, 15.751083293211362, 25.055718475073313, 6.8124303471924561, 27.622029633771316, 24.052072263549416, 29.254463840399001, 29.541761304976255, 35.742133565874575, 49.413945253983385, 65.031529345144193, 64.128243307812824, 95.139953374777974, 94.371087753517187, 89.686049876123477, 90.663386929045672, 74.315022400332694, 53.529303946415638, 36.508407460545193, 35.073549516049653, 21.798761609907121, 19.400917010545623, 15.37729928586886, 15.973203157457416, 15.814027630180659, 18.172413793103448, 23.5625, 36.25, 2.2950964822284372]
# X2 = []
# Y2 = []
#
# for k,rr in zip(numpy.arange(0,1.01,0.02),R):
#     t = []
#
#     for x,y in zip(X,Y):
#         if math.fabs(x - k) < 0.01:
#             t.append(y)
#
#     if (t != []) and (rr is not None):
#         X2.append(k)
#         Y2.append(numpy.mean(t) - rr)
#
# plt.plot(X2,Y2,'o-')
# plt.show()

# n,bins,patches = plt.hist(Y,bins=50)
# Y2 = [max(random.expovariate(1/numpy.mean(Y)),1) for i in Y]
# plt.hist(Y2,bins=50,histtype='step')
#
# Y2 = [y/float(max(Y)) for y in Y]
# mean=numpy.mean(Y2)
# var=numpy.var(Y2,ddof=1)
# print var < (mean*(1-mean))
# alpha = mean*(mean*(1-mean)/var-1)
# beta = (1-mean)*(mean*(1-mean)/var-1)
# print alpha,beta,max(Y)
# Y3 = [max(random.betavariate(alpha,beta)*max(Y),1) for y in Y]
# plt.hist(Y3,bins=50,histtype='step')
# plt.show()


# XY.sort(key = lambda x:x[0])
#
# # create bins for a range of different X values between 0 and 1 - incrementing by 0.025
# # assume that within bin, the distribution of values is independent of X
# bins_endpts = numpy.arange(0,1.01,0.025)
# bins = {(bins_endpts[i],bins_endpts[i+1]):[] for i in range(len(bins_endpts)-1)}
# exp_lambda = []
# X2 = []
# error = []
# v = {}
# confidence = {}
# # store y values in appropriate bins
# for x,y in XY:
#     if x == 1:
#         bins[(bins_endpts[-2],bins_endpts[-1])].append(y)
#     else:
#         for lb,ub in bins.keys():
#             if (lb <= x) and (x < ub):
#                 bins[(lb,ub)].append(y/float(max_classifications))
#                 break
#
# X = []
# Y = []
# Y2 = []
#
# for i in range(len(bins_endpts)-1):
#     lb = bins_endpts[i]
#     ub = bins_endpts[i+1]
#
#     t= bins[(lb,ub)]
#     if len(t) > 0:
#         X.append((lb+ub)/2.)
#         Y.append(numpy.mean(t))
#         Y2.append(numpy.median(t))
#
# plt.plot(X,Y,color="red")
# plt.plot(X,Y2,color="blue")
# plt.yscale("log")
# plt.show()
#
# def beta_(a, b, mew):
#     e1 = ss.gamma(a + b)
#     e2 = ss.gamma(a)
#     e3 = ss.gamma(b)
#     e4 = mew ** (a - 1)
#     e5 = (1 - mew) ** (b - 1)
#     return (e1/(e2*e3)) * e4 * e5
#
# def plot_beta(a, b):
#     Ly = []
#     Lx = []
#     mews = numpy.mgrid[0:1:100j]
#     for mew in mews:
#         Lx.append(mew)
#         Ly.append(beta_(a, b, mew))
#     pl.plot(Lx, Ly, label="a=%f, b=%f" %(a,b))
#
# Y2 = [y/float(max_classifications) for y in Y]
# mean=numpy.mean(Y2)
# var=numpy.var(Y2,ddof=1)
# print var < (mean*(1-mean))
# alpha = mean*(mean*(1-mean)/var-1)
# beta = (1-mean)*(mean*(1-mean)/var-1)
# # plot_beta(alpha,beta)
# # pl.show()
#
# for i in range(len(bins_endpts)-1):
#     lb = bins_endpts[i]
#     ub = bins_endpts[i+1]
#
#     t= len(bins[(lb,ub)])
#     if t > 0:
#     # print t
#         v = sorted(bins[(lb,ub)])
#         mid = len(v)/2.-0.5
#         lc = int(mid-3.6)
#         uc = int(mid+3.6)
#         #lc = int(round(len(v)/2.-1.96*math.sqrt(len(v))/2.)-1)
#         #uc = int(round(1+len(v)/2.+1.96*math.sqrt(len(v))/2.)-1)
#         #print numpy.median(bins[(lb,ub)])
#         print v[lc],v[uc]
#     # r = beta_func.rvs(alpha, beta, size=t)
#     # print numpy.median(r)
#     # print
#
# assert False
# # def main():
# #     plot_beta(0.1, 0.1)
# #     plot_beta(1, 1)
# #     plot_beta(2, 3)
# #     plot_beta(8, 4)
# #     pl.xlim(0.0, 1.0)
# #     pl.ylim(0.0, 3.0)
# #     pl.legend()
# #     pl.show()
#
# # use an exponential distribution to approximate the values in each bin
# # estimate lambda - the one param for an exponential distribution
# for i in range(len(bins_endpts)-1):
#     lb = bins_endpts[i]
#     ub = bins_endpts[i+1]
#
#     if len(bins[(lb,ub)]) >= 5:
#         values = bins[(lb,ub)]
#         mean=numpy.mean(values)
#         var=numpy.var(values,ddof=1)
#
#         # print len(values), var >= (mean*(1-mean))
#
#         exp_lambda.append(1/mean)
#         v[lb] = mean
#         lambd = 1/mean
#
#         confidence[lb] = (lambd * (1-1.96/math.sqrt(len(values))), lambd * (1+1.96/math.sqrt(len(values))))
#         alpha = mean*(mean*(1-mean)/var-1)
#         beta = (1-mean)*(mean*(1-mean)/var-1)
#
#         if var < (mean*(1-mean)):
#             print alpha,beta
#             plot_beta(alpha,beta)
#
#         X2.append((ub+lb)/2.)
#         error.append((1/mean)*1.96/math.sqrt(len(values)))
#
# pl.show()
# assert False
# print "variance is " + str(numpy.var(v.values()))
#
# # plot out the lambda values - with confidence regions - for each bin
# plt.errorbar(X2,exp_lambda,yerr=error)
# plt.xlabel("percentage of images per session which are blank")
# plt.ylabel("estimate of lambda param for exponential distribution")
# plt.yscale("log")
# plt.show()
# plt.close()
#
#
# # plot a cumulative distribution of values as well
# plt.hist(X, 50, normed=1,histtype='step', cumulative=True)
# plt.xlabel("percentage of images per session which are blank")
# plt.ylabel("cumulative distribution")
# plt.yscale("log")
# plt.show()
#
# X = []
# Y = []
#
# v2 = {}
#
# for i in range(len(bins_endpts)-1):
#     lb = bins_endpts[i]
#     ub = bins_endpts[i+1]
#
#     if lb in v:
#
#         for j in range(200):
#             X.append(random.uniform(lb,ub))
#             Y.append(random.expovariate(random.uniform(confidence[lb][0],confidence[lb][1])))
#
#             # Y.append(numpy.random.exponential(v[lb]))
#
# plt.plot(X,Y,'.')
#
# XY = zip(X,Y)
# bins2 = {(bins_endpts[i],bins_endpts[i+1]):[] for i in range(len(bins_endpts)-1)}
#
# # store y values in appropriate bins
# for x,y in XY:
#     if x == 1:
#         bins2[(bins_endpts[-2],bins_endpts[-1])].append(y)
#     else:
#         for lb,ub in bins2.keys():
#             if (lb <= x) and (x < ub):
#                 bins2[(lb,ub)].append(y)
#                 break
#
# for i in range(len(bins_endpts)-1):
#     lb = bins_endpts[i]
#     ub = bins_endpts[i+1]
#
#     if len(bins2[(lb,ub)]) >= 5:
#         values = bins2[(lb,ub)]
#         mean=numpy.mean(values)
#
#         v2[lb] = mean
# print "variance is now " + str(numpy.var(v2.values()))
# plt.show()
