#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import os
import pymongo
import urllib
import matplotlib.pyplot as plt
import cv2
from cassandra.cluster import Cluster
import cassandra
from cassandra.concurrent import execute_concurrent
import psycopg2
import psycopg2
import numpy as np
from scipy.stats import chi2
import math

client = pymongo.MongoClient()
db = client['serengeti_2015-08-20']
subjects = db["serengeti_subjects"]
classifications = db["serengeti_classifications"]

conn = psycopg2.connect("dbname='serengeti' user='ggdhines' host='localhost' password='apassword'")
cur = conn.cursor()

# cur.execute("drop table serengeti")
# cur.execute("create table serengeti(user_name text,created_at timestamp)")#, PRIMARY KEY (user_name,created_at))")
# cur.execute("create index i1 on serengeti (user_name)")
# cur.execute("create index i2 on serengeti (user_name,created_at)")
#
# for i,c in enumerate(classifications.find().limit(15000000)):
#     if "user_name" in c:
#         user_name = c["user_name"]
#         assert isinstance(user_name,unicode)
#         # print user_name
#         user_name = user_name.replace("'","")
#         cur.execute("insert into serengeti (user_name,created_at) values ('"+user_name+"','"+str(c["created_at"])+"')")
#         if i% 1000 == 0:
#             print i
#conn.commit()

# connect to the mongodb server

cur.execute("select distinct user_name from serengeti")
users = [c[0] for c in cur.fetchall()]

all_f = []

percentiles = [0.01,0.02,0.03,0.04]
percentiles.extend(np.arange(0.05,1,0.05))
percentiles.extend([0.96,0.97,0.98,0.99])

A = {p:[] for p in percentiles}
B = {p:[] for p in percentiles}
C = {p:[] for p in percentiles}

pp = 0.6

percentile_hist = []

lengths = []
lengths_2 = []

X = []
Y = []

for ii,u in enumerate(users):
    # print u
    cur.execute("select created_at from serengeti where user_name = '" + u + "'")

    timestamps = [c[0] for c in cur.fetchall()]

    previous_index = 0

    session_lengths = []
    time_out = []

    for t_index in range(1,len(timestamps)):
        delta_t = timestamps[t_index]-timestamps[t_index-1]

        if delta_t.seconds == 0:
            continue

        if delta_t.seconds > 60*15:
            session_lengths.append(t_index-previous_index)
            previous_index = t_index

    session_lengths.append(len(timestamps)-previous_index)
    if (len(session_lengths) < 100):# or (len(session_lengths) >= 40):
        continue

    # p = np.percentile(np.asarray(session_lengths),50)
    # session_lengths = [s-p for s in session_lengths if s > p]
    # session_lengths.sort()

    # # print len(session_lengths)
    # mean = np.mean(session_lengths)
    # t = [s/mean for s in session_lengths if s/mean < 20]
    # # n,bins,patches = plt.hist(t, 25, normed=1, facecolor='green', alpha=0.5)
    # n,bins = np.histogram(t,40)
    # # plt.close()
    # n_bins = zip(n,bins)
    # threshold = max(n_bins,key = lambda x:x[0])[1]
    #
    #
    # # if ii >= 300:
    # #     assert False
    #
    # mean = np.mean(session_lengths)
    # session_lengths = [s-mean*threshold for s in session_lengths if s > mean*threshold]


    # if mean > 80:
    #     print u

    c = float(max(session_lengths))
    a = float(min(session_lengths))
    # normalized_s = [(s-min_s)/(max_s-min_s) for s in session_lengths]
    # print normalized_s
    x_bar = (np.mean(session_lengths)-a)/(c-a)
    v_bar = np.var(session_lengths,ddof=1)/((c-a)**2)

    # if v_bar >= (x_bar*(1-x_bar)):
    #     print "skipping"
    #     continue
    #
    # alpha = x_bar*(x_bar*(1-x_bar)/v_bar-1)
    # beta = (1-x_bar)*(x_bar*(1-x_bar)/v_bar-1)
    #
    # from scipy.stats import beta as beta_func
    #
    #
    #
    # if min(alpha,beta) > 1:
    #     mode =  (alpha-1)/(alpha+beta-2) * (c-a) + a
    #
    #     session_lengths = [s for s in session_lengths if s >= mode]
    #
    #     # r = beta_func.rvs(alpha, beta)
    #     # fig, ax = plt.subplots(1, 1)
    #     # x = np.linspace(beta_func.ppf(0.01, alpha, beta),beta_func.ppf(0.99, alpha, beta), 100)
    #     # rv = beta_func(alpha, beta)
    #     # ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
    #     # plt.show()
    #     #
    #     # plt.hist(session_lengths,10)
    #     # plt.plot([mode,mode],[0,10])
    #     # plt.show()





    #
    mean = np.mean(session_lengths)
    lengths.extend(session_lengths)
    lengths_2.append(np.mean(session_lengths))

    num_samples = len(session_lengths)
    ub = (2*num_samples*mean)/chi2.ppf(0.025, 2*num_samples)
    lb = (2*num_samples*mean)/chi2.ppf(1-0.025, 2*num_samples)

    # print lb,mean,ub
    for pp in percentiles:
        l_median = -math.log(1-pp)*lb
        median = -math.log(1-pp)*mean
        u_median = -math.log(1-pp)*ub

        A[pp].append(len([1. for s in session_lengths if s <= l_median])/float(num_samples))
        B[pp].append(len([1. for s in session_lengths if s <= median])/float(num_samples))
        C[pp].append(len([1. for s in session_lengths if s <= u_median])/float(num_samples))

print len(A[percentiles[0]])
# print np.mean(A)
# print np.mean(B)
# print np.mean(C)

# lower:
m = [np.mean(A[pp]) for pp in percentiles]
plt.plot(percentiles,m)
m = [np.mean(B[pp]) for pp in percentiles]
plt.plot(percentiles,m)
m = [np.mean(C[pp]) for pp in percentiles]
plt.plot(percentiles,m)

plt.plot([0,1],[0,1],"--",color="black")

plt.show()
print np.mean(lengths),np.median(lengths),np.percentile(lengths,90),np.percentile(lengths,99)
print np.percentile(lengths_2,1),np.mean(lengths_2),np.median(lengths_2),np.percentile(lengths_2,90),np.percentile(lengths_2,99)
plt.hist(lengths_2, 50, normed=1, facecolor='green')
plt.show()