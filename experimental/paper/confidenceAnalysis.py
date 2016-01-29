#!/usr/bin/env python
__author__ = 'greg'
from scipy.stats import ks_2samp
import cPickle as pickle
import os.path
import matplotlib.pyplot as plt
import math

# for Greg - which computer am I on?
if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
    code_directory = base_directory + "/github"
elif os.path.exists("/Users/greg"):
    base_directory = "/Users/greg"
    code_directory = base_directory + "/Code"

else:
    base_directory = "/home/greg"
    code_directory = base_directory + "/github"

sigValues = []
size = []
correctness = []

results = {}

for jj in range(100):
    fname = base_directory+"/Databases/serengeti/users/"+str(jj)+".pickle"
    if os.path.isfile(fname):
        (correct_blanks,false_blanks) = pickle.load(open(fname,"rb"))

        correct_blanks = [c for c in correct_blanks if c < 120]
        false_blanks = [f for f in false_blanks if f < 120]

        d,p = ks_2samp(correct_blanks,false_blanks)
        sigValues.append(p)
        results[p] = (correct_blanks,false_blanks)
        size.append(len(false_blanks))
        correctness.append(len(correct_blanks)/float(len(correct_blanks+false_blanks)))
# print len(sigValues)
# print len([s for s in sigValues if s < 0.01])/float(len(sigValues))
# print len([s for s in sigValues if s < 0.001])/float(len(sigValues))

# plt.plot(sigValues,correctness,'.')
# plt.xlabel("p-value")
# plt.ylabel("Overall accuracy")
# plt.show()

# n, bins, patches = plt.hist(sigValues, 70, normed=1,histtype='step', cumulative=True)
# plt.xlabel("p-value")
# plt.ylabel("Cumulative Probability")
# plt.show()
#
#
#
# plt.plot(size,sigValues,'.')
# plt.xlabel("p-value")
# plt.ylabel("Number of Incorrect Classifications")
# plt.show()
#
keys = sorted(results.keys())
minp = keys[0]
correct,incorrect = results[minp]

times = [(t,1) for t in correct]
times.extend([(t,0) for t in incorrect])
times.sort(key = lambda x:x[0])
#print times
y = []
correcty = []
incorrecty = []
#print len(incorrect)
offset = 4
#print correct
#print times
#times = list(set(times))
for ii,(t0,a) in enumerate(times):
    y1 = sum([1/float(math.fabs(t-t0)+offset) for (t,b) in times if b == 1])
    y2 = sum([1/float(math.fabs(t-t0)+offset) for (t,b) in times])
    #print [1/float(math.fabs(t-t0)+offset) for t in times]
    print [1/float(math.fabs(t-t0)+offset) for (t,b) in times if b == 1]
    print [1/float(math.fabs(t-t0)+offset) for (t,b) in times]

    f = times[:ii]
    f.extend(times[ii+1:])
    #print 1/float(offset),sum([1/float(math.fabs(t-t0)+offset) for t in f])
    print t0,(y1,y2,y1/y2)
    print
    y.append(y1/y2)

    if t0 in correct:
        correcty.append(y)
    else:
        incorrecty.append(y)

plt.plot(zip(*times)[0],y,)
#print sorted(correct)
#print sorted(incorrect)
plt.xlabel("Time to classify single subject")
plt.ylabel("Probability of correct classification")
#plt.plot(correct,correcty,'o',color="red")
print minp
plt.show()

print len(incorrect)