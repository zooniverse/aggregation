import matplotlib
matplotlib.use('WXAgg')
import math
import random
import matplotlib.pyplot as plt
import numpy as np

num_classifications = 20
info_gain = [[] for i in range(num_classifications)]
correctness = [[] for i in range(num_classifications)]

def shannon(p):
    try:
        return -p*math.log(p,2) -(1-p)*math.log(1-p,2)
    except ValueError:
        print p
        raise



# print shannon(prior_probability)

true_positive = 0.75
true_negative = 0.75

print "==----"

for j in range(1000):

    pos_probability = 0.5
    neg_probability = 1 - pos_probability

    if random.uniform(0,1) > pos_probability:
        actual_classication = 0
    else:
        actual_classication = 1

    votes = []

    for i in range(num_classifications):
        old_shannon = shannon(pos_probability)
        report_pos = pos_probability*true_positive + neg_probability*(1-true_negative)
        report_false = pos_probability*(1-true_positive) + neg_probability*true_negative

        # what does the user report?
        if (actual_classication == 1):
            if (random.uniform(0,1) <= true_positive):
                report = 1
            else:
                report = 0
        else:
            if (random.uniform(0,1) <= true_negative):
                report = 0
            else:
                report = 1

        votes.append(report)

        if report == 1:
            # print pos_probability,true_positive,report_pos
            pos_probability = pos_probability*true_positive/report_pos
        else:
            # print pos_probability,1-true_positive,report_false
            pos_probability = pos_probability*(1-true_positive)/report_false

        assert pos_probability > 0
        assert pos_probability < 1
        neg_probability = 1 - pos_probability

        new_shannon = shannon(pos_probability)
        gain = old_shannon-new_shannon

        info_gain[i].append(new_shannon)

        if np.mean(votes) >= 0.5:
            if actual_classication == 1:
                correctness[i].append(1)
            else:
                correctness[i].append(0)
        else:
            if actual_classication == 1:
                correctness[i].append(0)
            else:
                correctness[i].append(1)
    # print actual_classication
    # print votes

Y = [np.mean(g) for g in info_gain]
plt.plot(range(num_classifications),Y)
plt.show()