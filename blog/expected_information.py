import matplotlib
#matplotlib.use('WXAgg')
import matplotlib.pyplot as plt
import numpy as np

num_classifications = 30
info =  [[] for i in range(num_classifications)]
info_gain = [[] for i in range(num_classifications)]
correctness = [[] for i in range(num_classifications)]

def shannon(p):
    #this will propperly treat p=1 and p=0 as values of 0
    return np.nansum(-p * np.log2(p))

def confusion_matrix(truth):
    #construct a confusion matrix that use truth to indicate the "true positive" rates for all responses
    #"false positives" are evenly split between remaining choices
    #"true positive" goes down the main diag
    N = len(truth)
    M = np.zeros([N,N])
    for ndx,n in enumerate(truth):
        other = (1 - n)/(N - 1.0)
        M[:,ndx] = other
        M[ndx,ndx] = n
    return M

def weighted_choice(weights):
    #return a random index based on the weights entered (don't need to be normalized)
    rnd = np.random.uniform(0,1) * sum(weights)
    for i, w in enumerate(weights):
        rnd -= w
        if rnd < 0:
            return i

def clip(x, min_val=0, max_val=1):
    #make probability values stay in a sane range
    x[x>max_val] = max_val
    x[x<min_val] = min_val
    return x

N=3
M = confusion_matrix([.55]*N)

for j in range(5000):
    P = np.array([1.0/N]*N)
    actual_classification = weighted_choice(P)
    votes = np.array([], dtype='int')
    for i in range(num_classifications):
        old_shannon = shannon(P)
        prob_classification = np.dot(M, P)
        report = weighted_choice(M[:,actual_classification])
        votes = np.append(votes, report)

        for n in range(N):
            P[n] = P[n] * M[report, n] / prob_classification[report]
        assert np.isclose(sum(P), 1.0)

        new_shannon = shannon(P)
        gain = old_shannon-new_shannon
        info[i].append(new_shannon)
        info_gain[i].append(gain)
        #check if the correct answer is in the majority
        vote_count = np.bincount(votes, minlength=N)
        if np.argmax(vote_count)==actual_classification:
            correctness[i].append(1)
        else:
            correctness[i].append(0)

C = [np.mean(g) for g in correctness]
S = [np.mean(g) for g in info]
G = [np.mean(g) for g in info_gain]
plt.figure(1, figsize=(15,15))
plt.subplot(221)
plt.hlines(0.95, 0, num_classifications, linestyles='dashed', colors='k')
plt.plot(range(1, num_classifications + 1), C)
plt.ylim(0,1)
plt.xlabel("# classifications")
plt.ylabel("Correctness")
plt.subplot(222)
plt.plot(range(1, num_classifications + 1), S)
plt.xlabel("# classifications")
plt.ylabel("Bits")
plt.subplot(223)
plt.plot(range(1, num_classifications + 1), G)
plt.xlabel("# classifications")
plt.ylabel("Delta Bits")
plt.tight_layout()
plt.show()
