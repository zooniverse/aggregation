#!/usr/bin/env python
import os, sys, csv
from graphicalEM import graphicalEM
if os.path.exists("/home/ggdhines/github/pyIBCC/python"):
    sys.path.append("/home/ggdhines/github/pyIBCC/python")
else:
    sys.path.append("/Users/greghines/Code/pyIBCC/python")
import ibcc


class graphicalIBCC(graphicalEM):
    def __init__(self):
        graphicalEM.__init__(self)

    def __classify__(self):
        userNames = self.userDict.keys()
        subjectNames = self.subjectDict.keys()

        f = open(self.baseDir+"ibcc/input",'wb')
        for u in self.userDict:
            classifications = self.userDict[u].__getClassifications__()

            for (s,r) in classifications:
                f.write(str(userNames.index(u)) + "," + str(subjectNames.index(s)) + "," + str(r) + "\n")
        f.close()

        #now - write the config file
        f = open(self.baseDir+"ibcc/config.py",'wb')
        f.write("import numpy as np\nscores = np.array([0,1])\n")
        f.write("nScores = len(scores)\n")
        f.write("nClasses = 2\n")
        f.write("inputFile = '"+self.baseDir+"ibcc/input'\n")
        f.write("outputFile =  '"+self.baseDir+"ibcc/output'\n")
        f.write("confMatFile = '"+self.baseDir+"ibcc/confusion'\n")
        #f.write("nu0 = np.array([45.0,55.0])\n")
        f.close()

        ibcc.runIbcc(self.baseDir+"ibcc/config.py")

    def __analyze__(self):
        subjectNames = self.subjectDict.keys()

        total = 0.
        correct = 0
        f = open(self.baseDir+"ibcc/output",'rb')
        for lines in f.readlines():
            total += 1
            words = lines.split(' ')
            subjectIndex = int(float(words[0]))
            probabilities = [float(w) for w in words[1:]]
            assert(len(probabilities) == 2)
            classification = probabilities.index(max(probabilities))
            sNode = self.subjectDict[subjectNames[subjectIndex]]

            if sNode.__correct__(classification=classification):
                correct += 1

        print correct/total


if __name__ == "__main__":
    c = graphicalIBCC()
    c.__readIndividualClassifications__()
    c.__readGoldStandard__()
    c.__setSpecies__(["gazelleThomsons"])
    c.__classify__()
    c.__analyze__()

