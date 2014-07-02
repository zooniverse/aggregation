from __future__ import print_function
import numpy as np

class Classifier:
    def __init__(self,subjectNodes,userNodes):

        self.subjectNodes = subjectNodes
        self.userNodes = userNodes
        self.alpha = 0.6

    def __classify__(self,attributeList):
        for att in attributeList:
            for subject in self.subjectNodes:
                subject.__shiftFocus__(att)

                #what alpha value would this subject need to get correct positive?

    def __alphaPlot__(self,attributeList):
        p = []
        for alpha in np.arange(0,1.01,0.02):
            print(alpha)
            correct = 0
            total = 0.
            for i,subject in enumerate(self.subjectNodes):
                a1,a2 = subject.__getAlphas__(attributeList)

                if (a1 <= alpha) and (alpha <= a2):
                    correct += 1

                total += 1

                if i == 200:
                    break

            p.append(correct/total)

        return p

    def __rocAnalyze__(self,attributeList):
        inAlpha = []
        exAlpha = []

        for i,subject in enumerate(self.subjectNodes):
            a1,a2 = subject.__getAlphas__(attributeList)

            inAlpha.append(a1)
            exAlpha.append(a2)




        inAlpha.sort(reverse=True)
        exAlpha.sort(reverse=True)

        pEnumerated = list(enumerate(inAlpha))

        nEnumerated = list(enumerate(exAlpha))

        lx = [1]
        ly = [1]

        for alpha in np.arange(0,1.01,0.01):
            found = False
            for pIndex,pAlpha in pEnumerated:
                if pAlpha <= alpha:
                    found = True
                    break
            assert(found)

            pPercent = pIndex/float(len(inAlpha))

            found = False
            for nIndex,nAlpha in nEnumerated:
                if nAlpha <= alpha:
                    found = True
                    break
            assert(found)

            nPercent = nIndex/float(len(exAlpha))

            lx.append(pPercent)
            ly.append(nPercent)

        return lx,ly