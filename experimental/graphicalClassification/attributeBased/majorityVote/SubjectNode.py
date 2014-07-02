from __future__ import print_function
__author__ = 'greghines'
from ...BaseNodes import BaseSubjectNode

class SubjectNode(BaseSubjectNode):
    def __init__(self):
        BaseSubjectNode.__init__(self,1)

    def __addUser__(self,user):
        self.user_l.append(user)

    def __getAlphas__(self,attributes):
        inAlpha = [1]
        exAlpha = [0]
        for att in attributes:
            votes = sum([u.__vote__(self,att) for u in self.user_l])
            percent = votes/float(len(self.user_l))
            #print((att,percent))
            if att in self.goldStandard:
                inAlpha.append(percent)
            else:
                exAlpha.append(percent)

        return max(exAlpha),min(inAlpha)