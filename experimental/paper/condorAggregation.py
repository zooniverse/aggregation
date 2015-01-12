__author__ = 'greg'
import aggregation


class CondorIteration(aggregation.AnnotationIteration):
    def __init__(self, classification):
        aggregation.AnnotationIteration(classification, scale=1.875)


class CondorMongo(aggregation.Aggregation):
    def __init__(self, to_skip=[]):
        #["carcassOrScale", "carcass", "other", ""]
        aggregation.Aggregation.__init__(self, "condor", "2014-11-23", ann_iterate=CondorIteration, to_skip=to_skip)