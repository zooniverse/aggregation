__author__ = 'ggdhines'
from ouroboros_api import MarkingProject
from agglomerative import Agglomerative
from classification import MajorityVote,IBCC
import numpy
from expert_classification import ExpertClassification
import matplotlib.pyplot as plt
import cPickle as pickle
import os

#db.plankton_users.find({},{name:1,classification_count:1}).sort({classification_count:-1}).limit(10)

class PlanktonPortal(MarkingProject):
    def __init__(self,date="2015-05-06"):
        MarkingProject.__init__(self, "plankton", date,experts=["yshish"])

    def __classification_to_annotations__(self,classification,):
        """
        This is the main function projects will have to override - given a set of annotations, we need to return the list
        of all markings in that annotation
        """
        (lb_roi,ub_roi) = self.current_roi
        markings = []
        for ann in classification["annotations"]:
            if "finished_at" in ann:
                break
            x_pts=[]
            y_pts=[]
            for point in ["p0","p1","p2","p3"]:
                x_pts.append(float(ann[point][0]))
                y_pts.append(float(ann[point][1]))

            x = numpy.mean(x_pts)
            y = numpy.mean(y_pts)

            if not(MarkingProject.__in_roi__(self,(x,y),lb_roi,ub_roi)):
                continue

            try:
                species = ann["species"]
            except KeyError:
                continue

            if species == "":
                continue
            markings.append(((x,y),species))

        # assert False
        return markings

    def __get_roi__(self,subject_id):


        subject = self.subject_collection.find_one({"zooniverse_id":subject_id})
        cutout = subject["metadata"]["cutout"]
        width = cutout["width"]
        height = cutout["height"]
        lb_roi = [[0,0],[width,0]]
        ub_roi = [[0,height],[width,height]]

        return lb_roi,ub_roi

project = PlanktonPortal()
# project.__top_users__()
#project.__set_subjects__([u'APK00011p5', u'APK0001bw9', u'APK0001dj4', u'APK00019zu', u'APK00018ri', u'APK0001dxl', u'APK0001ana', u'APK0000ppu', u'APK0000dvx', u'APK0000pyd', u'APK00019ol', u'APK00072zo', u'APK0000h5h', u'APK00001fk', u'APK0000a69', u'APK0000km2', u'APK000175z', u'APK00019yw', u'APK0000e39', u'APK0000kga'])
# project.__random_gold_sample__(max_subjects=50)
project.__gold_sample__(["yshish"],["ElisabethB","Damon22","MingMing","elizabeth","JasonJason","rlb66xyz","planetari7","fermor332002","artman40","Quia"],max_subjects=200)

clustering = Agglomerative(project)
# clustering2 = GoldClustering(project)
classifier = ExpertClassification(project,clustering)
#
for subject_id in project.gold_standard_subjects:
    print subject_id
    clustering.__fit__(subject_id,gold_standard=True)
    print clustering.goldResults[subject_id]
    classifier.__classify__([subject_id],True)
    print
    # project.__store_annotations__(subject_id,max_users=20)
    # clustering.__fit__(subject_id)
#     clustering.__fit__(subject_id,gold_standard=True)
#
# # # clustering.__check__()
# #
# # candidates,ridings,results = classifier.__classify__(project.gold_standard_subjects,gold_standard=False)
# errors,percentage = project.__evaluate__(candidates,ridings,results,clustering)
# #
# sorted_percentage = sorted(list(set(percentage)))
# c = zip(errors,percentage)
#
# avg_error = [numpy.mean([e for e,p in c if p >= s]) for s in sorted_percentage]
# coverage = [sum([1 for e,p in c if p >= s])/float(len(c)) for s in sorted_percentage]
# plt.plot(coverage,avg_error)
# plt.show()
# project.__set_gold_standard__(limit=100)
#
# agglomerative = Agglomerative(project)
# majority = MajorityVote(project,agglomerative)
#
# for subject_id in project.gold_standard_subjects:
#     print subject_id
#     project.__store_markings__(subject_id,max_users=20)
#     agglomerative.__cluster_subject__(subject_id)
#
#
# results = majority.__classify__(project.gold_standard_subjects)
# project.__evaluate__(results,agglomerative)
#
