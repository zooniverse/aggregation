__author__ = 'ggdhines'
from ouroboros_api import MarkingProject
from agglomerative import Agglomerative
from classification import MajorityVote,IBCC
import numpy
import matplotlib.pyplot as plt

class PlanktonPortal(MarkingProject):
    def __init__(self,date="2015-05-08"):
        MarkingProject.__init__(self, "plankton", date,experts=["yshish"])

    def __annotations_to_markings__(self,annotations):
        """
        This is the main function projects will have to override - given a set of annotations, we need to return the list
        of all markings in that annotation
        """
        markings = []
        for ann in annotations:
            if "finished_at" in ann:
                break
            x_pts=[]
            y_pts=[]
            for point in ["p0","p1","p2","p3"]:
                x_pts.append(float(ann[point][0]))
                y_pts.append(float(ann[point][1]))

            x = numpy.mean(x_pts)
            y = numpy.mean(y_pts)

            try:
                species = ann["species"]
            except KeyError:
                continue

            if species == "":
                continue
            markings.append(((x,y),species))

        # assert False
        return markings

project = PlanktonPortal()
# project.__top_users__()
project.__set_gold_standard__(max_subjects=500)

clustering = Agglomerative(project)
classifier = IBCC(project,clustering)

for subject_id in project.gold_standard_subjects:
    print subject_id
    project.__store_markings__(subject_id,max_users=20)
    clustering.__fit__(subject_id)
    clustering.__fit__(subject_id,gold_standard=True)


candidates,results = classifier.__classify__(project.gold_standard_subjects,gold_standard=False)
errors,percentage = project.__evaluate__(candidates,results,clustering)

sorted_percentage = sorted(list(set(percentage)))
c = zip(errors,percentage)

avg_error = [numpy.mean([e for e,p in c if p >= s]) for s in sorted_percentage]
coverage = [sum([1 for e,p in c if p >= s])/float(len(c)) for s in sorted_percentage]
plt.plot(coverage,avg_error)
plt.show()
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
