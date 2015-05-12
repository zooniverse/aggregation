__author__ = 'ggdhines'
from ouroboros_api import MarkingProject
from agglomerative2 import Agglomerative
from classification import MajorityVote,AllOrNothing
import numpy

class FloatingForests(MarkingProject):
    def __init__(self,date="2015-05-06",pickle_directory="/tmp/"):
        MarkingProject.__init__(self, "plankton", date,experts=["yshish"],pickle_directory=pickle_directory)

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
            markings.append(((x,y),species))

        # assert False
        return markings

project = FloatingForests()
project.__top_users__()
agglomerative = Agglomerative(project)
classifier = AllOrNothing(project,agglomerative)

for subject_id in project.gold_standard_subjects:
    agglomerative.__cluster_subject__(subject_id,gold_standard =True)

print classifier.__fit__(project.gold_standard_subjects,gold_standard=True)
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
