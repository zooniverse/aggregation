__author__ = 'ggdhines'
from ouroboros_api import MarkingProject


class FloatingForests(MarkingProject):
    def __init__(self,date="2015-05-08",pickle_directory="/tmp/"):
        MarkingProject.__init__(self, "plankton", date,experts=["yshish"],pickle_directory=pickle_directory)

    def __annotations_to_markings__(self,annotations):
        """
        This is the main function projects will have to override - given a set of annotations, we need to return the list
        of all markings in that annotation
        """
        print annotations
        assert False
        return []

project = FloatingForests()
project.__set_gold_standard__(limit=100)
project.__classifications__per_gold_subject__()