from __future__ import print_function
from survey_aggregation import Survey

class GorongosaSurvey(Survey):
    def __init__(self):
        Survey.__init__(self)

    def __species_annotation__(self,aggregation_so_far,annotation):
        """
        WildCam Gorongosa needs the annotation to be wrapped inside a list - really not sure why
        but seems to work
        :param aggregation_so_far:
        :param annotation:
        :return:
        """
        return Survey.__species_annotation__(self,aggregation_so_far,[annotation])