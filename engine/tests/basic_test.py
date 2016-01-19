from unittest import TestCase
import engine
from engine import aggregation_api
__author__ = 'ggdhines'


class BasicTest(TestCase):
    def first_test(self):
        project = aggregation_api.AggregationAPI(376,"development")
        self.assertTrue(isinstance(project, aggregation_api.AggregationAPI))