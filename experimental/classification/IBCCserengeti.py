__author__ = 'greghines'
from Container import Container

class IBCC(Container):
    def __init__(self):
        self.photos = []
        self.users = []

    def __subjectExists__(self,subjectName):