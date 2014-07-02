__author__ = 'greghines'
from Container import Container


class IBCCcontainer(Container):
    def __init__(self):
        self.subjects = []
        self.users = []
        self.classifications = {}

    def __subjectExists__(self,subjectName):
        return subjectName in self.subjects

    def __addSubject__(self,subjectName):
        assert(not(subjectName in self.subjects))
        self.subjects.append(subjectName)

    def __userExists__(self,userName):
        return userName in self.users

    def __addUser__(self,userName):
        assert(not(self.__userExists__(userName)))
        self.users.append(userName)

    def __classifiedBy__(self,subjectName,userName):
        return (subjectName,userName) in self.classifications

    def __classify__(self,attributes=None):
        pass

    def __newClassification__(self,subjectName,userName,classification):
        self.classifications[(subjectName,userName)] = classification

    def __getClassification__(self,subjectName,userName):
        return self.classifications[(subjectName,userName)]

    def __addAttributeList__(self,subjectName,userName,newAttributes):
        currentAttributes = self.classifications[(subjectName,userName)]
        assert(type(currentAttributes) == list)
        assert(type(newAttributes) == list)
        currentAttributes.extend(newAttributes)
        self.classifications[(subjectName,userName)] = currentAttributes