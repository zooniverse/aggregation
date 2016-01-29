from ...BaseNodes import BaseUserNode

class UserNode(BaseUserNode):
    def __init__(self):
        self.subjectsViewed = []
        self.classifications = []

    def __vote__(self,subject,attribute):
        subjectIndex = self.subjectsViewed.index(subject)

        if attribute in self.classifications[subjectIndex]:
            return 1
        else:
            return 0


