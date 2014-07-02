from SubjectNodes import  SubjectNode

class MultiClassSubjectNode(SubjectNode):
    def __init__(self):
        SubjectNode.__init__(numClasses=1)

    def __changeClassificationAttributes__(self,attributesList):
        pass
