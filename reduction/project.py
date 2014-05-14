class Project(object):

    def __init__(self, config):
        self.config = config

class BinaryQuestionProject(Project):

    def __init__(self, config):
        super().__init__(config)

class OneDimensionalMarkingProject(BinaryQuestionProject):

    def __init__(self, config):
        super().__init__(config)

class TwoDimensionalMarkingProject(BinaryQuestionProject):

    def __init__(self, config):
        super().__init__(config)


