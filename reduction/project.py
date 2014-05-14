from reduction.graph import Graph


class Project(object):

    def __init__(self, config):
        self.config = config


class BinaryQuestionProject(Project):

    def __init__(self, config):
        super().__init__(config)


class OneDimensionalMarkingProject(BinaryQuestionProject):

    def __init__(self, **kwargs):
        self.length = kwargs['x']

    def __call__(self, db, algo):
        g = self.build_graph(db)
        return algo(g)

    def build_graph(self, db, algo):
        users = db.users()
        subjects = db.users()
        annotations = db.users()


class TwoDimensionalMarkingProject(BinaryQuestionProject):

    def __init__(self, **kwargs):
        self.x = kwargs['x']
        self.y = kwargs['y']

    def __call__(self, db, algo):
        graph = self.build_graph(db)
        return algo(graph)

    def build_graph(self, db):
        g = Graph()
        return g
