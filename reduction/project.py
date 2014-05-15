from reduction.graph import Graph


class Question(object):

    def __init__(self, **kwargs):
        self._config = kwargs


class MarkingQuestion(Question):

    def __init__(self, **kwargs):
        print('here')

    def build_graph(self):
        print('here')


class PlanetHunters(Question):

    def __init__(self, **kwargs):
        self.config = kwargs

    def __call__(self, db, algo):
        graph = self.build_graph(db)
        return algo(graph)

    def build_graph(self, db):
        g = Graph()
        self.add_graph_workers(g, db)
        self.add_graph_tasks(g, db)
        self.add_graph_answer(g, db)
        return g

    def add_graph_workers(self, g, db):
        for worker_id, session_id in db.workers():
            if id:
                g.add_worker(worker_id)
            elif session_id:
                g.add_worker(session_id)

    def add_graph_tasks(self, g, db):
        for task_id, kind in db.subjects():
            if kind == 'candidate':
                g.add_task(task_id)
            elif kind == 'planet' or kind == 'simulation':
                g.add_gold_task(task_id)

    def add_graph_answers(self, g, db):
        for worker_id, light_curve_id, answer_id in db.clicks():
            if answer_id == 9:
                g.add_answer(worker_id, light_curve_id, 1)
            else:
                g.add_answer(worker_id, light_curve_id, -1)
