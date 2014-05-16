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
        print("Build Graph")
        g = Graph()
        print("Adding Tasks")
        self.add_graph_tasks(g, db)
        print("Adding Workers and Answers")
        self.add_graph_workers_and_answers(g, db)
        return g

    def add_graph_tasks(self, g, db):
        for task_id, kind in db.subjects():
            task_id = "lc" + str(task_id)
            if kind == 'candidate':
                g.add_task(task_id)
            elif kind == 'planet' or kind == 'simulation':
                g.add_gold_task(task_id, 1)

    def add_graph_workers_and_answers(self, g, db):
        for worker_id, light_curve_id, answer_id, session_id in db.clicks():
            light_curve_id = "lc" + str(light_curve_id)

            if light_curve_id in g:
                if not worker_id:
                    worker_id = session_id

                worker_id = "w" + str(worker_id)

                if worker_id not in g:
                    g.add_worker(worker_id)

                if answer_id == 9:
                    g.add_answer(worker_id, light_curve_id, 1)
                else:
                    g.add_answer(worker_id, light_curve_id, -1)
