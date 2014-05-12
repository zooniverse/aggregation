

class KOS:
    def sigma_task_p(self, task):
        delta = self.tasks[task]['delta']
        weighted_answers = map(lambda edge: self.workers[edge['id']]['p'] * edge['answer'], delta)
        reduce(lambda x,y: x + y, weighted_answers, 0)

    def sigma_worker_p(self, worker):
        delta = self.workers[worker]['delta']
        weighted_answers = map(lambda edge: self.tasks[edge['id']]['p'] * edge['answer'], delta)
        reduce(lambda x,y: x + y, weighted_answers, 0)

    def task_answer(self, task):
        if 0 < sigma_task_p(task):
          {'id': task, 'answer': 1}
        else:
          {'id': task, 'answer': -1}

    def iterative_reduction(k):
        for _ in repeat(None, k):
          map(lambda t: sigma_task_p(t), self.tasks.keys())
          map(lambda w: sigma_worker_p(w), self.worker.keys())
        map(lambda t: task_answer(t), self.tasks.keys())

