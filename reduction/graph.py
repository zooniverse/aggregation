import networkx as nx


class Node(object):
    def __init__(self, id, p):
        self.p = p
        self.id = id

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, test):
        return isinstance(test, type(self)) and hash(self) == hash(test)

    def __str__(self):
        return str(self.id) + " " + str(self.p)


class Task(Node):
    def __init__(self, t_id, p=0):
        super().__init__(t_id, p)


class Worker(Node):
    def __init__(self, w_id):
        super().__init__(w_id, 0)


class GoldTask(Task):
    def __init__(self, gt_id, answer=None):
        self.answer = answer
        super().__init__(gt_id, p=1)

    def __str__(self):
        return super().__str__() + " correct answer: " + str(self.answer)


class NodeSet(object):
    def __init__(self, graph, type):
        self._nodes = set()
        self._parent = graph
        self._type = type

    def add_node(self, node):
        if type(node) is self._type:
            self._nodes.add(node)

    def __contains__(self, n):
        return self._type(n) in self._nodes

    def neighbors_and_edges(self, node):
        return [(self._parent[node][n], n) for n
                in self._parent.neighbors_iter(node)]

    def __iter__(self):
        for n in self._nodes:
            attr = self._parent.node[n]
            neighbors = self.neighbors_and_edges(n)
            degree = self._parent.degree(n)
            yield n, degree, attr, neighbors


class WorkerSet(NodeSet):
    def __init__(self, graph):
        super().__init__(graph, Worker)

    def __iter__(self):
        for n, d, a, ns in super().__iter__():
            gold_correct = len([t for e, t in ns
                                if type(t) is GoldTask
                                if e['answer'] == t.answer])
            yield n, d, gold_correct, a, ns


class Graph(object):
    def __init__(self):
        self._graph = nx.Graph()
        self._workers = WorkerSet(self._graph)
        self._tasks = NodeSet(self._graph, Task)
        self._gold_tasks = NodeSet(self._graph, GoldTask)

    def _add_node(self, n, attributes):
        self._graph.add_node(n)
        for k, v in attributes.items():
            self._graph[n][k] = v
        if type(n) is Worker:
            self._workers.add_node(n)
        elif type(n) is Task:
            self._tasks.add_node(n)
        elif type(n) is GoldTask:
            self._gold_tasks.add_node(n)

    def __contains__(self, n):
        return n in self._workers or n in self._tasks or n in self._gold_tasks

    def add_worker(self, worker_id, attributes={}):
        self._add_node(Worker(worker_id), attributes={})

    def add_task(self, task_id, attributes={}):
        self._add_node(Task(task_id), attributes)

    def add_gold_task(self, gold_task_id, answer, attributes={}):
        self._add_node(GoldTask(gold_task_id, answer), attributes)

    def add_answer(self, worker_id, task_id, answer, attributes={}):
        if task_id in self._tasks:
            task = Task(task_id)
        elif task_id in self._gold_tasks:
            task = GoldTask(task_id)
        else:
            task = None

        worker = Worker(worker_id)

        self._graph.add_edge(worker, task, answer=answer)

        for k, v in attributes.items():
            if k != 'answer':
                self._graph.edge[worker][task][k] = v

    def workers(self):
        return iter(self._workers)

    def tasks(self):
        return iter(self._tasks)

    def gold_tasks(self):
        return iter(self._gold_tasks)
