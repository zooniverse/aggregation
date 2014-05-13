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
    def __init__(self, gt_id):
        super().__init__(gt_id, p=1)

class NodeSet(object):
    def __init__(self, graph, type):
        self._nodes = set()
        self._parent = graph
        self._type = type

    def add_node(self, node):
        self._nodes.add(node)

    def contains(self, n):
        return self._type(n) in self._nodes

    def neighbors_and_edges(self, id, condition):
        return [(self._parent[id][n],self._parent.node[n]) for n 
                    in self._parent.neighbors_iter(id)
                    if condition(n)]

    def __iter__(self, condition=lambda n: True):
        for n in self._nodes:
            attr = self._parent.node[n]
            neighbors = self.neighbors_and_edges(n, condition)
            degree = self._parent.degree(n)
            yield n,degree,attr,neighbors

class WorkerSet(NodeSet):
    def cond(self, n):
        return type(n) is Task
    
    def __iter__(self):
        for n,degree,attr,neighbors in super().__iter__(condition=self.cond):
            neighbor_degree = len(neighbors)
            excluded = degree - neighbor_degree
            degree = degree,excluded,neighbor_degree
            yield n,degree,attr,neighbors

class Graph(object):
    def __init__(self):
        self._graph = nx.Graph()
        self._workers = WorkerSet(self._graph, Worker)
        self._tasks = NodeSet(self._graph, Task)
        self._gold_tasks = NodeSet(self._graph, GoldTask)

    def _add_node(self, n, attributes):
        self._graph.add_node(n)
        for k,v in attributes.items():
            self._graph[n][k] = v 
        if type(n) is Worker:
            self._workers.add_node(n)
        elif type(n) is Task:
            self._tasks.add_node(n)
        elif type(n) is GoldTask:
            self._gold_tasks.add_node(n)

    def add_worker(self, worker_id, attributes={}):
        self._add_node(Worker(worker_id), attributes={})

    def add_task(self, task_id, attributes={}):
        self._add_node(Task(task_id), attributes)

    def add_gold_task(self, gold_task_id, attributes={}):
        self._add_node(GoldTask(gold_task_id), attributes)

    def add_answer(self, worker_id, task_id, answer, attributes={}):
        if self._tasks.contains(task_id):
            task = Task(task_id)
        elif self._gold_tasks.contains(task_id):
            task = GoldTask(task_id)
        else:
            task = None

        worker = Worker(worker_id)

        print(task)
        self._graph.add_edge(worker, task, answer=answer)

        for k,v in attributes.items():
            if k != 'answer':
                self._graph.edge[worker][task][k] = v 

    def workers(self):
        return iter(self._workers)

    def tasks(self):
        return iter(self._tasks)

    def gold_tasks(self):
        return iter(self._gold_tasks)

