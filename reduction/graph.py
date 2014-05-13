import networkx as nx

class Graph(object):
    class NodeSet(object):
        def __init__(self, graph):
            self._nodes = []
            self._parent = graph

        def add_node(self, node):
            self._nodes.append(node)

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
        def __iter__(self):
            cond = lambda n: self._parent.node[n]['kind'] == 'task'
            for n,degree,attr,neighbors in super().__iter__(condition=cond):
                neighbor_degree = len(neighbors)
                excluded = degree - neighbor_degree
                degree = degree,excluded,neighbor_degree
                yield n,degree,attr,neighbors

    def __init__(self):
        self._graph = nx.Graph()
        self._workers = self.WorkerSet(self._graph)
        self._tasks = self.NodeSet(self._graph)
        self._gold_tasks = self.NodeSet(self._graph)

    def _add_node(self, id, type, attributes):
        self._graph.add_node(id, kind=type)
        for k,v in attributes.items():
            if k != 'kind':
                self._graph[id][k] = v 

    def add_worker(self, worker_id, attributes={}):
        self._add_node(worker_id, 'worker', attributes={})
        self._workers.add_node(worker_id)

    def add_task(self, task_id, attributes={}):
        self._add_node(task_id, 'task', attributes)
        self._tasks.add_node(task_id)

    def add_gold_task(self, gold_task_id, attributes={}):
        self._add_node(gold_task_id, 'gold_task', attributes)
        self._gold_tasks.add_node(gold_task_id)

    def add_answer(self, worker_id, task_id, answer, attributes={}):
        self._graph.add_edge(worker_id, task_id, answer=answer)
        for k,v in attributes.items():
            if k != 'answer':
                self._graph.edge[worker_id][task_id][k] = v 

    def workers(self):
        return self._workers.__iter__()

    def tasks(self):
        return self._tasks.__iter__()

    def gold_tasks(self):
        return self._gold_tasks.__iter__()
