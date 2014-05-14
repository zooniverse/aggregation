import reduction.graph as rg
import unittest

class TestNode(unittest.TestCase):
    def setUp(self):
        self.node = rg.Node('id', 0)

    def test_properties(self):
        #make sure properties are set
        self.assertEqual(self.node.id, 'id')
        self.assertEqual(self.node.p, 0)

    def test_hash(self):
        #the object's hash should be the same as the hash of the id property
        self.assertEqual(hash(self.node), hash('id'))
        self.assertEqual(hash(self.node), hash(self.node.id))

    def test_eq(self):
        #the object should be equal to other objects with the same id
        self.assertTrue(self.node == rg.Node('id', 1))
        self.assertTrue(self.node == rg.Node('id', 0))

class TestTask(unittest.TestCase):
    def setUp(self):
        self.task = rg.Task('id')

    def test_properties(self):
        #it should have a p of 0
        self.assertEqual(self.task.p, 0)

class TestWorker(unittest.TestCase):
    def setUp(self):
        self.worker = rg.Worker('id')

    def test_properties(self):
        # should have a p of 0
        self.assertEqual(self.worker.p, 0)

class TestGoldTaask(unittest.TestCase):
    def setUp(self):
        self.goldtask = rg.GoldTask('id')

    def test_properties(self):
        # should have a p of 1
        self.assertEqual(self.goldtask.p, 1)

class TestNodeSet(unittest.TestCase):
    def setUp(self):
        self.graph = rg.Graph()
        self.node_set = self.graph._tasks

        for t in range(3):
            self.graph.add_task("t" + str(t))

        for w in range(3):
            self.graph.add_worker("w" + str(w))
            for t in range(3):
                self.graph.add_answer("w" + str(w), "t" + str(t), -1)

    def test_add_node(self):
        # should only add a node if it is of the same type 
        node = rg.Task('id')
        self.node_set.add_node(node)
        self.assertTrue(node in self.node_set._nodes)

        other_node = rg.GoldTask('id1')
        self.node_set.add_node(other_node)
        self.assertFalse(other_node in self.node_set._nodes)

    def test_contains(self):
        # should return true is the node id is the set
        node = rg.Task('id')
        self.node_set.add_node(node)
        self.assertTrue('id' in self.node_set)

    def test_neighbors_and_edges(self):
        # should return a list of tuples of neighbors and edges meeting a condition
        node = rg.Task('t1')
        neighbors = self.node_set.neighbors_and_edges(node, lambda t: True)
        self.assertEqual(len(neighbors), 3)

        neighbors = self.node_set.neighbors_and_edges(node, lambda t: False)
        self.assertEqual(len(neighbors), 0)

    def test_iter(self):
        # it should iterate through all nodes in the set
        tasks = [t for t in iter(self.node_set)]
        
        self.assertEqual(len(tasks), 3)
        self.assertEqual(len(tasks[0]), 4)
        self.assertEqual(tasks[0][1], self.graph._graph.degree(tasks[0][0]))

class TestWorkerSet(TestNodeSet):
    def setUp(self):
        super().setUp()
        self.worker_node_set = self.graph._workers

    def test_iter(self):
        # it should return the degree excluded degree and included degree

        workers = [w for w in iter(self.worker_node_set)]
        self.assertEqual(workers[0][1], (3,0,3))

class TestGraph(unittest.TestCase):
    def setUp(self):
        self.graph = rg.Graph()

    def test_add_worker(self):
        self.graph.add_worker('w1')
        self.assertEqual(len([w for w in self.graph.workers()]), 1)

    def test_add_task(self):
        self.graph.add_task('t1')
        self.assertEqual(len([t for t in self.graph.tasks()]), 1)

    def test_add_gold_task(self):
        self.graph.add_gold_task('w1')
        self.assertEqual(len([t for t in self.graph.gold_tasks()]), 1)

    def test_add_answer(self):
        w = rg.Worker('w1')
        t = rg.Task('t1')
        self.graph.add_worker('w1')
        self.graph.add_task('t1')
        self.graph.add_answer('w1', 't1', -1)
        self.assertEqual(self.graph._graph.edge[w][t], {'answer' : -1})

if __name__ == '__main__':
    unittest.main()

