import reducer.adapters
import reducer.algos
import reducer.project
import ConfigParser

class Config(object):
    parser = ConfigParser.SafeConfigParser()

    def __init__(self, filename):
        self.set_config(self.parser.read(filename))

    def set_config(self, config):
        self.db = self.set_db(config)
        self.project = self.set_project(config)
        self.algo = self.set_alog(config)

    def __call__(self):
        return self.project(self.db, self.algo)

    def set_db(self, config):
        s = "Databse"
        db_type = config.get(s, 'database')
        if db_type == 'mongo':
            db = reducer.adpaters.mongo.Mongo
        else:
            raise Exception("No supported adapter")
        host = config.get(s, 'host')
        port = config.get(s, 'port')
        user = config.get(s, 'user')
        password = config.get(s, 'pass')
        user_cmd = config.get(s, 'user')
        clicks_cmd = config.get(s, 'clicks')
        return db(user_cmd,
                  clicks_cmd,
                  host=host,
                  port=port,
                  user=user,
                  password=password)

    def set_project(self, config):
        s = "project"
        project_type = config.get(s, 'project_type')

        if project_type == 'binary':
            project = reducer.project.BinaryQuestionProject
        elif project_type == '1-Dimension':
            project = reducer.project.OneDimensionalMarkingProject
        elif project_type == '2-Dimensions':
            project = reducer.project.TwoDimensionalMarkingProject
        else:
            raise Exception("No supported Project Type")
        conf = dict((key, value) for key, value in config.items(s))
        return project(**conf)

    def set_algo(self, config):
        s = 'algo'

        algo_type = config.get(s, 'algo_type')

        if algo_type == 'lpi':
            algo = reducer.algos.lpi.LPI
        elif algo_type == 'kos':
            algo = reducer.algos.kos.KOS
        else:
            raise Exception("No support Algorithm Type")
        conf = dict((key, value) for key, value in config.items(s))
        return algo(**conf)

