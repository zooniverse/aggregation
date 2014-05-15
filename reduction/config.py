import reduction.adapters.mysql
import reduction.algos
import reduction.project
import configparser


class Config(object):
    parser = configparser.SafeConfigParser()

    def __init__(self, filename):
        self.parser.read(filename)
        self.set_config(self.parser)

    def set_config(self, config):
        print(config)
        self.db = self.set_db(config)
        self.project = self.set_project(config)
        self.algo = self.set_alog(config)

    def __call__(self):
        return self.project(self.db, self.algo)

    def set_db(self, config):
        s = "Database"
        db_type = config.get(s, 'database_type')
        if db_type == 'mongo':
            db = reduction.adpaters.mongo.Mongo
        elif db_type == 'mysql':
            db = reduction.adapters.mysql.Mysql
        else:
            raise Exception("No supported adapter")
        user_cmd = config.get(s, 'users_cmd')
        clicks_cmd = config.get(s, 'clicks_cmd')
        subjects_cmd = config.get(s, 'subjects_cmd')
        conf = dict((key, value) for key, value in config.items(s)
                    if key not in ["clicks_cmd", "users_cmd", "subjects_cmd"])
        return db(user_cmd,
                  clicks_cmd,
                  subjects_cmd,
                  **conf)

    def set_project(self, config):
        s = "Project"
        project_type = config.get(s, 'project_type')

        if project_type == 'PlanetHunters':
            project = reduction.project.PlanetHunters
        else:
            raise Exception("No supported Project Type")
        conf = dict((key, value) for key, value in config.items(s))
        return project(**conf)

    def set_algo(self, config):
        s = 'Algo'

        algo_type = config.get(s, 'algo_type')

        if algo_type == 'lpi':
            algo = reduction.algos.lpi.LPI
        elif algo_type == 'kos':
            algo = reduction.algos.kos.KOS
        else:
            raise Exception("No support Algorithm Type")
        conf = dict((key, value) for key, value in config.items(s))
        return algo(**conf)
