import pymysql


class Mysql(object):

    def __init__(self, workers_cmd, clicks_cmd, subjects_cmd, **kwargs):
        host = kwargs.get('host', None) or '127.0.0.1'
        port = kwargs.get('port', None) or 3306
        user = kwargs.get('user', None) or 'root'
        passwd = kwargs.get('password', None) or ''
        db = kwargs.get('name', None) or 'project'

        self.connection = pymysql.connect(host=host,
                                          port=port,
                                          user=user,
                                          passwd=passwd,
                                          db=db)
        self._workers_cmd = workers_cmd
        self._clicks_cmd = clicks_cmd
        self._subjects_cmd = subjects_cmd

    def workers(self):
        self._execute(self._workers_cmd)

    def clicks(self):
        self._execute(self._clicks_cmd)

    def subjects(self):
        self._execute(self._subjects_cmd)

    def _execute(self, cmd):
        cur = self.connection.cursor()

        for row in cur.execute(cmd):
            yield row

        cur.close()
