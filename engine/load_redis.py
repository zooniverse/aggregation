from redis import Redis
import yaml

def configure_redis(environment):
    with open("config/redis.yml") as redis_file:
        options = yaml.load(redis_file)
        options_for_env = options[environment]
        return Redis(host=options_for_env['host'],
                     port=options_for_env['port'],
                     db=options_for_env['db'])