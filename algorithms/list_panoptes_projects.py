__author__ = 'greg'
import os
import yaml
import psycopg2

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

environment = "staging"

try:
    database_file = open("config/database.yml")
except IOError:
    database_file = open(base_directory+"/Databases/database.yml")

database_details = yaml.load(database_file)

database = database_details[environment]["database"]
username = database_details[environment]["username"]
password = database_details[environment]["password"]
host = database_details[environment]["host"]

# try connecting to the db
details = "dbname='"+database+"' user='"+ username+ "' host='"+ host + "' password='"+password+"'"
postgres_cursor = None
for i in range(20):
    try:
        postgres_session = psycopg2.connect(details)
        postgres_cursor = postgres_session.cursor()
        break
    except psycopg2.OperationalError as e:
        pass

if postgres_cursor is None:
    raise

select = "SELECT * from projects"
postgres_cursor.execute(select)

for p in postgres_cursor.fetchall():
    project_name = p[2]
    if project_name is None:
        continue
    if "season" in project_name.lower():
        print p