__author__ = 'greg'
import psycopg2
import pymongo
import json
import re

# the directory to store the movie preview clips in
image_directory = "/home/greg/Databases/chimp/images/"

# connect to the mongodb server
client = pymongo.MongoClient()
db = client['serengeti_2015-06-27']
subjects = db["serengeti_subjects"]
classifications = db["serengeti_classifications"]

details = "dbname='zooniverse' user='greg' host='localhost' password='apassword'"
postgres_session = psycopg2.connect(details)
postgres_cursor = postgres_session.cursor()

postgres_cursor.execute("DROP TABLE Snapshot2")
postgres_cursor.execute("CREATE TABLE Snapshot2(subject_id char(10),user_ip inet,user_name text,created_at timestamp,annotations json)")
postgres_cursor.execute("CREATE INDEX user_2 ON Snapshot2(user_ip,user_name,created_at)")
postgres_cursor.execute("CREATE INDEX subjects_2 ON Snapshot2(subject_id)")
postgres_cursor.execute("CREATE INDEX time_2 ON Snapshot2(created_at)")

t = ""

for ii,c in enumerate(classifications.find({"tutorial":{"$ne":True}}).limit(1000000)):
    if ii % 5000 == 0:
        print ii
        print c["created_at"]


    user_ip = str(c["user_ip"])
    created_at = str(c["created_at"])
    annotations = json.dumps(c["annotations"])
    annotations = re.sub("'","",annotations)
    subject_id = c["subjects"][0]["zooniverse_id"]
    if "user_name" in c:
        user_name = c["user_name"]
    else:
        user_name = ""

    user_name = re.sub("'","",user_name)

    t += "('"+subject_id+"','"+user_ip+"','"+user_name+"', TIMESTAMP '"+created_at+"', '"+annotations+"') " + ","

    if (ii % 2000 == 0) and (ii > 0):
        stmt = "INSERT INTO Snapshot2 (subject_id,user_ip,user_name,created_at,annotations) VALUES " + t[:-1]
        postgres_cursor.execute(stmt)
        t = ""

stmt = "INSERT INTO Snapshot2 (subject_id,user_ip,user_name,created_at,annotations) VALUES " + t[:-1]
postgres_cursor.execute(stmt)
postgres_session.commit()