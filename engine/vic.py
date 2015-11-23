import transcription_3
import pickle
import random
import json
import numpy as np
import json_transcription
import latex_transcription

retired_subjects = pickle.load(open("/home/ggdhines/245.retired","rb"))
# print "retired subjects is " + str(len(retired_subjects))
with transcription_3.Tate(245,"development") as project:
    cursor = project.postgres_session.cursor()

    cursor.execute("SELECT id FROM users WHERE login = 'bootnecksbs'")

    id_ = cursor.fetchone()[0]

    cursor.execute("SELECT * FROM classifications WHERE user_id = " + str(id_) + " and project_id = 245")

    subjects = 0
    lines = 0

    for t in cursor.fetchall():
        subjects += 1

        for task in t[4]:
            if task["task"] == "T2":
                for marking in task["value"]:
                    if marking["type"] == "text":
                        lines += 1

    print subjects
    print lines

