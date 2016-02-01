import transcription
import pickle

retired_subjects = pickle.load(open("/home/ggdhines/245.retired","rb"))
# print "retired subjects is " + str(len(retired_subjects))
with transcription.Tate(245, "development") as project:
    cursor = project.postgres_session.cursor()

    cursor.execute("SELECT id FROM users WHERE login = 'bootnecksbs'")

    id_ = cursor.fetchone()[0]

    cursor.execute("SELECT annotations FROM classifications WHERE user_id = " + str(id_) + " and project_id = 245")

    subjects = 0
    lines = 0

    for t in cursor.fetchall():
        subjects += 1

        for task in t[0]:
            if task["task"] == "T2":
                try:
                    for marking in task["value"]:
                        if marking["type"] == "text":
                            lines += 1
                        else:
                            assert marking["type"] == "image"
                except TypeError:
                    print task
            else:
                assert task["task"] in ["T0","T3"]



    print subjects
    print lines

