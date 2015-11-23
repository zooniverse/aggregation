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
    subjects_to_aggregate = random.sample(retired_subjects,100)
    # subjects_to_aggregate = [649573]
    # project.__migrate__()
    # project.__aggregate__(subject_set = [671541,663067,664482,662859])
    project.__aggregate__(subject_set=subjects_to_aggregate)

    stats = json.load(open("/home/ggdhines/245.stats","rb"))
    # print stats

    # latex_transcription.latex_output(project,121,random.sample(subjects_to_aggregate,min(20,len(subjects_to_aggregate))))
    json_transcription.json_dump(project,subjects_to_aggregate)


    print stats["capitalized"]
    print stats["double_spaces"]
    print stats["errors"]
    print stats["retired lines"]
    print np.mean(stats["line_length"])