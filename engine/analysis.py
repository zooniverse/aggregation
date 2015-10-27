__author__ = 'ggdhines'
from transcription import Tate
import numpy as np
import matplotlib.pyplot as plt

# 121
with Tate(245,"development") as project:
    annotation_generator = project.__cassandra_annotations__()
    subjects = project.__get_subjects__(121)

    i = 0
    annotated_subjects = set()
    for subject_id,user_id,annotation,dimensions in annotation_generator(121,subjects):
        i += 1
        annotated_subjects.add(subject_id)
    print i
    print len(annotated_subjects)



    all_lines = []

    for subject_id,annotations in project.__yield_aggregations__(121):
        lines = 0
        for cluster_index,cluster in annotations['T2']['text clusters'].items():
            if cluster_index == "all_users":
                continue
            if cluster["num users"] >= 5:
                lines += 1

        if lines > 0:
            all_lines.append(lines)

    print sum(all_lines)
    print len(all_lines),np.mean(all_lines),np.median(all_lines)
    print len([1 for s in all_lines if s >= 5])
    plt.hist(all_lines,range(1,27))
    plt.show()