__author__ = 'ggdhines'
import re
import numpy as np

with open("/home/ggdhines/folger_results","rb") as f:
    subject_id = None
    percentage = None
    num_subjects = 0

    consensus = []
    subject_complete = []

    started_lines = 0
    total_lines = 0.

    completed = 0

    for l in f.readlines():
        if re.search('[a-z]',l) is not None:
            continue
        if l[:-1] == '':
            continue
        # print l[:-1]
        if subject_id is None:
            subject_id = int(l[:-1])
            num_subjects += 1
        else:
            a,b = l.split(" ")

            if b == "0\n":
                pass
            else:
                try:
                    a = int(a)
                    b = int(b[:-1])

                    started_lines += a
                    total_lines += b

                    if (a/float(b)) > 0.8:
                        completed += 1

                    subject_id = None
                except ValueError:
                    percent_complete = float(b[:-1])
                    percent_consensus = float(a)

                    consensus.append(percent_consensus)
                    subject_complete.append(percent_complete)


print num_subjects
print started_lines/float(total_lines)
print np.mean(subject_complete)
print np.mean(consensus)
print completed