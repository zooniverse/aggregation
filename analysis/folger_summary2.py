__author__ = 'ggdhines'
import re
import numpy as np

i = 0
total_a = 0
total_b = 0

with open("/home/ggdhines/folger_results2","rb") as f:

    for l in f.readlines():
        if re.search('[a-z]',l) is not None:
            continue
        if l[:-1] == '':
            continue
        try:
            a,b = l.split(" ")
            total_a += int(a)
            total_b += int(b[:-1])
            i += 1
        except ValueError:
            c = int(l[:-1])

print total_a,total_b
print i