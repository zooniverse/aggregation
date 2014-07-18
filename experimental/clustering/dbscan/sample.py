#!/usr/bin/env python
import random


with open('/Users/greghines/Downloads/simple-gps-points-120312.txt','r') as f:
    f.seek(0,2)                 # seek to end of file
    b = f.tell()
    for i in range(40000):
        f.seek(int(b*random.random()))

        # Now seek forward until beginning of file or we get a \n
        while True:
            f.seek(-2,1)
            ch = f.read(1)
            if ch=='\n': break
            if f.tell()==1: break

        # Now get a line
        print f.readline()[:-1]

