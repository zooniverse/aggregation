__author__ = 'ggdhines'
import csv
import json

classifications = {}
total = 0
a_total = 0
errors = 0
with open("/home/ggdhines/Downloads/9e6c81af-3688-48b9-aa13-61a612e9b32b.csv","rb") as infile:
    reader = csv.reader(infile)
    columns = reader.next()

    for row in reader:
        total += 1
        if int(row[3]) != 1224:
            continue

        try:
            ann = json.loads(row[-2])[0]["value"]
            # print ann#["value"]
            subject_data = json.loads(row[-1])
            subject_id = int(subject_data.keys()[0])
            errors += 1
        except ValueError:
            continue

        a_total += 1

        if subject_id not in classifications:
            classifications[subject_id] = [ann]
        else:
            classifications[subject_id].append(ann)

print total
print a_total
print len(classifications)
old_subjects = []

# with open("/home/ggdhines/Dropbox/764_old/1224_PreProduction_Workflow/initDoes_this_look_like_a_pulsar.csv","rb") as old_subject_file:
#     reader = csv.reader(old_subject_file)
#     next(reader, None)
#
#     for row in reader:
#         old_subjects.append(int(row[0]))

results = {}
for subject_id,votes in classifications.items():
    if subject_id in old_subjects:
        continue
    p = sum([1 for v in votes if v == "Yes"])/float(len(votes))
    results[subject_id] = (p,len(votes))

ordered_results = sorted(results.items(),key= lambda x:x[1][0],reverse=True)

with  open("/home/ggdhines/Downloads/dab0480f-1e81-4e2d-8403-e7cd81899a58 (1).csv","rb") as subject_file:
    metadata = {}
    reader = csv.reader(subject_file)
    next(reader, None)

    for row in reader:
        # values = json.loads(row)
        id = int(row[0])
        metadata[id] = json.loads(row[-4])

print len(metadata)
print len(ordered_results)

with open("/tmp/pulsars.csv","wb") as f:
    for subject_id,(p,count) in ordered_results:
        if subject_id not in metadata:
            continue



        m = metadata[subject_id]["CandidateFile"]
        if "#Class" in metadata[int(subject_id)]:
            print "skipping"
            continue

        # print prob,d
        # if prob == 1. and (int(d) == 1):
        #     print a
        #     assert False
        # print str(a)+","+str(prob)+","+str(c)+"," + "https://www.zooniverse.org/projects/zooniverse/pulsar-hunters/talk/subjects/"+a + "," + str(m)
        f.write(str(subject_id)+","+str(p)+","+str(count)+"," + "https://www.zooniverse.org/projects/zooniverse/pulsar-hunters/talk/subjects/"+str(subject_id) + "," + str(m) + "\n")

