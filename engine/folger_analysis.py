import transcription_3

folger_ids = [49241,50769,53795,67272,81278,86365,86365,110543,110755,111232,1122]
metadict = {}
corrected_ids = {}

unmatched_ids = []

# print "retired subjects is " + str(len(retired_subjects))
with transcription_3.Tate(376,"development") as project:
    postgres_cursor = project.postgres_session.cursor()

    postgres_cursor.execute("select distinct(subject_ids) from classifications where workflow_id = 205;")
    transcribed_subjects = []
    for j in postgres_cursor.fetchall():
        if len(j[0]) == 0:
            continue
        subject_id = int(j[0][0])
        postgres_cursor.execute("select metadata from subjects where id = " + str(subject_id) + ";")
        metadata = postgres_cursor.fetchone()[0]

        metadict[subject_id] = metadata
        if 'RF/JPG#' in metadata:
            fname = metadata['RF/JPG#']
            f_id = int(fname[:-4])

            corrected_ids[f_id] = subject_id
        else:
            unmatched_ids.append(subject_id)

    # print len(corrected_ids)
    # print len([f for f in folger_ids if f in corrected_ids])
    #     # if f_id not in corrected_ids:
    #     #     print f_id
    #
    # print
    # for subject_id in metadict:
    #     if subject_id not in list(corrected_ids.values()):
    #         print subject_id,metadict[subject_id]
    # print
    for m in metadict.values():
        if "file name" in m:
            print m["file name"]
            print m
            print
        elif "Filename" in m:
            print m["Filename"]
            print m
            print
        else:
            print "***"
            print m
            print

    assert False

    project.__migrate__()
    project.__aggregate__(subject_set=subjects)

    for s in subjects:
        aggregations = list(project.__yield_aggregations__(205,s))
        print s