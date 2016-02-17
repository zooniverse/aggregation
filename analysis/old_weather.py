__author__ = 'ggdhines'
import matplotlib
matplotlib.use('WXAgg')
import aggregation_api
from matplotlib import pyplot as plt
import matplotlib.cbook as cbook

project = aggregation_api.AggregationAPI(project_id = 195, environment="staging")
project.__setup__()
cursor = project.postgres_session.cursor()
# stmt = "select name from projects "
# cursor.execute(stmt)

# for r in cursor.fetchall():
#     if r[1] == None:
#         continue
#     if "old" in r[1]:
#         print r

# stmt = "select * from users where email = 'greg@zooniverse.org'"
# cursor.execute(stmt)
# for r in cursor.fetchall():
#     print r
# assert False
#
# stmt = "select * from workflows where project_id = 195"
# cursor.execute(stmt)
#
# for r in cursor.fetchall():
#     print r

# assert False

# print project.workflows
# print project.__sort_annotations__()

stmt = "select subject_ids,annotations from classifications where workflow_id = 611"# and user_id = 42"
cursor.execute(stmt)

for subject_ids,annotations  in cursor.fetchall():
    subject_id = subject_ids[0]
    # subject_id = r[13][0]
    # print subject_id
    # if subject_id == 2369:
    #     print r
    # continue

    print subject_id
    fname = project.__image_setup__(subject_id)

    image_file = cbook.get_sample_data(fname)
    image = plt.imread(image_file)
    fig, ax = plt.subplots()
    im = ax.imshow(image)

    print annotations

    for box in annotations:
        #print box["type"]
        scale = 1.5
        x1 = box["x"]/scale
        y1 = box["y"]/scale

        x2 = x1 +box["width"]/scale
        y2 = y1 + box["height"]/scale



        # x1 = 146/scale
        # y1 = 154/scale
        # x2 = x1 + 1009/scale
        # y2 = y1 + 1633/scale



        plt.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],"-",color="blue")

    plt.show()

