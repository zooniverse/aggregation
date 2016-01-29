__author__ = 'ggdhines'
import matplotlib
matplotlib.use('WXAgg')
import aggregation_api
import cv2
import numpy as np
import matplotlib.pyplot as plt
from aggregation_api import AggregationAPI
from sklearn.cluster import KMeans
import matplotlib.cbook as cbook

jungle = AggregationAPI(153,"development")
# jungle.__migrate__()
# jungle.__aggregate__()

postgres_cursor = jungle.postgres_session.cursor()
postgres_cursor.execute("select subject_ids,annotations from classifications where project_id = 153")

markings = {}

for subject_ids,annotations in postgres_cursor.fetchall():

    if subject_ids == []:
        continue
    s = subject_ids[0]
    for task in annotations:
        if task["task"] == "T2":
            try:
                m = task["value"][0]["points"]
                if s not in markings:
                    markings[s] = [m]
                else:
                    markings[s].append(m)
            except (KeyError,IndexError) as e:
                pass

for subject_id,points in markings.items():
    fname = jungle.__image_setup__(subject_id)

    image_file = cbook.get_sample_data(fname)
    image = plt.imread(image_file)

    fig, ax1 = plt.subplots(1, 1)
    ax1.imshow(image)

    all_points = []
    for a in points:
        for b in a:
            all_points.append((b["x"],b["y"]))

    if len(all_points) < 6:
        plt.close()
        continue

    kmeans = KMeans(init='k-means++', n_clusters=6, n_init=10)
    all_points = np.asarray(all_points)
    kmeans.fit(all_points)

    labels = kmeans.labels_

    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        class_member_mask = (labels == k)

        xy = all_points[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=14)


    plt.show()

assert False

def analyze(f_name,display=False):
    image = cv2.imread(f_name)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image2 = cv2.threshold(image,127,255,cv2.THRESH_BINARY)

    # cv2.imwrite("/home/ggdhines/jungle.jpeg",gray_image)

    # print gray_image.ravel()
    ix = np.in1d(gray_image.ravel(),range(175)).reshape(gray_image.shape)
    x,y = np.where(ix)

    if display:
        plt.plot(y,x,".")
        plt.ylim((gray_image.shape[0],0))
        plt.show()

    n, bins, patches = plt.hist(y,range(min(y),max(y)+1))
    med = np.median(n)


    peaks = [i for i,count in enumerate(n) if count > 2*med]
    buckets = [[peaks[0]]]




    for j,p in list(enumerate(peaks))[1:]:
        if (p-1) != (peaks[j-1]):
            buckets.append([p])
        else:
            buckets[-1].append(p)

    bucket_items = list(enumerate(buckets))
    bucket_items.sort(key = lambda x:len(x[1]),reverse=True)

    a = bucket_items[0][0]
    b = bucket_items[1][0]


    vert = []
    for x,p in bucket_items:
        if (a <= x <= b) or (b <= x <= a):
            vert.extend(p)

    print vert
    print min(vert)
    print max(vert)

    if display:
        plt.plot((min(y),max(y)+1),(2*med,2*med))
        plt.show()

        n, bins, patches = plt.hist(x,range(min(x),max(x)+1))
        med = np.median(n)

        plt.plot((min(x),max(x)+1),(2*med,2*med))
        plt.plot((min(x),max(x)+1),(1*med,1*med),color="red")
        plt.xlim((0,max(x)+1))
        plt.show()

if __name__ == "__main__":
    project = aggregation_api.AggregationAPI(153,"development")
    f_name = project.__image_setup__(1125393)
    print f_name
    analyze(f_name,display=True)