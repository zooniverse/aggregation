__author__ = 'ggdhines'
from learning import NearestNeighbours
import random
from skimage.data import load
import json
from sklearn.decomposition import PCA
import numpy as np
class updatedNN(NearestNeighbours):
    def __init__(self):
        NearestNeighbours.__init__(self)

        cursor = self.conn.cursor()

        buckets = {i:[] for i in range(10)}

        for i in range(len(self.training[0])):
            a = self.training[0][i]
            b = self.training[1][i]

            buckets[b].append(a)

        cursor.execute("select subject_id,region_id,column_id,row_id,digit_index from cells")

        probabilities = {}
        correctness = {}

        ids = []


        for id_ in cursor.fetchall():
            # print alg,gold
            # id_ = subject_id,region_id,column_id,row_id

            ids.append(id_)

        training_indices = random.sample(ids,40)

        new_training = {i:[] for i in range(10)}

        cursor.execute("select subject_id,region_id,column_id,row_id,digit_index,algorithm_classification, probability, gold_classification,pixels from cells")
        for (subject_id,region_id,column_id,row_id,digit,alg,p,gold,pixels) in cursor.fetchall():
            # print alg,gold
            id_ = subject_id,region_id,column_id,row_id,digit

            if gold < 0:
                continue

            if id_ in training_indices:
                cursor.execute("select fname from subject_info where subject_id = " + str(subject_id))
                fname = cursor.fetchone()[0]
                image = load(fname)

                pixels,_ = self.__normalize_pixels__(image,json.loads(pixels))
                print type(pixels)

                new_training[gold].append(list(pixels))

        for i in new_training.keys():
            print i,len(new_training[i])

        updated_training = []
        updated_labels = []

        for i in buckets:
            if new_training[i] == []:
                continue
            replacement_indices = random.sample(range(len(buckets[i])),int(0.5*len(buckets[i])))

            for j in replacement_indices:
                buckets[i][j] = random.sample(new_training[i],1)[0]


            updated_training.extend(buckets[i])
            updated_labels.extend([i for k in range(len(buckets[i]))])

        pca = PCA(n_components=50)
        print updated_training[0]
        print self.training[0][0]
        # updated_training = np.asarray(updated_training)
        self.T = pca.fit(updated_training)
        reduced_training = self.T.transform(updated_training)
        self.clf.fit(reduced_training, updated_labels)

        correct = 0
        total = 0.

        cursor.execute("select subject_id,region_id,column_id,row_id,digit_index, gold_classification,pixels from cells")
        for (subject_id,region_id,column_id,row_id,digit,gold,pixels) in cursor.fetchall():
            id_ = subject_id,region_id,column_id,row_id,digit

            if gold < 0:
                continue

            if id_ not in training_indices:
                cursor.execute("select fname from subject_info where subject_id = " + str(subject_id))
                fname = cursor.fetchone()[0]
                image = load(fname)

                _,algorithm_digit,_ = self.__identify_digit__(image,json.loads(pixels),False)
                # pixels = self.__normalize_pixels__(image,json.loads(pixels))

                if algorithm_digit == gold:
                    correct += 1
                total += 1
        print correct/total

a = updatedNN()