#!/usr/bin/env python
from __future__ import print_function
import cv2
import numpy as np
import glob
import matplotlib
# matplotlib.use('WXAgg')
import matplotlib.pyplot as plt
import database_connection
import numpy as np
import tesseract_font
from active_weather import ActiveWeather
import math
classifier = tesseract_font.ActiveTess()

# cass_db = database_connection.Database()

project = ActiveWeather()
subject_id = "Bear-AG-29-1940-0720"
fname = "/home/ggdhines/Databases/old_weather/aligned_images/Bear/1940/"+subject_id+".JPG"
image = cv2.imread(fname,0)

approximate_image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,151,0)
confidences = []
is_correct = []
for h_index in range(len(project.horizontal_grid)-1):
    end = False

    for v_index in range(len(project.vertical_grid)-1):
        cell,_ = project.__extract_cell__(approximate_image,h_index,v_index)

        classifier.tess.set_image(cell)
        classifier.tess.get_utf8_text()

        words = list(classifier.tess.words())
        words_in_cell = ".".join([w.text for w in words if w.text is not None])

        conf_in_cell = [w.confidence for w in words if w.text is not None]

        if conf_in_cell != []:
            print(words_in_cell)
            confidences.append(np.mean(conf_in_cell))

            if project.cass_db.__has_cell_been_transcribed__(subject_id,0,h_index,v_index):
                gold_standard = project.cass_db.__get_gold_standard__(subject_id,0,h_index,v_index)
                # print(gold_standard)
                if gold_standard == words_in_cell:
                    is_correct.append(1)
                else:
                    is_correct.append(0)
            else:
                end = True
                break

    if end:
        break

# plt.hist(confidences, bins=15, normed=1, histtype='step', cumulative=1)
# plt.show()

print(sum(is_correct)/float(len(is_correct)))
b = 0.1
y_vals = []
x_vals = np.arange(0,1.01,0.05)
print(is_correct)
for x in x_vals:
    kerl_dist = [math.exp(-((c/100. -x)**2/float(2*b**2))) for c in confidences]

    y = sum([c*k for (c,k) in zip(is_correct,kerl_dist)])/float(sum(kerl_dist))
    y_vals.append(y)

plt.plot(np.arange(0,1.01,0.05),y_vals)
plt.show()