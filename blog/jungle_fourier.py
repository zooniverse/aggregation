__author__ = 'ggdhines'
# import matplotlib
# import aggregation_api
import cv2
# import numpy as np
import matplotlib.pyplot as plt
from aggregation_api import AggregationAPI
# from sklearn.cluster import KMeans
# import matplotlib.cbook as cbook
import numpy as np

jungle = AggregationAPI(153,"development")
jungle.__setup__()
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

    img = cv2.imread(fname,0)
    f = np.fft.fft2(img)
    # fshift = np.fft.fftshift(f)

    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    # magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

    # print magnitude_spectrum
    # print dft_shift
    # m1 = dft_shift > 100
    # # magnitude_spectrum[low_values_indices] = 0
    # # print magnitude_spectrum
    # rows, cols = img.shape
    # mask = np.zeros((rows,cols,2),np.uint8)
    # mask[m1] = 1
    # fshift = dft_shift*mask
    # f_ishift = np.fft.ifftshift(fshift)
    # img_back = cv2.idft(f_ishift)
    # img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    #
    # plt.subplot(121),plt.imshow(img, cmap = 'gray')
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
    # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    # plt.show()

    # break
    #
    # # magnitude_spectrum = 20*np.log(np.abs(fshift))
    #
    # plt.subplot(121),plt.imshow(img, cmap = 'gray')
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    # plt.show()
    #
    rows, cols = img.shape
    crow,ccol = rows/2 , cols/2
    # # create a mask first, center square is 1, remaining all zeros
    # mask = np.zeros((rows,cols,2),np.uint8)
    # mask[crow-30:crow+30, ccol-30:ccol+30] = 1
    #
    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows,cols,2),np.uint8)
    thres = 15
    mask[crow-thres:crow+thres, ccol-thres:ccol+thres] = 1

    # apply mask and inverse DFT
    fshift = dft_shift*mask
    print fshift.shape
    for i in range(crow-thres,crow+thres+1):
        for j in range(ccol-thres,ccol+thres+1):
            mask = np.zeros((rows,cols,2),np.uint8)
            mask[i,j] = 1
            fshift = dft_shift*mask

            print fshift[i][j]

            f_ishift = np.fft.ifftshift(fshift)
            img_back = cv2.idft(f_ishift)
            img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

            plt.subplot(121),plt.imshow(img, cmap = 'gray')
            plt.title('Input Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
            plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
            plt.show()


    break
    print fshift
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    #
    # fshift = np.fft.fftshift(f)
    # fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
    # f_ishift = np.fft.ifftshift(fshift)

    # f = np.fft.fft2(img)
    # fshift = np.fft.fftshift(f)
    # fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
    # f_ishift = np.fft.ifftshift(fshift)
    # img_back = np.fft.ifft2(f_ishift)
    # img_back = np.abs(img_back)
    #
    # plt.subplot(131),plt.imshow(img, cmap = 'gray')
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
    # plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
    #
    # plt.show()