Detecting Blank Images
######################

Suppose you have a series of images taken over a period of time from a fixed location. You want to know if there is something in each of those images. For example, you have a webcam set up that regularly takes a photo of a room - does anyone enter that room?
If you have a gold standard blank image and you know that the only thing that can change is someone entering the room - the solution is simple. If there is any difference between one of the images and the blank - someone is there. But what if other things can change? For example, lighting - there might be a window in the room. Or for something like Snapshot Serengeti we could be looking at a bunch of trees - the leaves could be blowing in the background. That's technically movement but not the kind we want.

Snapshot Serengeti provides timestamps and locations for all images so we can look at  a time series of images. There is a tradeoff - the more images we have in our time series the accurate our calculations can be. But things change over time - grass, trees and leafs grow and die. So the time series probably wouldn't last months - probably more just days at most. We should also remove night time images - images where the average brightness is less than some certain threshold. We'll then read in the images ::

    axis = 0
    time_series = []
    for fname in glob.glob("/home/ggdhines/Databases/images/time_series/*.jpg"):
        img = cv2.imread(fname)[:,:,axis]
        equ = cv2.equalizeHist(img)
        f = equ.astype(float)
        time_series.append(f)

axis = 0 means that we are only reading in the R values (out of the RGB values) - we could also read the images in grayscale. (Just experimenting with stuff). Equalizing the img (cv2.equalizeHist) (http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html#gsc.tab=0) helps to account for differences in lighting. We can look at the average image with ::

    mean_image = np.mean(time_series,axis=0)
    plt.imshow(mean_image)

.. image:: images/avg_img.jpg
    :width: 500px
    :align: center
    :alt: average image

We can also calculate "percentile images" ::

    upper_bound = np.percentile(time_series,80,axis=0)

which gives us at each pixel the 80th percentile value. (i.e. 80 percent of the values at the pixel in our time series are less than or equal to this value). Similarly we can calculate the lower bounds ::

    lower_bound = np.percentile(time_series,20,axis=0)

Let's read in the image again and look for places where we have "extreme" pixels - pixels that lie below the 20th percetile or above the 80th ::

    template = np.zeros(img.shape,np.uint8)
    t2 = np.where(np.logical_or(equ>upper_bound , equ < lower_bound))
    template[t2] = 255

Finally we apply an opening operation to remove isolated points (noise) - (http://docs.opencv.org/3.1.0/d9/d61/tutorial_py_morphological_ops.html#gsc.tab=0)::

    opening = cv2.morphologyEx(template, cv2.MORPH_OPEN, kernel)

Below are some examples - there are some false positives where a change in the sky is detected (we could filter out sky pixels) but false positives aren't bad. We see that animals are definitely detected. If we did DB scan we could look for clumps of "extreme" pixels - if there are none, we have a blank image.

0
.. image:: images/0_original.jpg
    :width: 500px
    :align: center
    :alt: average image

.. image:: images/0_modified.jpg
    :width: 500px
    :align: center
    :alt: average image
1
.. image:: images/1_original.jpg
    :width: 500px
    :align: center
    :alt: average image

.. image:: images/1_modified.jpg
    :width: 500px
    :align: center
    :alt: average image
2
.. image:: images/2_original.jpg
    :width: 500px
    :align: center
    :alt: average image

.. image:: images/2_modified.jpg
    :width: 500px
    :align: center
    :alt: average image
3
.. image:: images/3_original.jpg
    :width: 500px
    :align: center
    :alt: average image

.. image:: images/3_modified.jpg
    :width: 500px
    :align: center
    :alt: average image
4
.. image:: images/4_original.jpg
    :width: 500px
    :align: center
    :alt: average image

.. image:: images/4_modified.jpg
    :width: 500px
    :align: center
    :alt: average image
5
.. image:: images/5_original.jpg
    :width: 500px
    :align: center
    :alt: average image

.. image:: images/5_modified.jpg
    :width: 500px
    :align: center
    :alt: average image
6
.. image:: images/6_original.jpg
    :width: 500px
    :align: center
    :alt: average image

.. image:: images/6_modified.jpg
    :width: 500px
    :align: center
    :alt: average image
7
.. image:: images/7_original.jpg
    :width: 500px
    :align: center
    :alt: average image

.. image:: images/7_modified.jpg
    :width: 500px
    :align: center
    :alt: average image
8
.. image:: images/8_original.jpg
    :width: 500px
    :align: center
    :alt: average image

.. image:: images/8_modified.jpg
    :width: 500px
    :align: center
    :alt: average image
9
.. image:: images/9_original.jpg
    :width: 500px
    :align: center
    :alt: average image

.. image:: images/9_modified.jpg
    :width: 500px
    :align: center
    :alt: average image
10
.. image:: images/10_original.jpg
    :width: 500px
    :align: center
    :alt: average image

.. image:: images/10_modified.jpg
    :width: 500px
    :align: center
    :alt: average image
11
.. image:: images/11_original.jpg
    :width: 500px
    :align: center
    :alt: average image

.. image:: images/11_modified.jpg
    :width: 500px
    :align: center
    :alt: average image
12
.. image:: images/12_original.jpg
    :width: 500px
    :align: center
    :alt: average image

.. image:: images/12_modified.jpg
    :width: 500px
    :align: center
    :alt: average image
13
.. image:: images/13_original.jpg
    :width: 500px
    :align: center
    :alt: average image

.. image:: images/13_modified.jpg
    :width: 500px
    :align: center
    :alt: average image
14
.. image:: images/14_original.jpg
    :width: 500px
    :align: center
    :alt: average image

.. image:: images/14_modified.jpg
    :width: 500px
    :align: center
    :alt: average image
15
.. image:: images/15_original.jpg
    :width: 500px
    :align: center
    :alt: average image

.. image:: images/15_modified.jpg
    :width: 500px
    :align: center
    :alt: average image
16
.. image:: images/16_original.jpg
    :width: 500px
    :align: center
    :alt: average image

.. image:: images/16_modified.jpg
    :width: 500px
    :align: center
    :alt: average image
17
.. image:: images/17_original.jpg
    :width: 500px
    :align: center
    :alt: average image

.. image:: images/17_modified.jpg
    :width: 500px
    :align: center
    :alt: average image
18
.. image:: images/18_original.jpg
    :width: 500px
    :align: center
    :alt: average image

.. image:: images/18_modified.jpg
    :width: 500px
    :align: center
    :alt: average image
19
.. image:: images/19_original.jpg
    :width: 500px
    :align: center
    :alt: average image

.. image:: images/19_modified.jpg
    :width: 500px
    :align: center
    :alt: average image
20
.. image:: images/20_original.jpg
    :width: 500px
    :align: center
    :alt: average image

.. image:: images/20_modified.jpg
    :width: 500px
    :align: center
    :alt: average image
21
.. image:: images/21_original.jpg
    :width: 500px
    :align: center
    :alt: average image

.. image:: images/21_modified.jpg
    :width: 500px
    :align: center
    :alt: average image
22
.. image:: images/22_original.jpg
    :width: 500px
    :align: center
    :alt: average image

.. image:: images/22_modified.jpg
    :width: 500px
    :align: center
    :alt: average image
23
.. image:: images/23_original.jpg
    :width: 500px
    :align: center
    :alt: average image

.. image:: images/23_modified.jpg
    :width: 500px
    :align: center
    :alt: average image
24
.. image:: images/24_original.jpg
    :width: 500px
    :align: center
    :alt: average image

.. image:: images/24_modified.jpg
    :width: 500px
    :align: center
    :alt: average image
25
.. image:: images/25_original.jpg
    :width: 500px
    :align: center
    :alt: average image

.. image:: images/25_modified.jpg
    :width: 500px
    :align: center
    :alt: average image
26
.. image:: images/26_original.jpg
    :width: 500px
    :align: center
    :alt: average image

.. image:: images/26_modified.jpg
    :width: 500px
    :align: center
    :alt: average image