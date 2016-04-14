******************
Graphical Analysis
******************

In this section, we'll talk about some techniques that can be used to automatically do some common tasks in Zooniverse projects. You can try adapting these techniques to your project - the aim is reduce the number of subjects/tasks that you actually need volunteers to process.
This can make a significant difference in how long a project needs to complete and also (hopefully) make for happier volunteers by getting rid of the "unnecessary" work. For example, users may be bored in they need to watch 50 30-second clips of worms before they see an egg laid. It would be great if we could prune away 45 clips (for example).
This section doesn't actually talk about Worm Watch - it was just an example I recently came across and I know the Worm Watch researchers have some cool ideas for how they want to reduce the volunteers' work.

These ideas may only work some of the time - that's fine. As long as we can identify when the techniques fail (not always a trivial thing), we can give those subjects to the users.

All of the code below is in Python and uses common libraries such as openCV and numpy.

Let's start with the following image of some butterfly wings:

.. image:: images/butterfly1.jpeg
    :width: 300px
    :align: center
    :height: 300px
    :alt: alternate text

Our first task is to measure the length of the wings. To do so, we first need to "extract" the wings from the image. The background is a nice distinctive pink. Each pixel has 3 values (known the RGB values which give the red, green and blue intensities of that pixel). For the background, an example pixel is (211,44,124).
(I found this by opening the image in Gimp - a Linux graphics editor. ) We'll use this as the background reference colour - the background is so different from the wings that the exact reference colour really doesn't matter. We'll calculate the difference between every pixel in the image
and this reference colour. If the difference is large enough, we have a "foreground" pixel (wing, ruler or card). We'll start by using openCV to load the image (can also use Matplotlib or probably a bunch of other libraries) ::

    import cv2
    image = cv2.imread(file_name)

The image is now loaded as a Numpy array with size height x width x 3. Each pixel is actually a value in R^3 and so the difference between a pixel and our reference value is the Euclidean distance between the two ::

    import numpy as np
    image = np.asarray(image,dtype=np.int)
    distance_per_axis = np.power(image - [211,44,124],2)
    distance_squared = np.sum(distance_per_axis,axis=2)
    distance = np.power(distance_squared,0.5)

The second line converts image such that each pixel now has type "integer". This is really important since originally each pixel had type "unsigned integer 8" (np.uint8). Unsigned integers cannot have negative values which can cause trouble with our calculations. np.uint8 is the starting type for any loaded image. (8 means that the maximum brightness for a pixel is (2^8-1,2^8-1,2^8-1) or (255,255,255) which is white.)
The third through fifth lines then calculate the Euclidean distance from each pixel to our reference pixel. We could have done this with a loop but using numpy is much faster and gives cleaner code.

The variable distance is now a matrix of size height x width with each cell value having type 'numpy.float64' (a real number). Functions that we'll use later on require us to convert back to numpy.uint8 so we'll do that now ::

    distance = np.uint8(255 - cv2.normalize(distance,distance,0,255,cv2.NORM_MINMAX))

Not sure exactly why "distance" needs to be used 3 times, the openCV documentation isn't great. The distance calculation gives foreground pixels high values (the higher the value, the more white the pixel) but we want foreground pixels to be black. So we need to invert the image - hence the "255-".

Finally, we apply a threshold - map each pixel to be either 0 or 255 (instead of ranging between those values). We'll use binary thresholding which takes a given threshold - if a pixel is below that threshold, it gets maps to 0. Otherwise the pixel is mapped to 255.
If you don't want to rely on choosing a threshold, look into Otsu's binarization (i haven't played with it for this particular problem but seems like a good possibility). However, for our problem there is such a large difference between foreground and background pixel (and that difference is pretty constant over multiple images) that I choose a threshold of 200 and it worked fine ::

    _,overall_d = cv2.threshold(overall_d,200,255,cv2.THRESH_BINARY)

The "_" means that cv2 is returning another value that I don't care about. To save the image, we can use ::

    cv2.imwrite(file_name,overall_d)

Technically we could also matplotlib but I wouldn't recommend it - the image will get shrunk by borders and it is a complete pain to deal with. The image looks like

.. image:: images/butterfly2.jpg
    :width: 300px
    :align: center
    :height: 300px
    :alt: alternate text