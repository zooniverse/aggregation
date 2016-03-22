********************
Rectangle Clustering
********************

One of the marking tools available through the PFE is rectangles. This allows users to outline a rectangular region of an image. A rectangle is just a special type of polygon and in fact you can basically use polygon aggregation on rectangles. This turned out to be overkill and rather slow so we created a clustering technique specifically for rectangles.

.. image:: images/rectangle_overlap.jpg
    :width: 500px
    :align: center
    :height: 500px
    :alt: 3 rectangular markings with overlapping region in blue