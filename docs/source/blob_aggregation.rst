Blob (Polygon) Aggregation
##########################

A blob (or polygon) is the more general marking tool that we have in the project builder. Blob aggregation is different from other marking in that it isn't really clustering.
Instead we are just looking for regions which have been "outlined" by multiple people. For example, using the familar ven diagram below:

Suppose one person had outlined the left circle and an other the right. Then the inside red region would be the region that is outlined by both of them. If we set our threshold of requiring at least two people to have outlined an area than we would return this
region - otherwise we would return nothing.

The basic idea is to look for pixels which enough people have selected (i.e. pixels which are in regions which people have outlined). A brute force method would involve looking through all the pixels and for each pixel looking trhough all the polygons from each users. This is rather impractical.
(Not to mention that we would then need to group any such pixels, maybe using DBscan, and then return the outline of each cluster.)

We can use openCV (a python and C++ graphics library) to do the same basic idea but much more efficiently. Note that openCV can be a pain to install. There is a ubuntu package for opencv but it is rather old. As of time of writing this page, I'm still confirming that opencv will work. It should but in case it doesn't (or if in the future there are any trouble with opencv, everything in theory should be doable with scikit-image another python graphics library).
