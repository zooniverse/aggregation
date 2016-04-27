Blob (Polygon) Aggregation
##########################

A blob (or polygon) is the more general marking tool that we have in the project builder. Blob aggregation is different from other marking in that it isn't really clustering.
Instead we are just looking for regions which have been "outlined" by multiple people. For example, using the familar ven diagram below:

Suppose one person had outlined the left circle and an other the right. Then the inside red region would be the region that is outlined by both of them. If we set our threshold of requiring at least two people to have outlined an area than we would return this
region - otherwise we would return nothing.

The basic idea is to look for pixels which enough people have selected (i.e. pixels which are in regions which people have outlined). A brute force method would involve looking through all the pixels and for each pixel looking trhough all the polygons from each users. This is rather impractical.
(Not to mention that we would then need to group any such pixels, maybe using DBscan, and then return the outline of each cluster.)

We can use openCV (a python and C++ graphics library) to do the same basic idea but much more efficiently. Note that openCV can be a pain to install. There is a ubuntu package for opencv but it is rather old. As of time of writing this page, I'm still confirming that opencv will work. It should but in case it doesn't (or if in the future there are any trouble with opencv, everything in theory should be doable with scikit-image another python graphics library).

The polygon marking tools returns polygons such as the one below [EXAMPLE TO BE INSERTED]

Any pixel inside that polygon has been "selected". So we really just want to "fill in" the polygon -think of using a Paint fill in tool and then selecting all the white pixels. Which is exactly what we're going to do. (The way things work with openCV, the background, or pixels which have not been selected are going to be black)

A slight technical issue first -

Selecting tool type
###################

So the first part of the aggregation happens without regards to tool type. For example, if you highlight a region as being "red grass" and I highlight the same region as being "blue grass" - we're in agreement that there is grass there, just not what type of grass it is. There is a bit of an ongoing discussion about whether this is the right approach (maybe blue and red grass are so different that they really can't be confused with each other - so if there is any overlap between your region and mine, that region doesn't really represent any thing special). So this setup may be revisited in the future. Anyways ...

In __cluster__, the variable positive_area is a matrix the same dimension as the subject image. (In the rare chance that the dimensions have not been provided, we set the matrix size to be just big enough to hold all of the polygons)
The cells in positive_area are equal to 255 whereever enough people have selected that pixel and 0 elsewhere. The idea behind the name is that the pixels which are 255 are true positive pixels (i.e. they actually represent something real, not just someone confusing rocks and snow with a penguin).

We next create a variable called most_common_tool which is again an array equal to the same dimension as the subject image. Each cell in most_common_tool is equal to the most commonly used tool used to select that pixel (even if that pixel is not a true positive pixel).
We again build most_common_tool one user at a time. For user i, we fill in a canvas by tool type values - i.e. canvas[x,y]=t if user i selected the pixel (x,y) using tool t. So after going through each user, we again have a list of canvases (one per user). This time to combine these canvases into one, instead of adding the values together, we'll take the mode (i.e. the most common value) per cell ::

    most_common_tool_canvas = stats.mode(polygons_by_tools)[0][0]

So most_common_tool_canvas is a canvas with the same dimensions as the subject image where each pixel is equal to the most commonly used tool used to select that pixel. Note that if a user didn't select a pixel, that user should obviously have nothing to do with selecting the most commonly used tool wrt that pixel. This is conceptually trivial but implmenting is a slight pain. We could initialize the canvas with ::

    tool_canvas = np.zeros(dimensions, np.uint8)

But that would be just as the same as if the user had used tool 0 to select all of the pixels. We could "paint" tool_canvas with tool_index+1 - so if the mode for a particular tool was 1, we would know that that corresponds to tool 0. (And this would allow us to distinguish betwee a user not selecting a pixel and selecting using tool 0.) However, suppose we have 6 people who viewed a subject. Two people mark a pixel using tool 1, one person marks it with tool 2 and the last 3 people do not mark the pixel at all.
The mode as calculated using the scipy mode function (see above) would be 0 which isn't useful to use at all (we want the value 1) - there isn't a good way of checking when we want to get the second most common value and using it. (Efficiency really matters here since images will have thousands of pixels and having to iterate over them will take forever.) So we're going to have to cheat - we'll assign each user a value (which appears in their canvas) which means that they didn't mark that pixel.
These values have to be unique for each user, and can't be equal to any tool value - otherwise we could mess up the mode calculation. So for user i, we'll use 255-i. This means that the number of tools used plus the number of users who see an image must add to less than 256. This seems like a pretty reasonable restriction.

If the threshold for setting a pixel to be positive (i.e. not noise) is at least half the users, we could simply "paint" tool_canvas with (tool_index+1)


Then to actually find all of the true positive pixels which have been selected using tool 0 (for example) - we simply do ::

    area_by_tool = np.where((most_common_tool==(tool_index)) & (positive_area > 0))

We then create a template of size each to the subject dimension and set all the pixels in area_by_tool to be 255 ::

    template = np.zeros(dimensions,np.uint8)
    template[area_by_tool] = 255

