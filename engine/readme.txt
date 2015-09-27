This directory contains the aggregation results that you requested for your Zooniverse project. The file contains important information to help you understand what is in this directory. If you have any questions, please go the Aggregation talk board on the Zooniverse site.

Aggregation is the process of combining classifications/markings from multiple users for a single subject into one answer. A subject is a single image or group of images that the user  views at a time. Users will often disagree with each other - this is why every subject needs to be shown to multiple users. However, an aggregate answer based on multiple users is usually highly accurate.

Our aggregation engine is designed to work with all Panoptes projects. This means that the analysis is not necessarily optimized for your project. Many of the algorithms use values, listed below, which seem to work well in practice. However, you may want to experiment and try to find better settings. You may even want to create your own algorithms. In either case, feel free to fork the the zooniverse/aggregation repo. If you create code that you feel would benefit others, feel free to make a pull request.

The flexibility of Panoptes means that we don’t know exactly how you want to use your data. So we may provide several different files, each with a different type of analysis or summary in them. Not all of the files may be relevant to your project. Use the rest of this readme file to help understand what specific files and values you want.

If you do use any of the aggregation results, please cite Zooniverse. (We are working on publishing the papers and will update this readme accordingly.) This code is still very much being developed. There are a few things that still need to be done and these are highlighted in the documentation.

In the root directory there will be one directory per workflow (if you only have one workflow, there will just be one directory). The directory names will correspond to the workflow names you used when creating the project. Each task is processed independently and is stored in one or more distinct files.

Some of the subjects will have less classifications/markings then the retirement threshold you set. These cases are most likely due to you changing the workflow while the project is live. (Since we have no way of comparing the semantics of different workflows, classifications/markings for previous workflows are ignored.) This may also be due to the fact that other than the initial task, there may be no guarantee that each user does each task (this depends on your workflow).

Classification output -
Each row contains the subject id, and a column for each possible output and the total number of people to have done this task for the given subject_id. p(output) is our estimated probability of that output being true. We currently support just plurality voting so p(output) is equal to the percentage of people who voted for that option.

Marking output -
Every marking task has two output file for each shape possible from that task. Note - this is done by shape NOT by tool type. For example, in Wildebeest Watch, users are asked to mark which direction wildebeest are travelling. All of the marking tools use the point shape. Not all users agree on which direction the wildebeest are travelling. By clustering on shape rather than tool type, we can easily deal with this possible disagreement between users. The alternative would be to have a single marking tool which as the first follow up question asks the user what direction the wildebeest is travelling in. This effectively doubles the number of clicks a user needs to do. We know from previous projects that this can have a real effect on how much volunteers do.

Anyways - we have two files. A detailed file and a summary file. The detailed file will end in point.csv and the summary file will end in point_summary.csv. (Other shapes will have the obvious change). 

We process markings by clustering them into groups. Each row of the detailed file contains one cluster from one subject. We provide the centroid of that cluster which is the average (in this case median) of all the members of that cluster. For each shape we have the following param (each given in its own column). 
points - x,y
line segment - x1,y1,x2,y2
rectangle - x1,y1 (corner of rectangle) x2,y2 (opposite corner)
ellipse - x,y (center of ellipse), r1, r2 (major and minor axes) and theta - rotation

Since polygons can have an arbitrary number of points, for polygons we have one column with all of the values in the form (x1,y1),(x2,y2) … surrounded by double quotation marks (so it should load as just one column). If you know JSON, loading the polygon points is trivial.

We also provide the most likely tool for each cluster (right now decided by plurality voting) and the what the highest percentage is, i.e. that 65% of people voted for the most likely tool. In addition, we provide p(true positive) (again currently decided by majority voting) which is our measure of how likely a cluster represents something that actually exists (as opposed to someone confusing some rock and snow for a penguin). These values will allow you to filter out problematic clusters.

For the summary files, each subject gets one line. We give you a count of the number of different cluster types (requiring that p(true positive) >= 0.5). If you are using the marking tool to have users count things and are not interested in the specific xy coordinates, these are the results you want. The summary files next provide the mean and median probability of true positives. That is, the average probability of a cluster in that subject being a true positive. Finally, we give the average percentage for the most likely tool. These values will help identity difficult images.