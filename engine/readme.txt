File(s) summary :
Each subdirectory corresponds to a different workflow. Each file in a directory corresponds to a task associated with that workflow. Those files are identified by a the task id and a snippet of the instructions given to the users. For follow up questions (those questions associated with a specific marking), we also include the tool id and follow up question id.

For classification tasks (including those which are follow up questions), which only allow one choice, we have one file. This file has the following columns:
subject_id
cluster_index (if a follow up question)
p(...) - percent of users who chose each possible results
num_users - number of users who did this task for the given subject id

For classification tasks which also multiple choices, we have two files. The columns ar:
subject_id
cluster_index (if a follow up question)
p(...) - percent of users who chose each possible results (does not have to add to one)
num_users - number of users who did this task for the given subject id

The second of which is denoted by "_summary" and has the columns:
subject_id
cluster_index
mean_agreement,median_agreement - average p() over all labels
num_users

For marking files, other than polygons we have two files.
The first has the columns:
subject_id
cluster_index
most_likely_tool - markings are clustered based on shape not on tool. So what tool most likely actually corresponds to this cluster
cluster center
- for points we have x,y
- for lines we have x1,y1,x2,y2
- for rectangles we have x1,y1,x2,y2
- for ellipses we have x1,y1,r1,r2,theta
p(most_likely_tool) what percentage of users agreed on the most likely tool
p(true_positive) what percentage of users has markings in this cluster
num_users - number of users to have seen this subject

The second file is denoted by "_summary" and has the following clusters:
subject_id
median(...) - the median number of each type of markings made by each user who has seen this subject
mean_probability,median_probability - average agreement on likely a cluster is to exist (actually represent something)
mean_tool,median_tool - average agreement on what sort of 'thing' each cluster is

For polygons we also have two files have:
subject_id
cluster_index
most_likely_tool
(x,y)...(x_n,y_n) a list (in double quotation marks) giving all the points in the polygon

The second file is denoted by "_summary" has the columns:
subject_id
area(...) the area (as a percentage of the whole image) taken up by all polygons of that type.


This directory contains the aggregation results that you requested for your Zooniverse project. The file contains important information to help you understand what is in this directory. If you have any questions, please go the Aggregation talk board on the Zooniverse site.

Aggregation is the process of combining classifications/markings from multiple users for a single subject into one answer. A subject is a single image or group of images that the user  views at a time. Users will often disagree with each other - this is why every subject needs to be shown to multiple users. However, an aggregate answer based on multiple users is usually highly accurate.

Our aggregation engine is designed to work with all Panoptes projects. This means that the analysis is not necessarily optimized for your project. Many of the algorithms use values, listed below, which seem to work well in practice. However, you may want to experiment and try to find better settings. You may even want to create your own algorithms. In either case, feel free to fork the the zooniverse/aggregation repo. If you create code that you feel would benefit others, feel free to make a pull request.

The flexibility of Panoptes means that we don’t know exactly how you want to use your data. So we may provide several different files, each with a different type of analysis or summary in them. Not all of the files may be relevant to your project. Use the rest of this readme file to help understand what specific files and values you want.

If you do use any of the aggregation results, please cite Zooniverse. (We are working on publishing the papers and will update this readme accordingly.) This code is still very much being developed. There are a few things that still need to be done and these are highlighted in the documentation.

Some of the subjects will have less classifications/markings then the retirement threshold you set. These cases are most likely due to you changing the workflow while the project is live. (Since we have no way of comparing the semantics of different workflows, classifications/markings for previous workflows are ignored.) This may also be due to the fact that other than the initial task, there may be no guarantee that each user does each task (this depends on your workflow).  There has been some trouble with double counting of classifications. This has been mostly resolved by ask Martin for details.
