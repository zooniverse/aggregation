******************
Aggregation Output
******************

After all the aggregation clustering and classification has taken place, the final step is output a bunch of csv files with the results. This code is in csv_output.py

In csv_output.py, we have the class CsvOut. Its constructor takes just param - the project which is an instance of AggregationAPI. The main function to call in CsvOut is __write_out__ which takes one function

* subject_set - the set of subjects for which we want the aggregation results. We can set this to be None (the default) in which case we get aggregations for all subjects, even if they weren't updated during the aggregation run. This is usually what we want - really, specifying subject_set is mostly a development enviromnent thing.

The __write_out__ function returns one value

* tar_file_path - the path to the zipped file of results - this is what gets returned to the user in the end

The __write_out__ function goes through each workflow independently. Each workflow is processed in basically two steps

* __make_files__ creates the csv output files (with headers) for each task in the workflow. There are usually summary files and detailed results files.
* we then iterate over the results for this workflow, subject by subject, by calling the projects' __yield_aggregations__ generator. For each subject, we then call __subject_output__ to convert the results into the csv output.

Creating files
==============
In __make_files__ a bunch of things happen to setup the csv output files. We have sets of files for three kinds of output.

* classification results - this is for both simple classification (whether a single answer is required or multiple ones allowed) or as a follow up to marking task
* marking results - these are the clustering results
* survey results - for survey projects

The function __classification_file_setup__ creates the csv output files for the classification tasks and writes in the headers. The function takes the following parameters

* output_directory - where to write these files
* workflow_id - the id of the workflow. Allows us to create the file names (which are based on task labels) and use the answer labels as column headers

We'll talk about what headers are created later. This function creates both the detailed csv files and the summary ones as well.

The marking csv files (both detailed and summary) are created by __marking_file_setup__ with the same parameters as __classification_file_setup__. The function __marking_file_setup__ calls a couple of sub functions

* __marking_summary_header__ - creates the headers for all csv files except for polygon markings
* __polygon_summary_header__ - creates the headers for polygon markings
* __marking_detailed_header__ - creates the headers for the detailed files except for polygons
* __polygon_detailed_header__ - you can probably guess :P

The above functions have the following parameters in common

* shape - if not polygon
* workflow_id - for summary output, need to access labels so you can get understandable column names
* task_id

Detailed Markings Output Files
******************************
With the detailed markings output files, (except for polygon output - those will be explained later) we have the following headers

* subject_id
* cluster_index - there is no guaranteed ordering of the indices (such as left or right) but allows us to match up the results from follow up questions
* most_likely_tool - what do we think the most likely tool is

Then we have some columns which are dependent on the shape used

* for points we have x,y
* for rectangles we have x1,y1,x2,y2
* for lines we have x1,y1,x2,y2
* for ellipses we have x1,y1,r1,r2,theta

And finally we have more columns which all markings have

* p(most_likely_tool) - what percentage of users agreed about the tool
* p(true_positive) - what percentage of users have markings in the cluster (so how likely is the cluster to something that actually exists, as opposed to someone mistaking some snow and rocks for penguins)
* num_users - how many users saw this subject for this particular task


For polygons, we have the following detailed output

* subject_id
* cluster_index
* most_likely_tool
* area - as a percentage of the image size
* list_of_xy_polygon_coordinates - since this is going to be a list of arbitrary size - this will be a list enclosed by quotation marks

Summary Markings Output Files
*****************************
With the summary markings output files (again except polygons) we have the following headers

* subject_id
* median(tool) - [need to double check about this]
* mean_probability - the mean percentage of users with markings in each cluster - useful for determining how much agreement there is
* median_probability - same as above but with median
* mean_tool - how agreement there is, on average, with the most likely tool. [double check]
* median_tool - same as above but with median

For polygons we have

* subject_id
* area(tool) - for each tool, what percentage of the image is covered by that tool


Writing Aggregation Results Per Subject
=======================================
There are different functions that produce each set of the following outputs. Think of __subject_output__() as a dispatcher deciding which of those functions are called. We have the following parameters for __subject_output__

* subject_id - used in printing to the csv files (always the first column)
* aggregations - has all of the aggregations for this subject over all tasks for the current workflow (in dictionary format)
* workflow_id - allows us to get the tasks for the given workflow and the instructions as well (which allow us to give proper labels for the outputs). I had thought about passing in the list of tasks and instructions - this would have meant we didn't need to pass in the workflow_id (seemed a bit cleaner) but would have meant passing in more parameters which seemed counter-productive

Classification Results
**********************
Classification results for both simple tasks (is there a zebra in this image) and followup tasks (describe what you just clicked on) are dealt with in pretty similar manners. Every classification task has two output files

* summary - a brief overview, giving things like the most likely answer and the Shannon entropy
* detailed - giving more detailed results such as a break down in how many people chose each result

The files are identical whether only one answer is allowed or multiple are allowed. This means that, for a particular task, there are probably columns which aren't relevant. It is up to the researchers to make sure that they know which columns they need,

As mentioned __classification_file_setup__ creates the csv files for a given workflow,task (and possibly tool and followup question id). This function calls two other functions which setup the summary csv file and the detailed csv file.
The summary csv file is created by __summary_classification_file_setup__. Each summary file has the following columns (again not every column is going to be applicable for every task)

* subject_id
* cluster_id - only if this is a follow up classification
* most_likely - what was the most likely classification (right now, we only support a simple voting based classification so most likely is really just most popular)
* p(most_likely) - what percentage of people voted for the most likely (in the far future we might support IBCC, in which case "percentage" might be a slight abuse of terms - don't worry if you don't know what IBCC is)
* shannon entropy - what the shannon entropy is (a good measure of agreement amongst users if only one answer is allowed)
* mean agreement - the mean percentage of people who voted for each classification (a good measure of agreement when multiple answers are allowed)
* median agreement - same as above but median
* num_users - how many people classified this task for this subject

There are two functions for producing the classification output (for simple classifications) - __add_detailed_row__ and __add_summary_row__. Both are called from __subject_output__.

The detailed results file contains the following columns

* subject_id
* cluster_id - only if this is a follow up classification
* p(answer) - what percentage of people chose this particular answer


The main function for producing classification results is __classification_output__ which takes the following parameters:

* workflow_id - these first 3 parameters are used for access csv files (the csv files are stored in a dictionary where the keys are tuples containing the workflow_id
* task_id
* subject_id
* aggregations,shape_id=None,followup_id=None


Classification Summary Results
==============================
To add a single row of summary results for a classification task (whether simple of a followup question), we use __add_summary_row__ which takes the following parameters

* workflow_id,task_id,subject_id - the usual ones for accessing the right csv output file and getting the labels for different answers
* aggregations -
* shape_id=None,followup_id=None - set if this is a follow up classification task

This function basically looks at the aggregations and determines the most likely classification, converts it into a label (which the researchers provide) and writes that out. We also get the Shannon entropy (useful when only one answer is allowed) and mean/median percentage (useful when multiple answers are allowed)

Detailed Classification Results
===============================

The detailed classification results are written out using __add_detailed_row__ which uses the same parameters as __add_summary_row__.

Marking Results
***************

Lines for summary results of markings are produced by two function

* __add_marking_summary_row__
* __add_polygon_summary_row__ - since polygons are different

Results for follow up questions to markings are produced by __add_marking_followup_rows__. This function takes the usual params

* workflow_id,task_id,subject_id,aggregations

Detailed results - where results are given for each individual cluster - are created by __detailed_marking_row__ which takes the following parameters

* workflow_id,task_id,subject_id,aggregations,shape

So for a given shape, __detailed_marking_row__ will output all the clusters of that shape (for the given workflow and task and subject ids).

Survey Results
**************

For survey results, we just have one csv output file. We also assume that a survey task cannot contain any marking tasks (follow up questions are fine) - a separate task in the workflow can contain a marking task so this seems like a pretty reasonable assumption.
Since we only have one file to create, this is all done in one function __survey_file_setup__ which takes two parameters

* output_directory
* workflow_id