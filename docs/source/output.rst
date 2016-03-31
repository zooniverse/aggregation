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
In __make_files a bunch of things happen to setup the csv output files. We have sets of files for three kinds of output.

* classification results - this is for both simple classification (whether a single answer is required or multiple ones allowed) or as a follow up to marking task
* marking results - these are the clustering results
* survey results - for survey projects

The function __classification_file_setup__ creates the csv output files for the classification tasks and writes in the headers. The function takes the following parameters

* output_directory - where to write these files
* workflow_id - the id of the workflow. Allows us to create the file names (which are based on task labels) and use the answer labels as column headers
* task_id - same reason as above
* tool_id - same reason as above - but only used for followup questions to marking tasks
* followup_id - again, same reason

We'll talk about what headers are created later.

Writing Aggregation Results Per Subject
=======================================
There are different functions that produce each set of the following outputs. Think of __subject_output__() as a dispatcher deciding which of those functions are called. We have the following parameters for __subject_output__

* subject_id - used in printing to the csv files (always the first column)
* aggregations - has all of the aggregations for this subject over all tasks for the current workflow (in dictionary format)
* task_structure - a tuple containing the classification tasks, marking tasks and survey tasks. For example, allows us to determine that task "T5" in the aggregations is a marking task. Really just self.workflows[workflow_id] - but this means we don't have to pass in workflow_id as param
* instructions - basically just self.instructions[workflow] - allows us to get the string labels for different responses. So in the csv file, instead of saying that option 0 was the most likely, we can say that "zebra" was most likely

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


The main function for producing classification results is __classification_output__ which takes the following parameters:

* workflow_id - these
* task_id
* subject_id
* aggregations,shape_id=None,followup_id=None

Marking Results
***************

