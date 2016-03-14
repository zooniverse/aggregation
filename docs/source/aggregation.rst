**********************************
Aggregation in Zooniverse Projects
**********************************

So you've clicked on the "get aggregations results" button on the PFE project builder page. What happens next?

There are several files that are involved in getting the results. The first few are:

* jobs.py
* job_runner.py
* web_api.py

These files connect the PFE to the aggregation engine - they are what get the aggregation engine running and put the results in AWS S3 so that people can get them.

The next file involved is

* aggregation_apy.py

In this file is the class AggregationAPI which is the base class for the aggregation engine (even if you are running the aggregation for Annotate or Shakespeare's world which we'll get to later). AggregationAPI makes use of a bunch of other files. For example

* classification.py
* panoptes_ibcc.py

This takes care of aggregating all of the classification tasks in a project (e.g. when you are asked to click on a value - either single or multiple choice, also deals with some follow up stuff for marking tasks). There are basically two different ways that we can aggregate classification results. One is simple vote counting. The other is using IBCC. IBCC isn't officially supported and can't actually be used through the PFE but the panoptes_ibcc.py file contains what you need to get started with using ibcc (this file is probably decently out of date). Next is dealing with the marking tasks.

* clustering.py
* agglomerative.py
* blob_clustering.py
* rectangle_clustering.py

These files all deal with different types of marking tasks. In clustering.py, we have the base class Cluster which all of the other clustering files make use out of. If you have a point, line or ellipse marking task, you will use agglomerative.py. For polygons, blob_clustering (originally blob_clustering also took care of rectangles hence the slightly ambiguous name which should probably updated.) Finally, for rectangles, we have rectangle_clustering.

************************
Aggregation Step by Step
************************

Let's get into a bit more detail. Once you have created an instance of AggregationAPI (more details later) there are two functions that you need to call in AggregationAPI.

* __setup__
* __aggregate__

The first one sets up a bunch of things including connecting to the Cassandra and Postgres DB (not sure exactly why this isn't just called from within the __init__ method but seem to remember some reason - probably worth double checking). Each workflow is aggregated independently using the following steps:

* Get ids of subjects to aggregate for given workflows
* If a workflow has a survey task, do survey aggregation (assuming that for now a workflow cannot have both survey and classification or marking tasks)
* If a workflow has a marking task(s), do marking aggregation
* Do classification aggregation - will always happen since it includes follow up stuff for marking tasks


*******************
Sorting Annotations
*******************

Before we aggregate for a given set of subjects, we need to sort the annotations. Annotations are originally sorted by time - i.e. when they come in, but we need them sorted by subject id. That way we can process each subject independently. The function to do this is::

    def __sort_annotations__(self,workflow_id,subject_set):

This function will take all the annotations (classifications,markings,transcriptions and survey results) for a given workflow and subject set and returned those annotations sorted by subject id.