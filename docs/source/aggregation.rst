*****
Aggregation in Zooniverse Projects
*****

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