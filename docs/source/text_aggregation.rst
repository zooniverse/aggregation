********************************************
Annotate and Shakespeare's World Aggregation
********************************************

Annotate and Shakespeare's World are the two projects which currently have text aggregation. They also have rectangle/box aggregation going for marking images on a page.
Aggregation for these files is run automatically on a regular basis via a crontab (ask Adam or Cam for its location). There are a couple of files involved with aggregation for these projects. Let's start with the main one.

text_aggregation.py
###################

This is the file that is called by the crontab. The format for calling this file is
./text_aggregation.py -i project_id -e environment

* project_id is either 245 (annotate) or 376 (shakespeare's world)
* environment is either production or development. This affects which databases the code tries to connect to. In development, the code relies on a local instance of both Cassandra and postgres

In text_aggregation.py, we have the class TranscriptionAPI which is a subclass of AggregationAPI (from aggregation_api.py). TranscriptionAPI has 3 main methods to call:

* __setup__ - connects to the various databases (mostly just calls __setup__ in AggregationAPI but also takes care of some things that are specific to Annotate and Shakespeare's world)
* __aggregate__ - does the actual aggregation (just calls AggregationAPI's __aggregate__)
* __summarize__ - both projects want emails sent out with the aggregation results. This happens in __summarize__

\__agregate__
*************
So what does __aggregate__ do? As with __aggregate__ in AggregationAPI, we have the following steps

* migrate annotations from postgres to Cassandra (unlike other projects with both Annotate and Shakespeare's world, we want to run aggregation even over subjects that haven't been retired yet)
* we then get the raw annotations sorted by subject id
* for both image markings and transcriptions, we run clustering algorithms
* we then retire any subjects which enough people have said have been completely transcribed

\__clustering__
***************
To aggregate both image markings and transcriptions, we run clustering algorithms. Image markings are handled as rectangles and use the standard rectangle clustering algorithm (more on this later). Annotate and Shakespeare's World use different clustering algorithms - this is because with Shakespeare's World if a user can only transcribe a small part of a line that user is encouraged to place markings just around that portion of the line. (As opposed to placing markings at the beginning and the end of the line - this makes a major difference.)
 The clustering algorithms for both projects are subclasses of clustering.py. For Annotate, the code is in annotate.py (class AnnotateClustering) and for Shakespeare's world, the code is in folger.py (class FolgerClustering). For both of these classes, the main code is in __cluster__ which is called by code in the Clustering class..