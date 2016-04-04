Error Handling
##############

Sometimes things go wrong. If an error is raised during an aggregation run, that error is reported via Rollbar. Rollbar will report via email to Greg, Cam and Adam and automatically open a new issue in the Aggregation Github repo.
These error reports will contain the project id of the project that caused the error - this is useful for debugging and letting people know that there was a problem with their run.

If a run is submitted via the PFE project builder, error handling is setup in job_runner.py. Part of that code loads the Rollbar token from the aggregation.yml file. So if for whatever reason, the Rollbar token changes, we need to change that yml file.

Since annotate and folger are run via the crontab, that error handling is slightly different. That is done in the __exit__() function in TranscriptionAPI.

There currently is no way of knowing if deploying new code has killed an aggregation run. Those runs will just disappear and not raise any errors. To help with those cases, every run stores its pid (process id in Unix) and project id in the Cassandra database. (Since the Cassandra db is not in the same docker image as the aggregation code, a new deploy of the aggregation code will not affect the Cassandra DB.)
The table where these tuples are stored is running_processes. When a run finishes, one of the last things it does is remove its entry from running_processes. There is a spot check (run by spot_check.py) which runs every hour. If there is a project listed as running but that pid is not actually in use, an email is sent out (currently just to Greg) saying which project was affected.
Anyone other than admin can only request an aggregation run once every 24 hours. So currently they will still have to wait before being able to rerun their aggregation code. Not ideal but better than nothing. One possibility is for spot_check to automatically restart any interrupted runs (not sure how hard that would be).
