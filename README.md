Zooniverse Aggregation Code
(mostly for Panoptes)

There are a couple of key directories in this repo. 
- engine
- active_weather
- experimental

Probably the one you are most interested in is "engine" - this is where all of the code sits for doing aggregation in Panoptes. So when you click "get aggregations" for your project in the PFE project builder, it is the code in this directory that gets run. Projects like Annotate and Shakespeare's world which have aggregation code run automatically on a regular basis (via cron jobs), also have their code in this directory (and share a lot in common with how aggregation works for other projects).

"Experimental" is where I keep all of the random code that I've worked on - this is probably in need of a major cleaning. Lots of small bits of (hopefully) really useful code here but probably not easy to find.

"Active_weather" - I've been doing long term on getting OCR to work with Old Weather (and doing things like use active learning - hence the name). 
