Zooniverse Aggregation Code

This repo allows you to do aggregation with Zooniverse projects. Aggregation is the process of taking classifications/markings/transcriptions from the multiple users who see each subject and combining them into a "final" answer that is useful for researchers. 
For example, if 3 out of 4 people say that an image contains a zebra, aggregation would report that there is a 75% probability that the image does indeed contain a zebra.

The directory to do all of this in "engine". This is the code base that runs every time you press "export aggregations" in the builder builder page. You can also run things locally if you want - this is especially useful if you have an Ourboros project (just ignore that if you don't already know what Ourboros is).



There are a couple of key directories in this repo. 
- engine
- active_weather
- experimental

Probably the one you are most interested in is "engine" - this is where all of the code sits for doing aggregation in Panoptes. So when you click "get aggregations" for your project in the PFE project builder, it is the code in this directory that gets run. Projects like Annotate and Shakespeare's world which have aggregation code run automatically on a regular basis (via cron jobs), also have their code in this directory (and share a lot in common with how aggregation works for other projects).

"Experimental" is where I keep all of the random code that I've worked on - this is probably in need of a major cleaning. Lots of small bits of (hopefully) really useful code here but probably not easy to find.

"Active_weather" - I've been doing long term on getting OCR to work with Old Weather (and doing things like use active learning - hence the name). 

sudo apt-get install libeigen3-dev
libjasper-dev
libdc1394-22-dev libdc1394-22 libdc1394-utils
sudo apt-get install libv4l-0 libv4l-dev
sudo apt-get install ffmpeg libavcodec-dev libavformat-dev
swig
libxine2-dev
libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libxine-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev