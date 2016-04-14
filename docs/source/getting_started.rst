*********************************
The Zooniverse Aggregation Engine
*********************************

What is all this?
=================

This library contains everything related to the Zooniverse Aggregation Engine (except passwords). There is also code related to Active Weather (Greg's work on OCR with Old Weather logs) and experimental (where Greg made a bunch of experimental code - you can mostly ignore it but there is some useful code here and there which should eventually be documented).

This documentation was created using Sphinx following the Numpy documentation standard. A good tutorial for Sphinx is http://gisellezeno.com/tutorials/sphinx-for-python-documentation.html which is what Greg followed. To support Numpy style documentation, this project uses Napolean (https://sphinxcontrib-napoleon.readthedocs.org/en/latest/ ) A good example of Numpy style documentation is found at https://github.com/numpy/numpy/blob/master/doc/example.py

The two most important commands for updating the documentation are:

* sphinx-apidoc -f -o source/ ../engine/
* make html

A good cheat list of commands to use with rst files is found at http://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html#list-and-bullets

