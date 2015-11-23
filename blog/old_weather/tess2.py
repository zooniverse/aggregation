#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2012-2013 Zdenko Podobný
# Author: Zdenko Podobný
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple python demo script of tesseract-ocr 3.02 c-api and filehandle
"""

import os
import sys
import ctypes
from ctypes import pythonapi, util, py_object

# Demo variables
lang = "eng"
output = "dump.config"
filename = "/home/ggdhines/github/tesseract/testing/phototest.tif"
libpath = "/usr/lib/"
libpath_w = "../vs2008/DLL_Release/"
tessdata = "/usr/share/tesseract-ocr/"

if sys.platform == "win32":
    libname = libpath_w + "libtesseract302.dll"
    libname_alt = "libtesseract302.dll"
    os.environ["PATH"] += os.pathsep + libpath_w
else:
    libname = libpath + "libtesseract.so.3.0.2"
    libname_alt = libpath + "libtesseract.so.3"

try:
    tesseract = ctypes.cdll.LoadLibrary(libname)
except:
    try:
        tesseract = ctypes.cdll.LoadLibrary(libname_alt)
    except:
        print("Trying to load '%s'..." % libname)
        print("Trying to load '%s'..." % libname_alt)
        exit(1)

tesseract.TessVersion.restype = ctypes.c_char_p
tesseract_version = tesseract.TessVersion()

# We need to check library version because libtesseract.so.3 is symlink
# and can point to other version than 3.02

api = tesseract.TessBaseAPICreate()

rc = tesseract.TessBaseAPIInit3(api, tessdata, lang)
if (rc):
    tesseract.TessBaseAPIDelete(api)
    print("Could not initialize tesseract.\n")
    exit(3)

# Tested in linux - may cause problems on Windows.
fh = open(output,'wb')
PyFile_AsFile = pythonapi.PyFile_AsFile
PyFile_AsFile.argtypes = [ctypes.py_object]
PyFile_AsFile.restype = ctypes.c_void_p

t = tesseract.TessBaseAPIPrintVariables(api, PyFile_AsFile(fh))

# text_out = tesseract.TessBaseAPIProcessPages(api, filename, None, 0)
mBuffer=open(filename,"rb").read()
text_out = tesseract.ProcessPagesBuffer(mBuffer,len(mBuffer),api)
result_text = ctypes.string_at(text_out)

print result_text
fh.close()