#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 21:53:42 2018

Allows print both in console and to file, also controls verbose of print
function by turning it on or of

@author: asceta
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys


class PrintManager(object):
    def __init__(self):
        self.original_print = print
        self.original_stdout = sys.stdout

    def verbose_printing(self, verbose=True):
        return self.original_print if verbose else lambda *a, **k: None

    def file_printing(self, file):
        sys.stdout = Tee(sys.stdout, file)

    def stop_print_in_file(self):
        sys.stdout = self.original_stdout

    def close(self):
        self.stop_print_in_file()


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the logits to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()
