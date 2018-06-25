'''
Created on 22 Feb 2017

@author: roxana
'''
import traceback

import sys


class TracePrints(object):
    def __init__(self):
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write("Writing %r\n" % s)
        traceback.print_stack(file=self.stdout)
