import os,sys,time
import numpy as np

class Base:

    def __init__(self):
        self.category='all'

    def fit(self, data):
        pass
        #return self

    def is1d(self):
        return True
