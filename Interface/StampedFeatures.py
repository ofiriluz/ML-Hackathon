from Interface.Stamp import Stamp
import numpy as np


class StampedFeatures:
    def __init__(self, stamp=Stamp(), data=np.zeros(shape=(1, 1)), columns=[]):
        self.stamp = stamp
        self.data = data
        self.columns = columns

    def get_header(self):
        return self.stamp

    def get_features(self):
        return self.data

    def get_columns(self):
        return self.columns
