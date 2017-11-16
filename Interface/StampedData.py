from Interface.Stamp import Stamp
import numpy as np


class StampedData:
    def __init__(self, stamp=Stamp(), data=np.zeros(shape=(1, 1)), columns=[]):
        self.stamp = stamp
        self.data = data
        self.columns = columns

    def set_stamp(self, stamp):
        if not isinstance(stamp, Stamp):
            return
        self.stamp = stamp

    def reshape_data(self, shape):
        if isinstance(shape, tuple) and len(shape) == 2:
            self.data = np.reshape(self.data,shape)

    def set_data(self, data):
        self.data = np.array(data)