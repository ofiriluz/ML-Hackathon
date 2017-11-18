from Interface.IFeatureExtractor import IFeatureExtractor
from Interface.StampedFeatures import StampedFeatures
from Interface.Stamp import Stamp
import ctypes
import numpy as np


class DataReaderExtractor(IFeatureExtractor):
    def __init__(self):
        super().__init__()

    def extract_features(self, data_vector):
        return StampedFeatures(stamp=Stamp('KeyboardHook', user=ctypes.windll.user32),
                               data=np.array(data_vector),
                               columns=['Feature-'+str(i+1) for i in range(len(data_vector))])

    def can_be_extracted(self, data):
        return True
