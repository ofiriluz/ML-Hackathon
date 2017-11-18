from Interface.IFeatureExtractor import IFeatureExtractor
from Interface.StampedFeatures import StampedFeatures
from Interface.Stamp import Stamp
import numpy as np
import ctypes


class KeyboardExtractor(IFeatureExtractor):
    def __init__(self, data_chunk_duration_sec=10):
        super().__init__()
        self.data_chunk_duration_sec = data_chunk_duration_sec

    def extract_features(self, data_vector):
        curr_min = 999999999
        curr_max = 0
        curr_sum = 0
        num_keys = len(data_vector)
        diffs_list = []
        for k in range(0, num_keys - 1):
            curr_diff = (data_vector[k + 1][1] - data_vector[k][1])
            diffs_list.append(curr_diff)
            curr_min = min(curr_min, curr_diff)
            curr_max = max(curr_max, curr_diff)
            curr_sum = curr_sum + curr_diff
        features = [curr_min, curr_max, curr_sum / num_keys, num_keys, np.median(diffs_list), np.std(diffs_list)]
        return StampedFeatures(stamp=Stamp('KeyboardHook', user=ctypes.windll.user32),
                               data=np.array(features),
                               columns=['MinDiff', 'MaxDiff', 'AverageDiff', 'NumKeys', 'MedianDiffs', 'StdDiff', 'User'])

    def can_be_extracted(self, data):
        first_timestamp = data[0][2]
        last_timestamp = data[-1][2]
        return last_timestamp - first_timestamp >= self.data_chunk_duration_sec*1000
