from abc import abstractmethod


class IFeatureExtractor:
    def __init__(self):
        pass

    @abstractmethod
    def extract_features(self, data_vector):
        pass

    @abstractmethod
    def can_be_extracted(self, data):
        pass
