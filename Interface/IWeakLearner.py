from abc import abstractmethod


class IWeakLearner:
    def __init__(self):
        pass

    @abstractmethod
    def init_weak_learner(self):
        pass

    @abstractmethod
    def train(self, X):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def save_model(self, path):
        pass

    @abstractmethod
    def load_model(self, path):
        pass

    @abstractmethod
    def dump_to_pickle_file(self, file):
        pass
