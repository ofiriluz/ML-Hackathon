from abc import abstractmethod


class IWeakLearner:
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def init_weak_learner(self):
        pass

    @abstractmethod
    def train(self, input):
        pass

    @abstractmethod
    def validate(self, input):
        pass

    @abstractmethod
    def test(self, input):
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