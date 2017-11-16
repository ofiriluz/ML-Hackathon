from Adabooster import Adabooster
from Interface.IWeakLearner import IWeakLearner
from Interface.IDataStream import IDataStream
from Interface.IFeatureExtractor import IFeatureExtractor

class Processor:
    def __init__(self):
        self.strong_learner = None
        self.data_processors = []

    def collect_training_data(self):
        pass

    def run_training_process(self):
        training_data = self.collect_training_data()
        self.strong_learner = Adabooster(training_set=training_data)
        for processor in self.data_processors:
            self.strong_learner.set_weak_learner(processor['Learner'])

    def start(self):


    def add_weak_learner(self, learner, extractor, data_stream):
        if isinstance(learner, IWeakLearner) and isinstance(data_stream, IDataStream) and isinstance(extractor, IFeatureExtractor):
            self.data_processors.append({'Learner': learner, 'Extractor': extractor, 'Stream': data_stream})
            if self.strong_learner:
                self.strong_learner.set_weak_learner(learner)