from Adabooster import Adabooster
from Interface.IWeakLearner import IWeakLearner
from Interface.IDataStream import IDataStream
from Interface.IFeatureExtractor import IFeatureExtractor
import numpy as np

class Processor:
    def __init__(self, sliding_window_time_frame=30,
                 stddev_threshold=1,
                 risk_iterations=10,
                 minimum_training_size=1000):
        self.strong_learner = None
        self.data_processors = []
        self.sliding_window_time_frame = sliding_window_time_frame
        self.stddev_threshold = stddev_threshold
        self.risk_iterations = risk_iterations
        self.minimum_training_size = minimum_training_size

    def collect_training_dataset(self):
        training_set = []
        left_over = self.minimum_training_size
        while left_over > 0:
            new_data = self.collect_next_dataset()
            if new_data:
                training_set.append(self.collect_next_dataset())
                left_over = left_over - 1
            else:
                break
        return training_set

    def collect_next_dataset(self):
        # Build a dataset from all the weak learners data streams
        dataset = []
        for stream in self.data_processors:
            # Get Raw Data
            raw_data = stream['Stream'].get_next_stamped_data()
            # Sanitize Data if sanitizer exists
            if stream['Sanitizer']:
                raw_data = stream['Sanitizer'].sanitize_data(raw_data)
            # Extract Features
            features = stream['Extractor'].extract_features(raw_data)
            dataset.append(features)
        return dataset

    def run_training_process(self):
        training_datasets = self.collect_training_dataset()
        self.strong_learner = Adabooster(training_set=training_datasets)
        for processor in self.data_processors:
            self.strong_learner.set_weak_learner(processor['Learner'])

    def start_process(self):
        self.run_training_process()
        prediction_window = []
        risk_count = 0
        while True:
            data = self.collect_next_dataset()
            pred = self.strong_learner.predict_data(data)
            # Evaluate a sliding window
            prediction_window.append(pred)
            while len(prediction_window) > 1:
                time_slice = data.get_header().get_timestamp() - prediction_window[0]
                if time_slice > self.sliding_window_time_frame:
                    prediction_window.pop(0)
                else:
                    break
            sliding_window_stdv = np.std(prediction_window)
            if sliding_window_stdv > self.stddev_threshold:
                risk_count = risk_count + 1
            else:
                risk_count = 0
            if risk_count >= self.risk_iterations:
                print("WE HAVE AN ISSUE HERE")

    def add_weak_learner(self, learner, extractor, data_stream, sanitizer=None):
        if isinstance(learner, IWeakLearner) and isinstance(data_stream, IDataStream) and isinstance(extractor, IFeatureExtractor):
            self.data_processors.append({'Learner': learner, 'Extractor': extractor, 'Stream': data_stream, 'Sanitizer': sanitizer})
            if self.strong_learner:
                self.strong_learner.set_weak_learner(learner)