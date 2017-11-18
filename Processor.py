from Adabooster import Adabooster
from Interface.IWeakLearner import IWeakLearner
from Interface.IDataStream import IDataStream
from Interface.IFeatureExtractor import IFeatureExtractor
from Interface.Stamp import Stamp
from Interface.StampedFeatures import StampedFeatures
import numpy as np


class Processor:
    def __init__(self, sliding_window_time_frame=30,
                 stddev_threshold=1,
                 risk_iterations=10,
                 minimum_training_size=1000,
                 save_trained_model=True,
                 save_path='model'):
        self.strong_learner = None
        self.data_processors = []
        self.sliding_window_time_frame = sliding_window_time_frame
        self.stddev_threshold = stddev_threshold
        self.risk_iterations = risk_iterations
        self.minimum_training_size = minimum_training_size
        self.save_trained_model = save_trained_model
        self.save_path = save_path

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
        print("FINAL SET = ")
        print(training_set)
        return training_set

    def unite_features(self, dataset):
        collective_features = []
        collective_columns = []
        collective_stamp = Stamp(user=dataset[0].get_header().get_user())
        for item in dataset:
            collective_features.extend(item.get_features().tolist())
            collective_columns.extend(item.get_columns())
            collective_stamp.set_source(collective_stamp.get_source() + "_" + item.get_header().get_source())
            collective_stamp.set_time(min(collective_stamp.get_time(), item.get_header().get_time()))
        print("COLLECTIVE = " + str(collective_features))
        return StampedFeatures(stamp=collective_stamp, data=collective_features, columns=collective_columns)

    def collect_next_dataset(self):
        # Build a dataset from all the weak learners data streams
        dataset = []
        for stream in self.data_processors:
            current_data_set = []
            next_data = stream['Stream'].get_next_stamped_data()
            if not next_data:
                break
            current_data_set.extend(next_data)
            while not stream['Extractor'].can_be_extracted(current_data_set):
                next_data = stream['Stream'].get_next_stamped_data()
                if not next_data:
                    break
                current_data_set.extend(next_data)
            # Sanitize Data if sanitizer exists
            if stream['Sanitizer']:
                current_data_set = stream['Sanitizer'].sanitize_data(current_data_set)
            features = stream['Extractor'].extract_features(current_data_set)
            dataset.append(features)
        if len(dataset) == 0:
            return None
        return self.unite_features(dataset)

    def run_training_process(self):
        training_datasets = self.collect_training_dataset()
        training_data = [item.get_features() for item in training_datasets]
        print(training_data)
        self.strong_learner = Adabooster(training_set=training_data)
        for processor in self.data_processors:
            self.strong_learner.set_weak_learner(processor['Learner'])
        if self.save_trained_model:
            self.strong_learner.save_model(self.save_path)

    def start_process(self):
        self.run_training_process()
        prediction_window = []
        risk_count = 0
        while True:
            data = self.collect_next_dataset()
            pred = self.strong_learner.predict_data(data.get_features())
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
            learner.init_weak_learner()
            data_stream.init_stream()
            self.data_processors.append({'Learner': learner, 'Extractor': extractor, 'Stream': data_stream, 'Sanitizer': sanitizer})
            if self.strong_learner:
                self.strong_learner.set_weak_learner(learner)
