from Adabooster import Adabooster
from Interface.IWeakLearner import IWeakLearner
from Interface.IDataStream import IDataStream
from Interface.IFeatureExtractor import IFeatureExtractor
from Interface.Stamp import Stamp
from Interface.StampedFeatures import StampedFeatures
import numpy as np
import json


class Processor:
    def __init__(self, sliding_window_frame_size=30,
                 stddev_threshold=1,
                 risk_iterations=10,
                 minimum_training_size=1000,
                 starting_eval_size=5,
                 save_trained_model=True,
                 save_path='./',
                 user=''):
        self.strong_learner = None
        self.data_processors = []
        self.sliding_window_frame_size = sliding_window_frame_size
        self.stddev_threshold = stddev_threshold
        self.risk_iterations = risk_iterations
        self.minimum_training_size = minimum_training_size
        self.starting_eval_size = starting_eval_size
        self.save_trained_model = save_trained_model
        self.save_path = save_path
        self.user = user
        self.is_running = False

    def collect_training_dataset(self):
        training_set = []
        left_over = self.minimum_training_size
        while left_over > 0:
            new_data = self.collect_next_dataset()
            if new_data:
                training_set.append(new_data)
                left_over = left_over - 1
            else:
                break
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
        return StampedFeatures(stamp=collective_stamp, data=np.array(collective_features), columns=collective_columns)

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
        self.strong_learner = Adabooster(training_set=training_data)
        for processor in self.data_processors:
            self.strong_learner.set_weak_learner(processor['Learner'])
        if self.save_trained_model:
            self.save_processor(self.save_path)

    def remove_outliers(self, x, outlier_const):
        upper_quartile = np.percentile(x, 90)
        lower_quartile = np.percentile(x, 10)
        IQR = (upper_quartile - lower_quartile) * outlier_const
        quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
        resultList = []
        for y in x:
            if y >= quartileSet[0] and y <= quartileSet[1]:
                resultList.append(y)
        return resultList

    def stop_processor(self):
        self.is_running = False

    def start_process(self, train=True, state_callback=None):
        if self.is_running:
            return
        if train:
            self.run_training_process()
        prediction_mse_window = []
        prediction_times = []
        risk_count = 0
        current_risk = "No Risk"
        self.is_running = True
        while self.is_running:
            data = self.collect_next_dataset()
            if not data:
                break
            (mse, pred) = self.strong_learner.predict_data(np.asmatrix(data.get_features()))
            print("PRED MSE = " + str(mse))
            prediction_times.append(data.get_header().get_time())
            # Evaluate a sliding window
            prediction_mse_window.append(mse[0, 0])
            if len(prediction_mse_window) == self.sliding_window_frame_size:
                prediction_mse_window.pop(0)
            # while len(prediction_mse_window) > 1:
                # time_slice = data.get_header().get_time() - prediction_times[0]
                # if time_slice > self.sliding_window_time_frame:
                #     prediction_mse_window.pop(0)
                # else:
                #     break
            prediction_mse_window = self.remove_outliers(prediction_mse_window, 1.5)
            sliding_window_stdv = np.std(prediction_mse_window)
            print(str(sliding_window_stdv))
            if len(prediction_mse_window) >= self.starting_eval_size and sliding_window_stdv > self.stddev_threshold:
                risk_count = risk_count + 1
            else:
                risk_count = max(risk_count - 1, 0)
            if risk_count >= self.risk_iterations:
                current_risk = "Risk Found"
            else:
                current_risk = "No Risk"
            if state_callback:
                state_callback(current_risk)

    def add_weak_learner(self, learner, extractor, data_stream, sanitizer=None):
        if isinstance(learner, IWeakLearner) and isinstance(data_stream, IDataStream) and isinstance(extractor, IFeatureExtractor):
            learner.init_weak_learner()
            data_stream.init_stream()
            self.data_processors.append({'Learner': learner, 'Extractor': extractor, 'Stream': data_stream, 'Sanitizer': sanitizer})
            if self.strong_learner:
                self.strong_learner.set_weak_learner(learner)

    def save_processor(self, folder_path):
        processor_state = {'sliding_window_time_frame': self.sliding_window_time_frame,
                           'stddev_threshold': self.stddev_threshold,
                           'risk_iterations': self.risk_iterations,
                           'minimum_training_size': self.minimum_training_size,
                           'starting_eval_size': self.starting_eval_size,
                           'save_trained_model': self.save_trained_model,
                           'save_path': self.save_path,
                           'user': self.user,
                           'booster': self.strong_learner.save_booster(folder_path, 'booster')}
        with open(folder_path + '/' + 'processor_data_' + self.user + ".json", 'w') as f:
            json.dump(processor_state, f, indent=4)
        return folder_path + '/' + 'processor_data_' + self.user + ".json"

    def load_processor(self, path, learners):
        with open(path, 'r') as f:
            processor_state = json.load(f)
            print(processor_state)
            self.sliding_window_time_frame = processor_state['sliding_window_time_frame']
            self.stddev_threshold = processor_state['stddev_threshold']
            self.risk_iterations = processor_state['risk_iterations']
            self.minimum_training_size = processor_state['minimum_training_size']
            self.starting_eval_size = processor_state['starting_eval_size']
            self.save_trained_model = processor_state['save_trained_model']
            self.save_path = processor_state['save_path']
            self.user = processor_state['user']
            self.strong_learner = Adabooster(None)
            for learner in learners:
                self.data_processors.append(learner)
                self.strong_learner.set_weak_learner(learner['Learner'], train=False)
            self.strong_learner.load_booster(processor_state['booster'])

    def get_user(self):
        return self.user

