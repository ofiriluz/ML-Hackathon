from JSONSocket import Server
from Processor import Processor
from KeyboardHook.KeyboardExtractor import KeyboardExtractor
from KeyboardHook.KeyboardStream import KeyboardDataStream
from MouseHook.MouseExtractor import MouseExtractor
from MouseHook.MouseStream import MouseStream
from DataReaderHook.DataReaderExtractor import DataReaderExtractor
from DataReaderHook.DataReaderStream import DataReaderStream
from AutoEncoderNN.AutoEncoderNNWeakLearner import AutoEncoderNNWeakLearner
import json
import threading
import os


class WebFlow:
    def state_callback(self, state):
        print('State = ' + state)
        self.current_state = state
        self.server.send({'command': 'Risk', 'data': state})

    def __run_processor(self, processor):
        self.current_processor = processor
        self.current_processor.start_process(train=False, state_callback=self.state_callback)
        self.current_processor = None

    def analyze_callback(self, user):
        if not self.current_processor:
            processor = list(filter(lambda proc: proc.get_user() == user, self.processes))[0]
            threading.Thread(target=self.__run_processor, args=(processor,)).start()

    def reset_callback(self):
        if self.current_processor:
            self.current_processor.stop_process()
            self.current_processor = None
        self.processes = []
        self.__init_models()

    def __init__(self, models_folder='./ModelsComb'):
        self.server = Server('', 3500)
        self.models_folder = models_folder
        self.processes = []
        self.current_processor = None
        self.current_state = 'No Risk'
        self.__init_models()

    def __init_models(self):
        for file in os.listdir(self.models_folder):
            if file.endswith(".json"):
                processor = Processor(sliding_window_frame_size=20,
                                      stddev_threshold=0.1,
                                      risk_iterations=5,
                                      minimum_training_size=900,
                                      starting_eval_size=5,
                                      save_trained_model=True,
                                      save_path='./')
                processor.load_processor(self.models_folder + '/' + file, [
                    {'Learner': AutoEncoderNNWeakLearner(cols_shape=24),
                     'Extractor': DataReaderExtractor(),
                     'Stream': DataReaderStream(data_folder='C:/Users/ofiri/Desktop/Tests/B/test'),
                     'Sanitizer': None},
                    {'Learner': AutoEncoderNNWeakLearner(cols_shape=6),
                     'Extractor': DataReaderExtractor(),
                     'Stream': DataReaderStream(data_folder='C:/Users/ofiri/Desktop/Tests/A/test'),
                     'Sanitizer': None}
                ])
                self.processes.append(processor)

    def start(self):
        while True:
            try:
                self.server.accept()
                self.server.send({'command': 'Users', 'data': [processor.get_user() for processor in self.processes]})
                while True:
                    json_data = self.server.recv()
                    print(json_data)
                    if json_data['command'] == 'Analyze':
                        self.analyze_callback(json_data['data'])
                    elif json_data['command'] == 'Reset':
                        self.reset_callback()
            except:
                print('Error')
                continue

flow = WebFlow()
flow.start()
