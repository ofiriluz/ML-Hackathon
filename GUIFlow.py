from tkinter import *
from tkinter import messagebox
from Processor import Processor
from KeyboardHook.KeyboardExtractor import KeyboardExtractor
from KeyboardHook.KeyboardStream import KeyboardDataStream
from MouseHook.MouseExtractor import MouseExtractor
from MouseHook.MouseStream import MouseStream
from DataReaderHook.DataReaderExtractor import DataReaderExtractor
from DataReaderHook.DataReaderStream import DataReaderStream
from AutoEncoderNN.AutoEncoderNNWeakLearner import AutoEncoderNNWeakLearner
import os
import threading


class GUIFlow:
    def state_callback(self, state):
        self.intruder_text.set(state)

    def __run_processor(self, processor):
        self.current_processor = processor
        self.current_processor.start_process(train=False, state_callback=self.state_callback)
        self.current_processor = None

    def analyze_callback(self):
        if not self.current_processor:
            processor = list(filter(lambda proc: proc.get_user() == self.user_var.get(), self.processes))[0]
            self.intruder_label.pack(pady=50)
            threading.Thread(target=self.__run_processor, args=(processor,)).start()

    def reset_callback(self):
        if self.current_processor:
            self.current_processor.stop_process()
            self.current_processor = None
        self.processes = []
        self.__init_models()
        self.intruder_label.pack_forget()

    def __init_models(self):
        for file in os.listdir(self.models_folder):
            if file.endswith(".json"):
                processor = Processor(sliding_window_frame_size=20,
                                      stddev_threshold=0.05,
                                      risk_iterations=5,
                                      minimum_training_size=900,
                                      starting_eval_size=5,
                                      save_trained_model=True,
                                      save_path='./')
                processor.load_processor(self.models_folder + '/' + file, [
                    {'Learner': AutoEncoderNNWeakLearner(cols_shape=24),
                     'Extractor': DataReaderExtractor(),
                     'Stream': DataReaderStream(data_folder='C:/Users/ofiri/Desktop/Tests/B/test'),
                     'Sanitizer': None}
                ])
                self.processes.append(processor)

    def __init__(self, models_folder='./Models'):
        self.root = Tk()
        self.root.geometry("400x200")

        self.intruder_text = StringVar(self.root)
        self.intruder_text.set("Intruder Alert!")
        self.intruder_label = Label(self.root, textvariable=self.intruder_text,
                                    font=("Helvetica", 16), bg='#ff0000', fg='#ffffff')

        self.analyze_button = Button(self.root, text="Start Analyze", command=self.analyze_callback)
        self.analyze_button.place(x=50, y=150)

        self.reset = Button(self.root, text="Reset", command=self.reset_callback)
        self.reset.place(x=200, y=150)

        # Init all the models
        self.models_folder = models_folder
        self.processes = []
        self.__init_models()

        choices = [processor.get_user() for processor in self.processes]
        self.user_var = StringVar(self.root, choices[0])
        self.options = OptionMenu(self.root, self.user_var, *choices)
        self.options.place(x=300, y=150)

        self.current_processor = None

    def run(self):
        self.root.mainloop()

flow = GUIFlow()
flow.run()
