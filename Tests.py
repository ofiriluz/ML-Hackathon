from Processor import Processor
from KeyboardHook.KeyboardExtractor import KeyboardExtractor
from KeyboardHook.KeyboardStream import KeyboardDataStream
from MouseHook.MouseExtractor import MouseExtractor
from MouseHook.MouseStream import MouseStream
from DataReaderHook.DataReaderExtractor import DataReaderExtractor
from DataReaderHook.DataReaderStream import DataReaderStream
from AutoEncoderNN.AutoEncoderNNWeakLearner import AutoEncoderNNWeakLearner
import numpy as np
import random


def pos_generator(mu, stddev):
    x = np.random.normal(mu, stddev)
    return x if x >= 0 else pos_generator(mu, stddev)


def generate_random_features():

    #with open('C:/Users/ofiri/Desktop/Tests/features.txt') as f:
    with open('C:/Users/Jony/RDPML_Tests/features.txt') as f:
        lines = f.readlines()
        features = list(map(lambda x: list(map(lambda y: float(y), x)), [line.split(',') for line in lines]))
        generated_features = []
        generator_count = 1000
        stddev = 0.01
        for i in range(generator_count):
            generated_features.append([pos_generator(arg, stddev) for arg in random.choice(features)])
        #with open('C:/Users/ofiri/Desktop/Tests/generated_features.txt', 'w') as f2:
        with open('C:/Users/Jony/RDPML_Tests/generated_features.txt', 'w') as f2:
            for line in generated_features:
                f2.write(','.join([str(item) for item in line]) + '\n')


def generate_features(stream, extractor):
    data_set = []
    next_data = stream.get_next_stamped_data()
    if not next_data:
        return None
    data_set.append(next_data)
    while not extractor.can_be_extracted(data_set):
        next_data = stream.get_next_stamped_data()
        if not next_data:
            continue
        data_set.append(next_data)
    return extractor.extract_features(data_set)


def generate_mouse_features():
    stream = MouseStream()
    extractor = MouseExtractor(data_chunk_duration_sec=5)
    stream.init_stream()
    features = None
    count = 3
    #f = open('C:/Users/ofiri/Desktop/Tests/mouse_features.txt', 'w')
    f = open('C:/Users/Jony/RDPML_Tests/mouse_features.txt', 'w')
    for c in range(count):
        features = None
        while not features:
            features = generate_features(stream, extractor)
        print("FEATURES=")
        print(features.get_features())
        print([str(x) for x in features.get_features().tolist()])
        f.write(','.join([str(x) for x in features.get_features().tolist()]) + '\n')
    f.close()
    stream.stop_stream()


def generate_keyboard_features():
    stream = KeyboardDataStream()
    extractor = KeyboardExtractor(data_chunk_duration_sec=5)
    stream.init_stream()
    features = None
    count = 30
    #f = open('C:/Users/ofiri/Desktop/Tests/keyboard_features.txt', 'w')
    f = open('C:/Users/Jony/RDPML_Tests/keyboard_features.txt', 'w')
    for c in range(count):
        while not features:
            features = generate_features(stream, extractor)
        f.write(','.join(features) + '\n')
    f.close()
    stream.stop_stream()


if __name__ == '__main__':
    # processor = Processor(sliding_window_time_frame=30,
    #                       stddev_threshold=1,
    #                       risk_iterations=5,
    #                       minimum_training_size=900,
    #                       save_trained_model=True,
    #                       save_path='./')

    # Add the weak learners
    # processor.add_weak_learner(learner=AutoEncoderNNWeakLearner(cols_shape=6),
    #                            extractor=KeyboardExtractor(),
    #                            data_stream=KeyboardDataStream())
    #
    # processor.add_weak_learner(learner=AutoEncoderNNWeakLearner(cols_shape=24),
    #                            extractor=MouseExtractor(),
    #                            data_stream=MouseStream())

    # processor.add_weak_learner(learner=AutoEncoderNNWeakLearner(cols_shape=6),
    #                            extractor=DataReaderExtractor(),
    #                            #data_stream=DataReaderStream(data_folder="C:/Users/ofiri/Desktop/Tests"))
    #                            data_stream=DataReaderStream(data_folder="C:/Users/Jony/RDPML_Tests"))

    # Run the processor
    #processor.start_process()

    generate_mouse_features()

