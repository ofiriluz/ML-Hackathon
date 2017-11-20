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


def float_pos_generator(mu, stddev):
    x = np.random.normal(mu, stddev)
    return x if x >= 0 else float_pos_generator(mu, stddev)


def int_pos_generator(mu, stddev):
    x = np.random.randint(max(mu-stddev, 0), high=mu+stddev)
    return x


def coerce(x):
    try:
        a = float(x)
        b = int(x)
        if a == b:
            return b
        else:
            return a
    except:
        return float(x)


def generate_random_features_typed():
    # Min, Max, Average, Count, Median, StdDev, Counter1, Counter2, Counter3, Counter4, Counter5, Counter6, MinDuration, MaxDuration, AverageDuration, CountReleases, MedianDuration, StdDevDuration, Counter11, Counter12, Counter13, Counter14, Counter15, Counter16, User
    types = [float, float, float, int, float, float, int, int, int, int, int, int, float, float, float, int, float, float, int, int, int, int, int, int]
    generators = {float: float_pos_generator, int: int_pos_generator}
    stddevs = {float: 0.01, int: 1}
    with open('C:/Users/ofiri/Desktop/Tests/B/features.txt') as f:
        lines = f.readlines()
        features = list(map(lambda x: list(map(lambda y: coerce(y), x)), [line.split(',') for line in lines]))
        generated_features = []
        generator_count = 1000
        for i in range(generator_count):
            generated_features.append([generators[type(arg)](arg, stddevs[type(arg)]) for arg in random.choice(features)])
        with open('C:/Users/ofiri/Desktop/Tests/B/generated_features.txt', 'w') as f2:
            for line in generated_features:
                f2.write(','.join([str(item) for item in line]) + '\n')


def generate_random_features():
    with open('C:/Users/ofiri/Desktop/Tests/A/features.txt') as f:
        lines = f.readlines()
        features = list(map(lambda x: list(map(lambda y: float(y), x)), [line.split(',') for line in lines]))
        generated_features = []
        generator_count = 1000
        stddev = 0.01
        for i in range(generator_count):
            generated_features.append([float_pos_generator(arg, stddev) for arg in random.choice(features)])
        with open('C:/Users/ofiri/Desktop/Tests/A/generated_features.txt', 'w') as f2:
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
    count = 30
    f = open('C:/Users/ofiri/Desktop/Tests/A/mouse_features.txt', 'w')
    for c in range(count):
        while not features:
            features = generate_features(stream, extractor)
        f.write(','.join(features) + '\n')
    f.close()
    stream.stop_stream()


def generate_keyboard_features():
    stream = KeyboardDataStream()
    extractor = KeyboardExtractor(data_chunk_duration_sec=5)
    stream.init_stream()
    features = None
    count = 30
    f = open('C:/Users/ofiri/Desktop/Tests/A/keyboard_features.txt', 'w')
    for c in range(count):
        while not features:
            features = generate_features(stream, extractor)
        f.write(','.join(features) + '\n')
    f.close()
    stream.stop_stream()


if __name__ == '__main__':
    # generate_random_features_typed()
    processor = Processor(sliding_window_frame_size=20,
                          stddev_threshold=0.1,
                          risk_iterations=5,
                          minimum_training_size=900,
                          starting_eval_size=5,
                          save_trained_model=True,
                          save_path='./',
                          user='ofir')

    # Add the weak learners
    # processor.add_weak_learner(learner=AutoEncoderNNWeakLearner(cols_shape=6),
    #                            extractor=KeyboardExtractor(),
    #                            data_stream=KeyboardDataStream())
    #
    # processor.add_weak_learner(learner=AutoEncoderNNWeakLearner(cols_shape=24),
    #                            extractor=MouseExtractor(),
    #                            data_stream=MouseStream())

    processor.load_processor('./processor_data_ofir.json', [
        {'Learner': AutoEncoderNNWeakLearner(cols_shape=24),
         'Extractor': DataReaderExtractor(),
         'Stream': DataReaderStream(data_folder='C:/Users/ofiri/Desktop/Tests/B/test'),
         'Sanitizer': None},
        {'Learner': AutoEncoderNNWeakLearner(cols_shape=6),
         'Extractor': DataReaderExtractor(),
         'Stream': DataReaderStream(data_folder='C:/Users/ofiri/Desktop/Tests/A/test'),
         'Sanitizer': None}
    ])

    processor.start_process(train=False)

    # processor.add_weak_learner(learner=AutoEncoderNNWeakLearner(cols_shape=24),
    #                            extractor=DataReaderExtractor(),
    #                            data_stream=DataReaderStream(data_folder="C:/Users/ofiri/Desktop/Tests/B"))
    #
    # processor.add_weak_learner(learner=AutoEncoderNNWeakLearner(cols_shape=6),
    #                            extractor=DataReaderExtractor(),
    #                            data_stream=DataReaderStream(data_folder="C:/Users/ofiri/Desktop/Tests/A"))

    # # Run the processor
    # processor.start_process()

    # generate_mouse_features()

