from Interface.IFeatureExtractor import IFeatureExtractor
from Interface.StampedFeatures import StampedFeatures
from Interface.Stamp import Stamp
import ctypes
import math
import numpy as np

class MouseExtractor(IFeatureExtractor):
    def __init__(self, data_chunk_duration_sec=10, data_availability_allowed_ratio=0.5, inactivity_threshold_samples=100):
        super().__init__()
        self.data_chunk_duration_sec = data_chunk_duration_sec
        self.data_availability_allowed_ratio = data_availability_allowed_ratio
        self.inactivity_threshold_samples = inactivity_threshold_samples

        # Get screen dimensions for division to areas
        user32 = ctypes.windll.user32
        self.screen_width = user32.GetSystemMetrics(0)
        self.screen_height = user32.GetSystemMetrics(1)

        self.areas_matrix_rows, self.areas_matrix_cols, = 4, 6
        self.areas_hits_matrix = np.zeros(shape=(self.areas_matrix_rows,self.areas_matrix_cols))
        self.areas_importance_matrix = np.zeros(shape=(self.areas_matrix_rows,self.areas_matrix_cols))

        self.inactivity_counter = 0
        self.distance_vec_per_seconds = []
        self.angle_vec_per_seconds = []

    def can_be_extracted(self, data):
        first_timestamp = data[0][2]
        last_timestamp = data[-1][2]
        is_full_length = (last_timestamp - first_timestamp) >= self.data_chunk_duration_sec*1000
        if not is_full_length:
            return False

        # Inactivity detection loop
        prev_item = (-1, -1, -1)  # First time init
        data_available = []
        for (x, y, millis) in data:
            x_prev = prev_item[0]
            y_prev = prev_item[1]
            millis_prev = prev_item[2]
            is_same = self.is_same_location(x, y, x_prev, y_prev)
            if is_same:
                self.inactivity_counter += 1
            else:
                if self.is_currently_active() and self.inactivity_counter > 0:
                    for i in range(self.inactivity_counter):
                        data_available.append((x_prev, y_prev, millis_prev))
                self.inactivity_counter = 0
                data_available.append((x, y, millis))
            # Update prev item to be the last one
            prev_item = (x, y, millis)

        return len(data_available) > len(data) * self.data_availability_allowed_ratio

    def extract_features(self, data_vector):
        self.areas_hits_matrix = np.zeros(shape=(self.areas_matrix_rows, self.areas_matrix_cols))
        total_moves = 0

        # Main loop
        prev_item = (-1, -1, -1)  # First time init
        prev_timestamp = -1
        curr_second_distance = []
        curr_second_angles = []
        total_seconds_count = 0
        for (x, y, millis) in data_vector:
            if prev_timestamp == -1:
                prev_timestamp = millis
            timestamp = millis

            # Check if reached a new second of raw data
            if timestamp - prev_timestamp >= 1000:
                prev_timestamp = timestamp
                self.distance_vec_per_seconds.append(self.calculate_distance_per_second(curr_second_distance))
                self.angle_vec_per_seconds.append(self.calculate_avg_angle_per_second(curr_second_angles))
                curr_second_distance.clear()
                curr_second_angles.clear()
                total_seconds_count += 1
                # If we have calculated data for the entire duration of the data,
                # truncate possible leftovers and continue with processing the final results
                if total_seconds_count == self.data_chunk_duration_sec:
                    break

            x_prev = prev_item[0]
            y_prev = prev_item[1]
            is_same = self.is_same_location(x, y, x_prev, y_prev)
            # Ignore first sample
            if prev_item != (-1, -1, -1):
                self.add_distance_measurement(x, y, x_prev, y_prev, curr_second_distance)
                if not is_same:
                    self.add_angle_measurement(x, y, x_prev, y_prev, curr_second_angles)
            self.update_area_hit_count(x, y)
            total_moves += 1
            # Update prev item to be the last one
            prev_item = (x, y, millis)

        # Add leftovers from last second if necessary
        if total_seconds_count == self.data_chunk_duration_sec - 1:
            self.distance_vec_per_seconds.append(self.calculate_distance_per_second(curr_second_distance))
            self.angle_vec_per_seconds.append(self.calculate_avg_angle_per_second(curr_second_angles))

        self.build_importance_matrix(total_moves)
        # Temp: print matrices:
        print("\n\nAreas Hits:\n")
        self.print_matrix(self.areas_hits_matrix, self.areas_matrix_rows, self.areas_matrix_cols)
        print("\n\nAreas Importance:\n")
        self.print_matrix(self.areas_importance_matrix, self.areas_matrix_rows, self.areas_matrix_cols)
        areas_importance_list =  self.matrix_to_array(self.areas_importance_matrix)
        final_list = areas_importance_list
        final_list.extend(self.distance_vec_per_seconds)
        final_list.extend(self.angle_vec_per_seconds)
        print("Distances list:\n")
        print(self.distance_vec_per_seconds)
        print("Angles list:\n")
        print(self.angle_vec_per_seconds)
        print("Importance list:\n")
        print(areas_importance_list)
        print("Final List:\n")
        print(final_list)
        # Clear for next extract iteration before returning
        self.clear_state()
        print("Final List:\n")
        print(final_list)
        return StampedFeatures(stamp=Stamp('MouseHook', user=ctypes.windll.user32),
                               data=np.array(final_list))
                               # columns=['AreaImportance-' +
                               #          str(np.ceil(i / self.areas_matrix_cols) + 1) + ',' +
                               #          str(i % self.areas_matrix_cols + 1) for i in range(len(areas_importance_list))])

    def update_area_hit_count(self, x, y):
        area_x = int(math.floor((y * self.areas_matrix_rows) / self.screen_height))
        area_y = int(math.floor((x * self.areas_matrix_cols)/ self.screen_width))
        #print(str.format("Coordinate ({0},{1}) hit area: [{2},{3}]", x, y, area_x, area_y))
        self.areas_hits_matrix[area_x][area_y] += 1

    def build_importance_matrix(self, total_moves):
        for i in range(self.areas_matrix_rows):
            for j in range(self.areas_matrix_cols):
                self.areas_importance_matrix[i][j] = round(self.areas_hits_matrix[i][j] / total_moves, 3)

    def print_matrix(self, matrix, rows, cols):
        for i in range(rows):
            for j in range(cols):
                print(str.format("{0} |", matrix[i][j]), end='')
            print("\n")

    def add_distance_measurement(self, x1, y1, x2, y2, buf):
        distance = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
        buf.append(round(distance,3))
        #print(str.format("distance : {0}",round(distance,3)))

    def add_angle_measurement(self, x1, y1, x2, y2, buf):
        angle = math.atan2(x1 - x2, y1 - y2) * (180 / math.pi)
        buf.append(round(angle,3))
        #print(str.format("angle : {0}", round(angle,3)))

    def calculate_avg_angle_per_second(self, angle_buf):
        num_of_angles = len(angle_buf)
        if num_of_angles == 0:
            return 0
        else:
            return sum(angle_buf) / num_of_angles

    def calculate_distance_per_second(self, distance_buf):
        return sum(distance_buf)

    def is_same_location(self, x1, y1, x2, y2):
        return x1 == x2 and y1 == y2

    def is_currently_active(self):
        return self.inactivity_counter < self.inactivity_threshold_samples

    def matrix_to_array(self, matrix):
        lst = []
        for row in matrix:
            lst.extend(row)
        return lst

    def clear_state(self):
        self.areas_hits_matrix = [[0 for y in range(self.areas_matrix_cols)] for x in range(self.areas_matrix_rows)]
        self.areas_importance_matrix = [[0 for y in range(self.areas_matrix_cols)] for x in range(self.areas_matrix_rows)]
        self.angle_vec_per_seconds.clear()
        self.distance_vec_per_seconds.clear()
        self.inactivity_counter = 0
