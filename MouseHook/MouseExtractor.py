from Interface.IFeatureExtractor import IFeatureExtractor
import ctypes
import math

class MouseExtractor(IFeatureExtractor):
    def __init__(self, data_chunk_duration_sec = 30):
        super().__init__()
        self.data_chunk_duration_sec = data_chunk_duration_sec

        # Get screen dimensions for division to areas
        user32 = ctypes.windll.user32
        self.screen_width = user32.GetSystemMetrics(0)
        self.screen_height = user32.GetSystemMetrics(1)

        self.areas_matrix_rows, self.areas_matrix_cols, = 4, 6;
        self.areas_hits_matrix = []
        self.areas_importance_matrix = []

    def can_be_extracted(self, data):
        first_timestamp = int(round(data[0][2] / 1000))
        last_timestamp = int(round(data[data.count() - 1][2]))
        if (last_timestamp - first_timestamp >= self.data_chunk_duration_sec):
            return True
        return False

    def extract_features(self, data_vector):
        total_moves = 0
        for (x,y, millis) in self.data_vector:
            self.update_area_hit_count(x,y)
            total_moves += 1
        self.build_importance_matrix(total_moves)
        # Temp: print matrices:
        print("\n\nAreas Hits:\n")
        self.print_matrix(self.areas_hits_matrix, self.areas_matrix_rows, self.areas_matrix_cols)
        print("\n\nAreas Importance:\n")
        self.print_matrix(self.areas_importance_matrix, self.areas_matrix_rows, self.areas_matrix_cols)
        areas_importance_list =  self.matrix_to_array(self.areas_importance_matrix)
        print(areas_importance_list)

    def update_area_hit_count(self, x, y):
        area_x = int(math.floor((y * self.areas_matrix_rows) / self.screen_height))
        area_y = int(math.floor((x * self.areas_matrix_cols)/ self.screen_width))
        print(str.format("Coordinate ({0},{1}) hit area: [{2},{3}]", x, y, area_x, area_y))
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

    def matrix_to_array(self, matrix, rows, cols):
        lst = []
        for i in range(rows):
            for j in range(cols):
                lst.append(matrix[i][j])
        return lst

    def reset_state(self):
        self.areas_hits_matrix = [[0 for y in range(self.areas_matrix_cols)] for x in range(self.areas_matrix_rows)]
        self.areas_importance_matrix = [[0 for y in range(self.areas_matrix_cols)] for x in range(self.areas_matrix_rows)]
