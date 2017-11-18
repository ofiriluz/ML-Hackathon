from Interface.IDataStream import IDataStream
import os


class DataReaderStream(IDataStream):
    def __init__(self, data_folder=""):
        super().__init__()
        self.data_folder = data_folder
        self.current_file = None
        self.current_lines = []
        self.files = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]

    def __open_next_file(self):
        self.current_file = None
        self.current_line = []
        if len(self.files) > 0:
            self.current_file = open(self.data_folder + "/" + self.files.pop(0), 'r')
            self.current_lines = self.current_file.readlines()

    def init_stream(self):
        self.__open_next_file()

    def get_next_stamped_data(self):
        if len(self.current_lines) == 0:
            self.__open_next_file()
        if not self.current_file:
            return None
        line = self.current_lines.pop(0)
        line = [float(item) for item in line.split(',')]
        return line
