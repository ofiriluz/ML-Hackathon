from Interface.IDataStream import IDataStream
from pynput import keyboard
from threading import Thread
from datetime import datetime


class KeyboardDataStream(IDataStream):
    def __init__(self, window_size=1000):
        super().__init__()
        self.records_window = []
        self.window_size = window_size
        self.listener = None
        self.last_time = datetime.now()

    def __listener_thread(self):
        self.listener.join()

    def __on_key_press(self, key):
        pass

    def init_stream(self):
        self.listener = keyboard.Listener(on_press=self.__on_key_press)


    def get_next_stamped_data(self):
        if len(self.records_window) == 0:
            return None
        return self.records_window.pop(0)
