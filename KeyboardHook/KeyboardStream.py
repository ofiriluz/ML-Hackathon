from Interface.IDataStream import IDataStream
from pynput import keyboard
from threading import Thread
from time import time


class KeyboardDataStream(IDataStream):
    def __init__(self, window_size=1000):
        super().__init__()
        self.records_window = []
        self.window_size = window_size
        self.listener = None

    def __listener_thread(self):
        self.listener.join()

    def __on_key_press(self, key):
        if len(self.records_window) == self.window_size:
            self.records_window.pop(0)
        self.records_window.append((key, int(round(time()))*1000))

    def init_stream(self):
        self.listener = keyboard.Listener(on_press=self.__on_key_press)
        Thread(target=self.__listener_thread())

    def get_next_stamped_data(self):
        return self.records_window.pop(0)
