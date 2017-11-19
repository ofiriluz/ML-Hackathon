import win32gui
import datetime
import time
import ctypes
import math
import threading

class MouseHook:
    def __init__(self,
                 sliding_window_size=5000,
                 sample_interval_ms=10,
                 ):
        # Init from ctor
        self.SlidingWindowSize = sliding_window_size
        self.SampleIntervalMs = sample_interval_ms

        # Init internally
        self.listenerThread = threading.Thread(target=self.listen)
        self.totalMoves = 0
        self.mouseCoordsList = []
        self.lastMeasuredSecond = int(round(time.time())) # Measure current time in seconds
        self.lastMillis = 0
        self.listener = None

        # Get screen dimensions for division to areas
        user32 = ctypes.windll.user32
        self.screenWidth = user32.GetSystemMetrics(0)
        self.screenHeight = user32.GetSystemMetrics(1)
        self.isRunning = False

    def get_data(self):
        if len(self.mouseCoordsList) == 0:
            return None
        return self.mouseCoordsList.pop(0)

    def recordMousePosition(self, x, y, millis):
        if self.is_in_screen_bounds(x, y):
            if len(self.mouseCoordsList) >= self.SlidingWindowSize:
                self.mouseCoordsList.pop(0)
            self.mouseCoordsList.append((x, y, millis))

    def is_in_screen_bounds(self, x, y):
        return 0 <= x < self.screenWidth and 0 <= y < self.screenHeight

    def listen(self):
        while self.isRunning:
            currSecond = time.time()
            millis = int(round(currSecond * 1000))
            if millis - self.lastMillis >= 10:
                (x, y) = win32gui.GetCursorPos()
                #print((x, y, millis))
                self.recordMousePosition(x, y, millis)
                self.lastMillis = millis

    def run_mouse_hook(self):
        self.isRunning = True
        self.listenerThread.start()

    def stop_mouse_hook(self):
        self.isRunning = False
