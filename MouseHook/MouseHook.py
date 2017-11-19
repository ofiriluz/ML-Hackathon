import pynput
import datetime
import time
import ctypes
import math
import threading
from pynput.mouse import Listener


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

        #csvFile = open("Output\mousemove_" + str(datetime.datetime.now().timestamp()) + ".csv", "w+")

    def get_data(self):
        if len(self.mouseCoordsList) == 0:
            return None
        return self.mouseCoordsList.pop(0)

    # CSV format: x,y,timestamp
    def on_move(self, x, y):
        # Take moues position update every 10ms
        currSecond = time.time()
        millis = int(round(currSecond * 1000))
        if millis - self.lastMillis >= self.SampleIntervalMs:
            #now = datetime.datetime.now()
            #outputLine = str.format('{0},{1},{2}', x, y, int(round(now.timestamp()))) + "\n"
            if self.is_in_screen_bounds(x, y):
                print((x,y, millis))
                # Save to CSV file
                #csvFile.write(outputLine)
                if len(self.mouseCoordsList) >= self.SlidingWindowSize:
                    self.mouseCoordsList.pop(0)
                self.mouseCoordsList.append((x, y, millis))
            self.lastMillis = millis

    def is_in_screen_bounds(self, x, y):
        return 0 <= x < self.screenWidth and 0 <= y < self.screenHeight

    def listen(self):
        self.listener = Listener(on_move=self.on_move)
        self.listener.start()
        self.listener.join()

    def run_mouse_hook(self):
        self.listenerThread.start()

    def stop_mouse_hook(self):
        if self.listener:
            self.listener.stop()
