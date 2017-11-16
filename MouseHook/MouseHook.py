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
        self.listenerThread = threading.Thread(target=self.Listen)
        self.totalMoves = 0
        self.mouseCoordsList = []
        self.lastMeasuredSecond = int(round(time.time())) # Measure current time in seconds
        self.lastMillis = 0

        # Get screen dimensions for division to areas
        user32 = ctypes.windll.user32
        self.screenWidth = user32.GetSystemMetrics(0)
        self.screenHeight = user32.GetSystemMetrics(1)

        #csvFile = open("Output\mousemove_" + str(datetime.datetime.now().timestamp()) + ".csv", "w+")

    def GetData(self):
        return self.mouseCoordsList.pop(0)

    # CSV format: x,y,timestamp
    def on_move(self, x, y):
        # Take moues position update every 10ms
        currSecond = time.time()
        millis = int(round(currSecond * 1000))
        if millis - self.lastMillis >= self.SampleIntervalMs:
            #now = datetime.datetime.now()
            #outputLine = str.format('{0},{1},{2}', x, y, int(round(now.timestamp()))) + "\n"
            if self.IsInScreenBounds(x, y):
                print((x,y, millis))
                # Save to CSV file
                #csvFile.write(outputLine)
                if (self.mouseCoordsList.count() >= self.SlidingWindowSize):
                    self.mouseCoordsList.pop(0)
                self.mouseCoordsList.append((x, y, millis))
            self.lastMillis = millis

    def IsInScreenBounds(self, x, y):
        return (x >= 0 and y >= 0 and x < self.screenWidth and y < self.screenHeight)

    def Listen(self):
        # Collect events until released
        with Listener(
                on_move=self.on_move,
                on_click=self.on_click,
                on_scroll=self.on_scroll) as listener:
            listener.join()

    def RunMouseHook(self):
        self.listenerThread.start()
