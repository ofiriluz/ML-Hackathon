import datetime


class Stamp:
    def __init__(self, source="", time_stamp=datetime.datetime.now()):
        self.source = source
        self.time_stamp = time_stamp

    def set_time(self, time):
        if not isinstance(time, datetime.datetime):
            return
        self.time_stamp = time

    def set_source(self, source):
        self.source = source

    def get_source(self):
        return self.source

    def get_time(self):
        return self.time_stamp
