from time import time


class Stamp:
    def __init__(self, source="", time_stamp=time(), user=""):
        self.source = source
        self.time_stamp = time_stamp
        self.user = user

    def set_time(self, stamp_time):
        self.time_stamp = stamp_time

    def set_source(self, source):
        self.source = source

    def set_user(self, user):
        self.user = user

    def get_source(self):
        return self.source

    def get_time(self):
        return self.time_stamp

    def get_user(self):
        return self.user
