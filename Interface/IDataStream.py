from abc import abstractmethod


class IDataStream:
    def __init__(self):
        self._is_streaming = False

    def start_stream(self, cb):
        self._is_streaming = True
        while self._is_streaming:
            data = self.get_stamped_data()
            if data:
                cb(data)

    def stop_stream(self):
        self._is_streaming = False

    @abstractmethod
    def get_stamped_data(self):
        pass
