from abc import abstractmethod


class IDataStream:
    def __init__(self):
        self._is_streaming = False

    @abstractmethod
    def init_stream(self):
        pass

    @abstractmethod
    def stop_stream(self):
        pass

    @abstractmethod
    def get_next_stamped_data(self):
        pass
