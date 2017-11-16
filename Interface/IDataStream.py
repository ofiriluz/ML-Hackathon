from abc import abstractmethod


class IDataStream:
    def __init__(self):
        self._is_streaming = False

    @abstractmethod
    def get_next_stamped_data(self):
        pass
