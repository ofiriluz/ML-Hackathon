from abc import abstractmethod


class ISanitizer:
    def __init__(self):
        pass

    @abstractmethod
    def sanitize_data(self, data):
        pass