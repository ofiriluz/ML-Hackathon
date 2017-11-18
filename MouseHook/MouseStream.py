from Interface.IDataStream import IDataStream
from MouseHook.MouseHook import MouseHook


class MouseStream(IDataStream):
    def __init__(self,
                 sliding_window_size=5000,
                 sample_interval_ms=10):
        super().__init__()
        self.sliding_window_size = sliding_window_size
        self.sample_interval_ms = sample_interval_ms
        self.mouse_hook = MouseHook(sliding_window_size, sample_interval_ms)

    def init_stream(self):
        self.mouse_hook.RunMouseHook()

    def get_next_stamped_data(self):
        return self.mouse_hook.GetData()

