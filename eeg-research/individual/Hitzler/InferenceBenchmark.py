import statistics
import torch
import lightning as L

class InferenceBenchmark(L.Callback):
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.times = []

    def median_time(self):
        return statistics.median(self.times)

    def on_test_batch_start(self, trainer, *args, **kwargs):
        self.start.record()

    def on_test_batch_end(self, trainer, *args, **kwargs):
        self.end.record()
        self.end.synchronize()
        self.times.append(self.start.elapsed_time(self.end))