from pytorch_lightning import Callback
from datetime import datetime


class ScheduledStopCallback(Callback):
    def __init__(self, stop_time):
        self.stop_time = None
        if stop_time:
            self.stop_time = datetime.strptime(stop_time, "%Y-%m-%d %H:%M:%S")
            print(f"Scheduled stop at: {self.stop_time}.")

    def on_train_batch_end(self, trainer, *args, **kwargs):
        self.stop_if_time_reached(trainer)

    def on_test_batch_end(self, trainer, *args, **kwargs):
        self.stop_if_time_reached(trainer)

    def on_validation_batch_end(self, trainer, *args, **kwargs):
        self.stop_if_time_reached(trainer)

    def on_predict_batch_end(self, trainer, *args, **kwargs):
        self.stop_if_time_reached(trainer)

    def stop_if_time_reached(self, trainer):
        if self.stop_time and datetime.now() >= self.stop_time:
            print("Scheduled stop time reached.")
            trainer.should_stop = True
