"""Request checkpointed stopping at the next complete PPO update."""
import signal


class TrainingStop:
    def __enter__(self):
        self.reason = None
        self.previous = {kind: signal.getsignal(kind) for kind in (signal.SIGTERM, signal.SIGINT)}
        for kind in self.previous:
            signal.signal(kind, self.request)
        return self

    def request(self, kind, _frame):
        self.reason = 'signal_' + signal.Signals(kind).name

    def __exit__(self, *_):
        for kind, handler in self.previous.items():
            signal.signal(kind, handler)
