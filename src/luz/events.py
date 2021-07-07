import enum

__all__ = ["Event"]


class Event(enum.Enum):
    BATCH_STARTED = "BATCH STARTED"
    BATCH_ENDED = "BATCH ENDED"
    EPOCH_STARTED = "EPOCH STARTED"
    EPOCH_ENDED = "EPOCH ENDED"
    TESTING_STARTED = "TESTING STARTED"
    TESTING_ENDED = "TESTING ENDED"
    TRAINING_STARTED = "TRAINING STARTED"
    TRAINING_ENDED = "TRAINING ENDED"
    VALIDATING_STARTED = "VALIDATING STARTED"
    VALIDATING_ENDED = "VALIDATING ENDED"

    def __call__(self, handlers, **kwargs):
        for h in handlers:
            getattr(h, self.name.lower())(**kwargs)
