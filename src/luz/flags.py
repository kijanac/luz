import enum

__all__ = ["Flag"]


class Flag(enum.Enum):
    TRAINING = "TRAINING"
    VALIDATING = "VALIDATING"
    TESTING = "TESTING"
