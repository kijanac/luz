import enum

__all__ = ["Flag"]


class Flag(enum.Enum):
    TRAINING = "TRAINING"
    TESTING = "TESTING"
