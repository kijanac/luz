from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["ConsoleLogger", "FileLogger", "Logger"]


class Logger(ABC):
    @abstractmethod
    def log(self, msg: str) -> None:
        pass


class ConsoleLogger(Logger):
    """Logs messages to console."""

    def log(self, msg: str) -> None:
        print(msg)


class FileLogger(Logger):
    """Logs messages to file.

    Parameters
    ----------
    filepath
        Path to log file.
    """

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath

    def log(self, msg: str) -> None:
        with open(self.filepath, "a") as f:
            f.write(f"{msg}\n")
