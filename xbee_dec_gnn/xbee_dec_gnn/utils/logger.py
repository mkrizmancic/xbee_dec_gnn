import logging
from datetime import datetime

from colorlog import ColoredFormatter


class MicrosecondColoredFormatter(ColoredFormatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat(sep=" ")


class ObjectWithLogger:
    def __init__(self, logger_name=None):
        formatter = MicrosecondColoredFormatter(
            "%(log_color)s%(asctime)s %(levelname)-8s%(reset)s %(message)s",
            datefmt="%H:%M:%S.%f",
            reset=True,
            log_colors={
                "DEBUG": "green",
                "INFO": "cyan",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red",
            },
        )

        self.logger = logging.getLogger(logger_name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

    def get_logger(self):
        return self.logger