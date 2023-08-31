import logging
from datetime import datetime
from pytz import timezone


def create_file_logger(name: str, filename: str, level: int):
    file_handler = logging.FileHandler(filename, mode="w")
    formatter = logging.Formatter("%(asctime)s %(message)s")
    formatter.converter = lambda *args: datetime.now(tz=timezone('US/Pacific')).timetuple()

    file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.addHandler(file_handler)

    logger.setLevel(level)

    return logger
