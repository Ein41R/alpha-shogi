"""
Logger class responsible for logging depending on Log Levels
Log levels are the following:
    0: DEBUG
    1: INFO
    2: WARN
    3: ERROR
    4: CRITICAL
    >=5: None
"""

from unittest import case

from config.miscConfig import (
    RESET,
    LOG_DEBUG,
    LOG_INFO,
    LOG_WARN,
    LOG_ERROR,
    LOG_CRITICAL,
)



class Logger():
    def __init__(self, log_level=-1):
        self.log_level = log_level

    def DEBUG(self, message):
        if self.log_level <= 0:
            print(f"[{LOG_DEBUG}DEBUG{RESET}] {message}")

    def INFO(self, message):
        if self.log_level <= 1:
            print(f"[{LOG_INFO}INFO{RESET}] {message}")

    def WARN(self, message):
        if self.log_level <= 2:
            print(f"[{LOG_WARN}WARN{RESET}] {message}")

    def ERROR(self, message):
        if self.log_level <= 3:
            print(f"[{LOG_ERROR}ERROR{RESET}] {message}")

    def CRITICAL(self, message):
        if self.log_level <= 4:
            print(f"[{LOG_CRITICAL}CRITICAL{RESET}] {message}")


logger = Logger()


def configure_logger(log_level):
    if type(log_level) is int:
        logger.log_level = log_level
    elif type(log_level) is str:
        log_level = log_level.upper()
        if log_level == "DEBUG":
            logger.log_level = 0
        elif log_level == "INFO":
            logger.log_level = 1
        elif log_level == "WARN":
            logger.log_level = 2
        elif log_level == "ERROR":
            logger.log_level = 3
        elif log_level == "CRITICAL":
            logger.log_level = 4
        else:
            raise ValueError(f"Invalid log level: {log_level}")
    else:
        logger.log_level = -1

if __name__ == "__main__":
    #testing dummy data
    configure_logger(0)
    logger.DEBUG("This is a debug message.")
    logger.INFO("This is an info message.")
    logger.WARN("This is a WARN message.")
    logger.ERROR("This is an error message.")
    logger.CRITICAL("This is a critical message.")