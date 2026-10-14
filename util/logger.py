"""
Logger class responsible for logging depending on Log Levels
Log levels are the following:
    0: DEBUG
    1: INFO
    2: WARNING
    3: ERROR
    4: CRITICAL
    >=5: None
"""


class Logger():
    def __init__(self, log_level=-1):
        self.log_level = log_level

    def DEBUG(self, message):
        if self.log_level <= 0:
            print(f"[DEBUG] {message}")

    def INFO(self, message):
        if self.log_level <= 1:
            print(f"[INFO] {message}")

    def WARNING(self, message):
        if self.log_level <= 2:
            print(f"[WARNING] {message}")

    def ERROR(self, message):
        if self.log_level <= 3:
            print(f"[ERROR] {message}")

    def CRITICAL(self, message):
        if self.log_level <= 4:
            print(f"[CRITICAL] {message}")


logger = Logger()


def configure_logger(log_level):
    logger.log_level = log_level