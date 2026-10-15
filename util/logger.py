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
            print(f"[\x1b[1;38;5;27mDEBUG\033[0m] {message}")

    def INFO(self, message):
        if self.log_level <= 1:
            print(f"[\x1b[1;38;5;47mINFO\033[0m] {message}")

    def WARNING(self, message):
        if self.log_level <= 2:
            print(f"[\x1b[1;38;5;226mWARNING\033[0m] {message}")

    def ERROR(self, message):
        if self.log_level <= 3:
            print(f"[\x1b[1;38;5;215mERROR\033[0m] {message}")

    def CRITICAL(self, message):
        if self.log_level <= 4:
            print(f"[\x1b[1;38;5;203mCRITICAL\033[0m] {message}")


logger = Logger()


def configure_logger(log_level):
    logger.log_level = log_level

if __name__ == "__main__":
    #testing dummy data
    configure_logger(0)
    logger.DEBUG("This is a debug message.")
    logger.INFO("This is an info message.")
    logger.WARNING("This is a warning message.")
    logger.ERROR("This is an error message.")
    logger.CRITICAL("This is a critical message.")