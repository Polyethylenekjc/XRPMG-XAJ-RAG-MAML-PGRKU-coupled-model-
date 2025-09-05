import time
import colorama
from colorama import Fore, Style

colorama.init()

class LogLevel:
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

class Logger:
    _instance = None
    _log_file = None
    LOG_LEVEL = LogLevel.INFO
    LEVELS = {
        LogLevel.DEBUG: 0,
        LogLevel.INFO: 1,
        LogLevel.WARNING: 2,
        LogLevel.ERROR: 3
    }

    @classmethod
    def set_level(cls, level: str) -> None:
        cls.LOG_LEVEL = level.upper()

    @classmethod
    def initialize(cls, log_file = None) -> None:
        cls._instance = cls()
        if cls._log_file:
            cls._log_file.close()
        if log_file:
            try:
                cls._log_file = open(log_file, 'w', encoding='utf-8')
            except Exception as e:
                print(f"Failed to open log file: {e}")

    @classmethod
    def log(cls, message: str, level: str = "INFO", module = None) -> None:
        if cls.LEVELS.get(level.upper(), 0) < cls.LEVELS.get(cls.LOG_LEVEL, 1):
            return

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        level_upper = level.upper()

        color_map = {
            LogLevel.DEBUG: Fore.BLUE,
            LogLevel.INFO: Fore.GREEN,
            LogLevel.WARNING: Fore.YELLOW,
            LogLevel.ERROR: Fore.RED
        }
        color = color_map.get(level_upper, Fore.WHITE)

        if module:
            header = f"[{timestamp}] [{level_upper}] [{module}]"
        else:
            header = f"[{timestamp}] [{level_upper}]"

        full_msg = f"{header} {message}"

        print(f"{color}{header}{Style.RESET_ALL} {message}")

        if cls._log_file:
            try:
                cls._log_file.write(full_msg + "\n")
                cls._log_file.flush()
            except Exception as e:
                print(f"Error writing to log file: {e}")

    @classmethod
    def debug(cls, message: str, module = None) -> None:
        cls.log(message, level=LogLevel.DEBUG, module=module)

    @classmethod
    def info(cls, message: str, module = None) -> None:
        cls.log(message, level=LogLevel.INFO, module=module)

    @classmethod
    def warning(cls, message: str, module = None) -> None:
        cls.log(message, level=LogLevel.WARNING, module=module)

    @classmethod
    def error(cls, message: str, module = None) -> None:
        cls.log(message, level=LogLevel.ERROR, module=module)

    @classmethod
    def close(cls) -> None:
        if cls._log_file:
            cls._log_file.close()