import logging
import os


def make_logger(file_: str = "NO_FILE") -> logging.Logger:
    log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO"))
    fmt = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    logging.basicConfig(level=log_level, format=fmt)
    return logging.getLogger(file_.split("/")[-1])