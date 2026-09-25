import logging

from utils.logger import get_logger, setup_logging


def test_get_logger():
    logger = get_logger("test")

    assert logger.name == "rag_system.test"


def test_setup_logging():
    root_logger = logging.getLogger()
    original_level = root_logger.level

    try:
        setup_logging("DEBUG")

        assert root_logger.level == logging.DEBUG
    finally:
        root_logger.setLevel(original_level)
