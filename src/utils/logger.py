import logging


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"rag_system.{name}")


def setup_logging(level: str = "INFO") -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)

    logging.getLogger().setLevel(log_level)
    logging.getLogger("rag_system").setLevel(log_level)
    logging.getLogger("uvicorn").setLevel(log_level)
    logging.getLogger("uvicorn.error").setLevel(log_level)
