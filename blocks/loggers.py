import io
import logging
import pymupdf

def keep_only_valid_metadata(metadata: dict) -> dict:
    return {k:v for k,v in metadata.items() if v}

class StreamValue:
    def __init__(self, handler: logging.StreamHandler):
        self._handler = handler

    def value(self):
        self._handler.flush()
        value = self._handler.stream.getvalue()
        self._handler.stream.truncate(0)
        self._handler.stream.seek(0)
        return value

class LoggerStream():
    def __init__(self, loggers: logging.Logger | list[logging.Logger]):
        pymupdf.set_messages(pylogging=True)
        if isinstance(loggers, logging.Logger):
            loggers = [loggers]
        self._loggers = loggers
        self._handler = None

    @property
    def handler(self):
        if self._handler is None:
            self._handler = logging.StreamHandler(io.StringIO())
            for logger in self._loggers:
                # Remove all existing handlers
                for h in logger.handlers:
                    logger.removeHandler(h)

                logger.addHandler(self._handler)
                # Don't propagate to root logger 
                logger.propagate = False
                # Set level to DEBUG
                logger.setLevel(logging.INFO)
        return self._handler

    def __enter__(self):
        self.handler.flush()
        return StreamValue(self.handler)

    def __exit__(self, exc_type, exc_value, traceback):
        pass