import logging

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "not-installed"


class ParallelLogger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter("ptrl %(asctime)s - %(message)s")

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        self.logger.addHandler(ch)

        self.is_root = None

    def set_rank(self, rank):
        self.is_root = rank == 0

    def info(self, msg):
        if self.is_root:
            self.logger.info(msg)

    def error(self, msg):
        return self.logger.error(msg)


logger = ParallelLogger()
