import logging

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "not-installed"


class ParallelLogger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Avoid adding duplicate handlers
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)

            # Use datefmt for uniform timestamps
            formatter = logging.Formatter(
                "[%(asctime)s: %(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        # Prevent propagation to avoid duplicate logs from the root logger
        self.logger.propagate = False

        self.is_root = False

    def set_rank(self, rank):
        self.is_root = rank == 0

    def info(self, msg):
        if self.is_root:
            self.logger.info(msg)

    def error(self, msg):
        # Always log errors, regardless of rank
        self.logger.error(msg)


logger = ParallelLogger()
