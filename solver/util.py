import logging
import sys

def setup_basic_logging(**kwargs):
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s\t%(message)s',
                        datefmt='%Y-%m-%d %X',
                        level=logging.INFO,
                        **kwargs)
