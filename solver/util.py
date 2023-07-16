import logging
import sys

import numpy as np


def setup_basic_logging(**kwargs):
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s\t%(message)s',
                        datefmt='%Y-%m-%d %X',
                        level=logging.INFO,
                        **kwargs)


def point_line_dist(px, pa, pb):
    """Distance between point x and line defined by two points: pa, pb"""
    px = np.asarray(px)
    pa = np.asarray(pa)
    pb = np.asarray(pb)
    return abs(np.cross(pb-pa, px-pa) / np.linalg.norm(pb-pa))
